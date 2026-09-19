"""Standalone TM-score fitness for the sequence genome: NOT wired into the GA.

    genome string -> ESMFold predictor (esmfold/fitness.py) -> coordinates (PDB)
    -> TM-align against the fixed reference esmfold/pdb_cache/7UR7.pdb -> float

The returned fitness is the TM-score normalised by the REFERENCE's length (63
resolved residues), so candidates of different lengths are on one scale. That
is TM-align's second "TM-score=" line when the query is Chain_1 and the
reference is Chain_2, and the module checks it really is: TM-align's reported
Chain_2 length must equal the reference's own CA count, and Chain_1's must equal
the genome length (a fold that lost residues raises instead of scoring).

The aligned length and the aligned-region RMSD are diagnostics only: logged
through the `esmfold.tm_fitness` logger at INFO and kept on the ScoreResult
that score_detailed() returns. score() returns the TM-score and nothing else.

The predictor is loaded once, when a TMFitness is built, and reused for every
call. A process-wide default (get_default / tm_fitness) builds one lazily. ESMFold
in fp32 is ~13.7GB and the GPU holds one copy, so a process that also uses
hpga.sequence_model's own ESMFoldFitness singleton should pass THAT object as
`predictor=` rather than letting this module load a second one.

TM-align binary: $TMALIGN_BIN, default /scratch/pcanaste/bin/TMalign (built
from esmfold/tmalign/TMalign.cpp; see the sanity-check output for the path and
hash actually used).
"""

import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_PDB = HERE / "pdb_cache" / "7UR7.pdb"
TMALIGN_BIN = Path(os.environ.get("TMALIGN_BIN", "/scratch/pcanaste/bin/TMalign"))
TMALIGN_TIMEOUT_S = 60

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
_THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

log = logging.getLogger(__name__)

_LEN1 = re.compile(r"Length of Chain_1:\s*(\d+)")
_LEN2 = re.compile(r"Length of Chain_2:\s*(\d+)")
_ALIGNED = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)")
_TM = re.compile(r"TM-score=\s*([\d.]+)")


def reference_sequence(pdb_path: Path = REFERENCE_PDB) -> str:
    """One-letter sequence of the reference's C-alpha atoms (read from the file,
    so 'the reference's own sequence' has a single source of truth)."""
    residues = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            residues.append(_THREE_TO_ONE[line[17:20].strip()])
    return "".join(residues)


@dataclass(frozen=True)
class ScoreResult:
    tm_score: float  # normalised by the reference length: the fitness
    aligned_length: int  # diagnostic
    rmsd: float  # diagnostic: RMSD over the aligned region
    genome_length: int
    reference_length: int
    predictor_s: float  # ESMFold fold_to_pdb wall time
    tmalign_s: float  # PDB temp-file write + TM-align subprocess wall time


class TMFitness:
    def __init__(self, predictor=None, tmalign_bin: Path = TMALIGN_BIN, reference_pdb: Path = REFERENCE_PDB):
        if not Path(tmalign_bin).exists():
            raise FileNotFoundError(f"TM-align binary not found at {tmalign_bin} (set TMALIGN_BIN)")
        self.tmalign_bin, self.reference_pdb = Path(tmalign_bin), Path(reference_pdb)
        self.reference_length = len(reference_sequence(self.reference_pdb))
        if predictor is None:
            from esmfold.fitness import ESMFoldFitness  # lazy: importing torch/transformers is not free

            predictor = ESMFoldFitness()  # loads ESMFold once, here
        self.predictor = predictor

    def score_detailed(self, genome: str) -> ScoreResult:
        if not isinstance(genome, str) or not genome:
            raise ValueError("genome must be a non-empty string")
        bad = sorted(set(genome) - CANONICAL)
        if bad:
            raise ValueError(f"genome has non-canonical letters {bad}; expected only {''.join(sorted(CANONICAL))}")

        t0 = time.perf_counter()
        pdb_str = self.predictor.fold_to_pdb(genome)  # ESMFold, already loaded; the coordinates
        t1 = time.perf_counter()
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        try:
            tf.write(pdb_str)
            tf.close()
            proc = subprocess.run(
                [str(self.tmalign_bin), tf.name, str(self.reference_pdb)],  # query = Chain_1, reference = Chain_2
                capture_output=True, text=True, timeout=TMALIGN_TIMEOUT_S,
            )
        finally:
            Path(tf.name).unlink(missing_ok=True)
        t2 = time.perf_counter()

        out = proc.stdout
        m1, m2, ma, tms = _LEN1.search(out), _LEN2.search(out), _ALIGNED.search(out), _TM.findall(out)
        if proc.returncode != 0 or not (m1 and m2 and ma) or len(tms) != 2:
            raise RuntimeError(f"could not parse TM-align output (rc={proc.returncode}):\n{out}\n{proc.stderr}")
        if int(m2.group(1)) != self.reference_length:
            raise RuntimeError(f"TM-align read Chain_2 as {m2.group(1)} residues, reference has {self.reference_length}")
        if int(m1.group(1)) != len(genome):
            raise RuntimeError(f"predicted structure has {m1.group(1)} residues for a {len(genome)}-letter genome")

        result = ScoreResult(
            tm_score=float(tms[1]),  # 2nd line = normalised by Chain_2 = the reference
            aligned_length=int(ma.group(1)), rmsd=float(ma.group(2)),
            genome_length=len(genome), reference_length=self.reference_length,
            predictor_s=t1 - t0, tmalign_s=t2 - t1,
        )
        log.info("TM=%.5f aligned_length=%d rmsd=%.2f genome_length=%d predictor_s=%.3f tmalign_s=%.3f",
                 result.tm_score, result.aligned_length, result.rmsd, result.genome_length,
                 result.predictor_s, result.tmalign_s)
        return result

    def score(self, genome: str) -> float:
        """The fitness: TM-score vs the reference, normalised by reference length."""
        return self.score_detailed(genome).tm_score


_default: TMFitness | None = None


def get_default() -> TMFitness:
    global _default
    if _default is None:
        _default = TMFitness()
    return _default


def tm_fitness(genome: str) -> float:
    """genome string -> TM-score against 7UR7 (reference-length normalised)."""
    return get_default().score(genome)
