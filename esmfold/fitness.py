"""ESMFold + TM-align fitness function against a fixed real-protein target.

Target: PDB 7UR7 chain A, resolved core (63 of 70 SEQRES residues -- the
C-terminal His-tag is disordered and unresolved). A de novo designed small
beta-barrel (see RESULTS.md and the FoldBench candidate selection): all
canonical amino acids, no metal/ligand/disulfide dependency, so a sequence
alone is sufficient to reach its fold -- unlike 5SBJ (non-canonical caps,
Cd-dependent) or the disulfide-stabilized defensin, which were ruled out
for that reason.

Folds a candidate sequence with ESMFold, writes it to a PDB, then scores it
against pdb_cache/7UR7.pdb with TM-align, normalized by the REFERENCE's
length (TM-align's own printed recommendation) so score is comparable
across candidate sequences of different lengths.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

HERE = Path(__file__).parent
TARGET_PDB = HERE / "pdb_cache" / "7UR7.pdb"
TARGET_SEQ = "SEVKELLEEFLKRNKPVRIHHKNGEEIKVRITHIGEDTVEFELNGRTHRINIKDILDVKEWLE"
TMALIGN_BIN = HERE / "tmalign" / "TMalign"


def tm_score(query_pdb: Path, reference_pdb: Path) -> float:
    """TM-score of query vs reference, normalized by the reference length."""
    out = subprocess.run(
        [str(TMALIGN_BIN), str(query_pdb), str(reference_pdb)],
        capture_output=True, text=True, timeout=60,
    ).stdout
    scores = re.findall(r"TM-score=\s*([\d.]+)", out)
    if len(scores) != 2:
        raise RuntimeError(f"could not parse TM-align output:\n{out}")
    return float(scores[1])  # 2nd line = normalized by Chain_2 = reference


class ESMFoldFitness:
    """Loads ESMFold once (fp32 -- the only dtype found to work, see
    RESULTS.md sec 4); call .score(sequence) per candidate."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True, torch_dtype=dtype,
        ).cuda().eval()

    def fold_to_pdb(self, sequence: str) -> str:
        with torch.no_grad():
            return self.model.infer_pdb(sequence)

    def score(self, sequence: str, keep_pdb_path: Path | None = None) -> float:
        pdb_str = self.fold_to_pdb(sequence)
        if keep_pdb_path is not None:
            keep_pdb_path.write_text(pdb_str)
            query_path = keep_pdb_path
        else:
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
            tf.write(pdb_str)
            tf.close()
            query_path = Path(tf.name)
        try:
            return tm_score(query_path, TARGET_PDB)
        finally:
            if keep_pdb_path is None:
                query_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    seq = sys.argv[1] if len(sys.argv) > 1 else TARGET_SEQ
    print(f"scoring sequence ({len(seq)} aa) against 7UR7 target...", flush=True)
    fitness = ESMFoldFitness()
    out_path = HERE / "fitness_test_out.pdb"
    score = fitness.score(seq, keep_pdb_path=out_path)
    print(f"TM-score vs 7UR7 target: {score:.4f}")
    print(f"predicted structure written to {out_path}")
