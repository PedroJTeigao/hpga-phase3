"""3D HP lattice protein folding model (simple cubic lattice): genome ->
walk -> fitness.

Migrated from Phase 1's 2D square-lattice model (see `../Phase 1/hpga/hp_model.py`,
byte-identical to this file's pre-migration state) to the 3D simple cubic
lattice, per advisor requirement. This is the *plain* 3D HP model
(backbone-only contacts) -- a step closer to the HP side-chain model (HPSC,
Benitez & Lopes 2009) used by Xue et al. (NoCS 2014, this project's base
paper) than the 2D model was, but not the same thing: HPSC adds a second
lattice point per residue for the side chain and a different contact/energy
scheme. See `docs/project_notes.md` (or the Phase 2 README) for the
validation benchmark and citation.

Genome encoding: relative moves in {STRAIGHT, LEFT, RIGHT, UP, DOWN} applied
to the current orientation frame, one move per residue after the second.
Six lattice directions are reachable from any cell; the sixth (BACKWARD,
directly reversing the current heading) is not representable, for the same
reason the 2D model has 3 symbols instead of 4 -- a backward move always
places the new residue on top of residue i-1, a guaranteed collision, so
excluding it from the encoding removes that failure mode by construction
rather than by runtime rejection. The walk can still self-intersect
elsewhere on the lattice; that is checked and penalised, not forbidden.

Orientation frame: (f, u) -- forward and "up" unit vectors, both always one
of the 6 axis-aligned directions (+-x, +-y, +-z); right r = f x u is
derived, not stored. All four turns are exact 90-degree rotations of the
frame implemented as integer cross products (v_rot = a x v for v
perpendicular to rotation axis a, the closed form of Rodrigues' formula at
theta=90 degrees) -- no floating point, so the frame never drifts off the
axis-aligned lattice directions:

  STRAIGHT: f'=f,            u'=u          (position += f)
  LEFT:     f'=cross(u,f),   u'=u          (yaw, about the up axis)
  RIGHT:    f'=cross(f,u),   u'=u          (yaw, opposite direction)
  UP:       f'=cross(r,f),   u'=cross(r,u) (pitch, about the right axis)
  DOWN:     f'=cross(f,r),   u'=cross(u,r) (pitch, opposite direction)

  where r = cross(f, u) is recomputed from the frame in effect *before* the
  move (only needed for UP/DOWN).

This is the standard reduced 3D relative-move encoding used in 3D HP-model
GA literature (5 symbols per step instead of the raw 6 lattice directions,
generalising the 2D "no U-turn" reduction) -- see the Phase 2 README for
where this specific formulation was checked against.

Note on sign conventions: nothing here fixes LEFT vs. RIGHT or UP vs. DOWN
to a "true" physical handedness -- that's a free choice, not something an
external reference could validate. It doesn't matter for correctness. HP
contact counting (fitness_from_coords below) depends only on the walk's
shape -- which residues end up lattice-adjacent to which -- and that
adjacency structure is invariant under any global rotation or reflection of
the coordinate system, including a "flip every LEFT/RIGHT" relabelling.
What must be correct (and is checked -- see the Phase 2 README's "3D
migration" section) is that turns compose consistently step to step and
that the resulting shape is genuinely self-avoiding-or-not as computed, not
that any one move code points along a specific labelled axis.
"""

from typing import Sequence

STRAIGHT, LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3, 4
MOVES = (STRAIGHT, LEFT, RIGHT, UP, DOWN)

Vec3 = tuple[int, int, int]

_INITIAL_FORWARD: Vec3 = (1, 0, 0)
_INITIAL_UP: Vec3 = (0, 0, 1)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _turn(f: Vec3, u: Vec3, move: int) -> tuple[Vec3, Vec3]:
    if move == STRAIGHT:
        return f, u
    if move == LEFT:
        return _cross(u, f), u
    if move == RIGHT:
        return _cross(f, u), u
    if move == UP:
        r = _cross(f, u)
        return _cross(r, f), _cross(r, u)
    if move == DOWN:
        r = _cross(f, u)
        return _cross(f, r), _cross(u, r)
    raise ValueError(f"invalid move code: {move}")


def genome_to_coords(genome: Sequence[int]) -> list[Vec3]:
    """Build lattice coordinates for a genome of length n-2 (n residues)."""
    f, u = _INITIAL_FORWARD, _INITIAL_UP
    coords = [(0, 0, 0), f]
    for move in genome:
        f, u = _turn(f, u, move)
        coords.append(_add(coords[-1], f))
    return coords


_NEIGHBOR_OFFSETS: tuple[Vec3, ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def fitness_from_coords(coords: Sequence[Vec3], sequence: str, collision_penalty: float = 50.0) -> float:
    """Same scoring rule as evaluate_fitness, applied directly to a
    coordinate list instead of deriving one from a genome via
    genome_to_coords. Factored out so external, independently-produced
    structures (e.g. a benchmark's published optimal structure, converted
    to coordinates through some other tool's own move convention) can be
    scored through the exact same contact/collision code the GA's fitness
    calls run — not a hand-reimplementation of it — as a correctness check
    against a known answer. See experiments/verify_3d_benchmark_structure.py.
    """
    n = len(coords)
    assert n == len(sequence)

    # Single O(n) pass builds both the collision count and the position
    # index used for the contact scan, so the cost of evaluating one
    # individual doesn't depend on whether the fold happens to be valid —
    # otherwise, for long random walks (which collide almost always), every
    # evaluation would take the cheap early-exit path and T_calc would stop
    # reflecting real per-individual cost.
    pos_index: dict[Vec3, int] = {}
    n_collisions = 0
    for i, c in enumerate(coords):
        if c in pos_index:
            n_collisions += 1
        pos_index[c] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != "H":
            continue
        x, y, z = coords[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = pos_index.get((x + dx, y + dy, z + dz))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1

    if n_collisions > 0:
        return -collision_penalty - n_collisions
    return float(contacts)


def evaluate_fitness(genome: Sequence[int], sequence: str, collision_penalty: float = 50.0) -> float:
    """Higher is better. Valid folds score = number of non-consecutive H-H
    lattice-adjacent contacts (>= 0). Invalid (self-intersecting) folds score
    negatively, graded by how many residues overlap, so the GA still has a
    gradient to climb out of collisions rather than a flat penalty plateau.
    """
    coords = genome_to_coords(genome)
    return fitness_from_coords(coords, sequence, collision_penalty)


def is_valid_fold(genome: Sequence[int]) -> bool:
    coords = genome_to_coords(genome)
    return len(set(coords)) == len(coords)
