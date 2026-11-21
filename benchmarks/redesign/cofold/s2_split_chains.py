"""Split predicted complexes (PDB text written by biohub_cofold.py) into per-chain PDB files for Foldseek.
Usage: python s2_split_chains.py OUT_DIR PDB [PDB ...]; chain files are <fold_id>_<chain>.pdb, complexes
are copied as <fold_id>.pdb into OUT_DIR/complex/."""
import shutil
import sys
from pathlib import Path

out = Path(sys.argv[1])
(out / "chains").mkdir(parents=True, exist_ok=True)
(out / "complex").mkdir(parents=True, exist_ok=True)
n = 0
for p in map(Path, sys.argv[2:]):
    by = {}
    for line in open(p):
        if line.startswith("ATOM"):
            by.setdefault(line[21], []).append(line)
    for ch, lines in by.items():
        open(out / "chains" / f"{p.stem}_{ch}.pdb", "w").write("".join(lines) + "END\n")
        n += 1
    shutil.copy(p, out / "complex" / p.name)
print(n, "chains from", len(sys.argv) - 2, "complexes")
