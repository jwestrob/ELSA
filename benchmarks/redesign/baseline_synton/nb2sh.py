#!/usr/bin/env python
"""Extract the code cells of compute_syntons.ipynb verbatim into cells/cell_<i>.sh (run in order by run_cells.sh).

'%%bash' cells: the magic line is dropped, the body is written unchanged.
'!cmd' cells: the leading '!' is dropped.
The cell text is otherwise byte-identical to the notebook (sha256 of each body recorded in cells/manifest.json).
"""
import hashlib
import json
import sys
from pathlib import Path

nb_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
nb = json.load(open(nb_path))
out_dir.mkdir(parents=True, exist_ok=True)
manifest = dict(notebook=str(nb_path), notebook_sha256=hashlib.sha256(nb_path.read_bytes()).hexdigest(), cells=[])
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.startswith("%%bash"):
        body = src.split("\n", 1)[1] if "\n" in src else ""
    elif src.startswith("!"):
        body = src[1:]
    else:
        raise SystemExit(f"cell {i}: unexpected cell type: {src[:40]!r}")
    p = out_dir / f"cell_{i:02d}.sh"
    p.write_text(body + "\n")
    manifest["cells"].append(dict(cell=i, file=p.name, sha256=hashlib.sha256(body.encode()).hexdigest(),
                                  first_line=body.strip().splitlines()[0] if body.strip() else ""))
json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
