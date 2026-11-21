"""Align every work/s2/msa/fa/*.faa with >= 2 sequences (mafft --auto, one thread; input order kept) into
work/s2/msa/aln/<name>.aln.faa; existing alignments are kept (resumable)."""
import subprocess
from pathlib import Path

M = Path(__file__).resolve().parent / "work" / "s2" / "msa"
(M / "aln").mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted((M / "fa").glob("*.faa")):
    out = M / "aln" / (p.stem + ".aln.faa")
    if out.exists() or sum(1 for l in open(p) if l.startswith(">")) < 2:
        continue
    r = subprocess.run(["mafft", "--auto", "--thread", "1", "--quiet", str(p)], capture_output=True, text=True, check=True)
    out.write_text(r.stdout)
    n += 1
print(n, "alignments written")
