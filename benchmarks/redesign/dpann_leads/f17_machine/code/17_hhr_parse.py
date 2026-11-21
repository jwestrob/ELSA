"""Parse hhsearch .hhr summaries: top hits (prob >= threshold) per query file, with descriptions."""
import sys, re, glob, os
def parse(path, pmin=50, top=12):
    lines = open(path).read().splitlines()
    out, desc = [], {}
    # descriptions from the alignment blocks ('>' lines)
    for l in lines:
        if l.startswith('>'):
            k = l[1:].split()[0]; desc[k] = l[1:][len(k):].strip()[:90]
    i = next(i for i, l in enumerate(lines) if l.startswith(' No Hit'))
    for l in lines[i + 1:]:
        if not l.strip(): break
        m = re.match(r'\s*(\d+)\s+(\S+)\s.*?\s+(\d+\.\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+-\d+)\s+(\d+-\d+)\s*\((\d+)\)', l)
        if not m: continue
        _, hit, prob, ev, pv, sc, ss, cols, qr, tr, tl = m.groups()
        if float(prob) < pmin: continue
        out.append((hit, float(prob), ev, int(cols), qr, tr, int(tl), desc.get(hit, '')))
    q = next((l.split()[1] for l in lines if l.startswith('Match_columns')), '?')
    nseq = next((l.split(None, 1)[1] for l in lines if l.startswith('No_of_seqs')), '?')
    return q, nseq, out[:top]
if __name__ == '__main__':
    pmin = float(sys.argv[2]) if len(sys.argv) > 2 else 50
    for p in sorted(glob.glob(sys.argv[1])):
        q, nseq, out = parse(p, pmin)
        print(f'## {os.path.basename(p)}  match_cols {q}  seqs {nseq}')
        seen = set()
        for hit, prob, ev, cols, qr, tr, tl, d in out:
            if d in seen: continue
            seen.add(d)
            print(f'   {prob:5.1f} {ev:>8s} cols {cols:4d} q {qr:>9s} t {tr:>9s} ({tl}) {hit:14s} {d}')
