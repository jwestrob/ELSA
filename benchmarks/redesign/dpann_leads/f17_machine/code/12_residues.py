"""Hallmark residues at reference positions: align role members (or a sample) with a curated reference by MAFFT
(--add --keeplength on the reference frame) and tabulate residues at the reference's annotated positions."""
import subprocess, sys, collections, os, tempfile
def readfa(p):
    d, k = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith('>'): k = l[1:].split()[0]; d[k] = ''
        elif k: d[k] += l
    return d
REF = readfa('refs/refs.faa')
REF = {k.split('|')[1]: v for k, v in REF.items()}
def table(role_faa, acc, positions, nmax=400, label=''):
    mem = readfa(role_faa)
    keys = list(mem)[:nmax]
    with tempfile.TemporaryDirectory() as t:
        open(f'{t}/r.faa', 'w').write(f'>REF\n{REF[acc]}\n')
        open(f'{t}/m.faa', 'w').write(''.join(f'>{k}\n{mem[k]}\n' for k in keys))
        out = subprocess.run(['mafft', '--quiet', '--thread', '4', '--addfragments' if False else '--add', f'{t}/m.faa', '--keeplength', f'{t}/r.faa'],
                             capture_output=True, text=True).stdout
    open(f'{t if False else "data"}/aln_{label}.faa', 'w').write(out)
    al = {}
    k = None
    for l in out.splitlines():
        if l.startswith('>'): k = l[1:].split()[0]; al[k] = ''
        else: al[k] += l.strip()
    ref = al.pop('REF')
    # reference positions -> alignment columns
    col, n = {}, 0
    for i, ch in enumerate(ref):
        if ch != '-': n += 1; col[n] = i
    print(f'== {label}: {len(al)} members vs {acc}')
    for p, name in positions:
        c = collections.Counter(s[col[p]].upper() for s in al.values())
        tot = sum(c.values())
        top = ', '.join(f'{a}:{v/tot:.2f}' for a, v in c.most_common(3))
        print(f'  {name:28s} ref {REF[acc][p-1]}{p}: {top}')
    return al, ref
if __name__ == '__main__':
    R = 'data/roles/'
    table(R + 'Top6B.all.faa', 'O05207', [(42, 'ATP bind N42 (GHKL motif I)'), (76, 'ATP D76 (motif II)'), (96, 'ATP 96'), (97, 'ATP 97'), (98, 'ATP 98'),
          (107, 'G-loop 107'), (109, 'G-loop 109'), (111, 'G-loop 111'), (113, 'G-loop 113'), (427, 'transducer K427'), (45, 'E45? catalytic')], label='Top6B')
    table(R + 'Top6A_f17genomes.all.faa', 'O05208', [(106, 'catalytic Tyr Y106'), (105, 'R105?'), (209, 'Mg E209 (Toprim)'), (261, 'Mg D261'), (263, 'D263 (DxD)')], label='Top6A')
    # ATPase504 against CedB (Q4JAQ5; Walker A 286-293, K292 mutation abolishes DNA transfer) and VirB4 (P17794; Walker A 433-440)
    table(R + 'ATPase504.all.faa', 'Q4JAQ5', [(286, 'CedB Walker A G286'), (291, 'Walker A G291'), (292, 'Walker A K292 (essential)'), (293, 'Walker A T293')], label='ATPase_vs_CedB')
    table(R + 'ATPase504.all.faa', 'P17794', [(433, 'VirB4 Walker A G433'), (439, 'Walker A K439'), (440, 'Walker A T440')], label='ATPase_vs_VirB4')
