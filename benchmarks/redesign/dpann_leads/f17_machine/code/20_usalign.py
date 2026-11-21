"""Domain-level structure comparisons (USalign, biotite login node, single-threaded): F17 models vs the N-terminal and
C-terminal halves of VirB4-family proteins (CedB, TedB, A. pernix CedB models; TrwK 7O43 chain A) and HerA / VirB4-CTD
crystal structures, with unrelated F17 models as negative controls. TM-score normalized by the query (first) length."""
import subprocess, os, re
W = '/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine'
F, P, O = W + '/folds/fsq', W + '/refs/pdb', W + '/usalign'
os.makedirs(O, exist_ok=True)
US = '/shared/software/bin/USalign'
def cif_chain_to_pdb(cif, chain, out, lo=1, hi=10**6):
    lines, hdr = [], None
    rows = []
    for l in open(cif):
        if l.startswith('_atom_site.'): rows.append(l.strip().split('.', 1)[1]); continue
        if l.startswith('ATOM') and rows:
            f = l.split(); d = dict(zip(rows, f))
            if d.get('auth_asym_id') != chain or d.get('pdbx_PDB_model_num', '1') != '1': continue
            r = int(d['auth_seq_id'])
            if not (lo <= r <= hi): continue
            nm = d['auth_atom_id'].strip('"')
            lines.append('ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n' % (len(lines) + 1, nm if len(nm) == 4 else ' ' + nm, d['auth_comp_id'], 'A', r,
                         float(d['Cartn_x']), float(d['Cartn_y']), float(d['Cartn_z'])))
    open(out, 'w').write(''.join(lines) + 'END\n')
def cut_pdb(src, out, lo, hi):
    keep = [l for l in open(src) if l.startswith('ATOM') and lo <= int(l[22:26]) <= hi]
    open(out, 'w').write(''.join(keep) + 'END\n')
# references
cif_chain_to_pdb(P + '/7O43.cif', 'A', O + '/TrwK_NTD.pdb', 1, 430)
cif_chain_to_pdb(P + '/7O43.cif', 'A', O + '/TrwK_CTD.pdb', 431, 900)
cif_chain_to_pdb(P + '/4AG5.cif', 'A', O + '/VirB4CTD_4AG5.pdb')
cif_chain_to_pdb(P + '/4D2I.cif', 'A', O + '/HerA_4D2I.pdb')
cut_pdb(F + '/ref_CedB.pdb', O + '/CedB_NTD.pdb', 1, 265); cut_pdb(F + '/ref_CedB.pdb', O + '/CedB_CTD.pdb', 266, 597)
cut_pdb(F + '/ref_TedB.pdb', O + '/TedB_NTD.pdb', 1, 380); cut_pdb(F + '/ref_TedB.pdb', O + '/TedB_CTD.pdb', 381, 645)
cut_pdb(F + '/ref_ApeCedB.pdb', O + '/ApeCedB_NTD.pdb', 1, 215); cut_pdb(F + '/ref_ApeCedB.pdb', O + '/ApeCedB_CTD.pdb', 216, 461)
NT = ['CedB_NTD', 'TedB_NTD', 'ApeCedB_NTD', 'TrwK_NTD']
CT = ['CedB_CTD', 'TedB_CTD', 'ApeCedB_CTD', 'TrwK_CTD', 'VirB4CTD_4AG5', 'HerA_4D2I']
qn = {'TM277a_A': F + '/TM277a_A.pdb', 'TM277a_atlasD': F + '/atlas_f17_c2_DARC01000018.1_31.pdb', 'TM277a_atlasM': F + '/atlas_f17_c2_MBS3067675.1.pdb',
      'TM278b_A': F + '/TM278b_A.pdb', 'ctl_TM132': F + '/atlas_f17_c5_JAJYXL010000018.1_73.pdb', 'ctl_wHTH244_A': F + '/wHTH244_A.pdb',
      'ctl_ArsR178': F + '/atlas_f17_c10_HVC58599.1.pdb', 'ctl_Top6A_A': F + '/ctlTop6A_A.pdb'}
qc = {'ATPase504_A': F + '/ATPase504_A.pdb', 'ATPase504_C': F + '/ATPase504_C.pdb', 'ctl_Top6B_A': F + '/Top6B_A.pdb', 'ctl_Top6A_A': F + '/ctlTop6A_A.pdb'}
def tm(a, b):
    out = subprocess.run([US, a, b, '-outfmt', '2'], capture_output=True, text=True).stdout.splitlines()
    f = out[1].split('\t')
    return float(f[2]), float(f[3]), float(f[4])  # TM1 (by first), TM2 (by second), RMSD
rows = []
for qs, refs in [(qn, NT), (qc, CT)]:
    for q, path in qs.items():
        for r in refs:
            t1, t2, rmsd = tm(path, f'{O}/{r}.pdb')
            rows.append((q, r, t1, t2, rmsd))
with open(O + '/usalign.tsv', 'w') as fh:
    fh.write('query\tref\tTM_by_query\tTM_by_ref\tRMSD\n')
    for r in rows: fh.write('\t'.join(map(str, r)) + '\n')
import collections
for q in list(qn) + list(qc):
    print(q.ljust(16), '  '.join(f'{r}:{t1:.2f}/{t2:.2f}' for qq, r, t1, t2, _ in rows if qq == q))
