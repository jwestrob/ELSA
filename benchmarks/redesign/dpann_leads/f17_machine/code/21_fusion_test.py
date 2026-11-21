"""Split-VirB4 test: does fusing the 2-TM partner to the ATPase reproduce full-length VirB4-family architecture?
USalign of single-chain fusion models (TM277a+ATPase504, TM278b+ATPase504, control ctl279+ATPase504) and of the
ATPase alone against full-length CedB (model) and TrwK (7O43 chain A); TM-score normalized by the reference."""
import subprocess, os
W = '/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine'
F, O = W + '/folds/fsq', W + '/usalign'
US = '/shared/software/bin/USalign'
src = open(W + '/20_usalign.py').read()
ns = {}
exec(src[src.index('def cif_chain_to_pdb'):src.index('# references')], ns)
ns['cif_chain_to_pdb'](W + '/refs/pdb/7O43.cif', 'A', O + '/TrwK_full.pdb')
refs = {'CedB_full': F + '/ref_CedB.pdb', 'TedB_full': F + '/ref_TedB.pdb', 'TrwK_full': O + '/TrwK_full.pdb'}
qs = ['fus_TM277a_ATP_A', 'fus_TM277a_ATP_C', 'fus_TM278b_ATP_A', 'fus_ctl279_ATP_A', 'ATPase504_A', 'ATPase504_C']
print('query'.ljust(18), '  '.join(r.ljust(12) for r in refs))
with open(O + '/fusion_test.tsv', 'w') as fh:
    fh.write('query\tref\tTM_by_ref\tTM_by_query\tRMSD\taligned\n')
    for q in qs:
        out = []
        for r, rp in refs.items():
            o = subprocess.run([US, F + f'/{q}.pdb', rp, '-outfmt', '2'], capture_output=True, text=True).stdout.splitlines()[1].split('\t')
            t1, t2, rmsd, L = float(o[2]), float(o[3]), float(o[4]), o[10] if len(o) > 10 else ''
            fh.write(f'{q}\t{r}\t{t2}\t{t1}\t{rmsd}\t{L}\n'); out.append(f'{t2:.3f}')
        print(q.ljust(18), '  '.join(x.ljust(12) for x in out))
