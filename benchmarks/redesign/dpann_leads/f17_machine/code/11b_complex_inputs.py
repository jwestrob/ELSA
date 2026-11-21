"""Fold inputs as submitted (written inline during the run; recorded here). Sequences from data/fold_candidates.tsv
(code/11_fold_inputs.py) plus locus-B proteins. Monomers via Sharur fold_biohub.py (esmfold2-fast), complexes via
code/cofold_pair.py (esmfold2-2026-05 all-atom), controls defined in the same batch."""
MONO = ['ATPase504_A', 'ATPase504_C', 'A2Ml794_A', 'A2Ml794_C', 'wHTH244_A', 'wHTH244_C', 'TM278b_A', 'TM277a_A', 'p49_A', 'Top6B_A', 'ctlTop6A_A']
MONO2 = {'fus_TM277a_ATP_A': 'TM277a_A+ATPase504_A', 'fus_TM277a_ATP_C': 'TM277a_C+ATPase504_C', 'ref_CedB': 'Q4JAQ5', 'ref_ApeCedB': 'Q9YFN0', 'ref_TedB': 'A3MU65'}
MONO3 = {'fus_ctl279_ATP_A': 'ctl279_A+ATPase504_A', 'fus_TM278b_ATP_A': 'TM278b_A+ATPase504_A'}
CX1 = [('hex_ATP_A', ['ATPase504_A*6']), ('hex_ATP_C', ['ATPase504_C*6']), ('hex_ctlThrRS', ['ctlThrRS_A*6']), ('dim_ATP_A', ['ATPase504_A*2']),
       ('ATP_TM277a_A', ['ATPase504_A', 'TM277a_A']), ('ATP_TM277a_C', ['ATPase504_C', 'TM277a_C']), ('ctl_ATP_ctl279', ['ATPase504_A', 'ctl279_A']),
       ('ctl_DUF87_TM277a', ['ctlDUF87_A', 'TM277a_A']), ('ATP_TM278b_A', ['ATPase504_A', 'TM278b_A']), ('ATP_TM132_A', ['ATPase504_A', 'TM132_A']),
       ('ctl_DUF87_TM132', ['ctlDUF87_A', 'TM132_A']), ('ATP_p102_A', ['ATPase504_A', 'p102_A']), ('ctl_DUF87_p102', ['ctlDUF87_A', 'p102_A']),
       ('A2Ml_wHTH_A', ['A2Ml794_A', 'wHTH244_A']), ('A2Ml_wHTH_C', ['A2Ml794_C', 'wHTH244_C']), ('ctl_A2Ml_ArsR', ['A2Ml794_A', 'ArsR178_A']),
       ('wHTH_p49_A', ['wHTH244_A', 'p49_A']), ('ctl_ArsR_p49', ['ArsR178_A', 'p49_A']), ('A2Ml_wHTH_p49_A', ['A2Ml794_A', 'wHTH244_A', 'p49_A']),
       ('TM277a_TM278b_A', ['TM277a_A', 'TM278b_A']), ('ctl_TM277a_ctl279', ['TM277a_A', 'ctl279_A']),
       ('motor4_A', ['ATPase504_A', 'TM277a_A', 'TM132_A', 'p102_A']), ('ctl_motor4_DUF87', ['ctlDUF87_A', 'TM277a_A', 'TM132_A', 'p102_A'])]
CX2 = [('ATP_p102_C', ['ATPase504_C', 'p102_C']), ('ctl_ATP_ctl90_C', ['ATPase504_C', 'ctl90_A']), ('ctl_ATP_ctl90_A', ['ATPase504_A', 'ctl90_A']),
       ('ATP_p102_B', ['ATPase504_B', 'p102_B']), ('TM277a_TM278b_B', ['TM277a_B', 'TM278b_B']), ('ctl_TM277aB_ctl279', ['TM277a_B', 'ctl279_A'])]
# locus A = 20814 (GCA_027333445.1, Micrarchaeaceae); B = 22577 (GCA_027363995.1, CABMGC01); C = 26611 (GCA_035290745.1, Anstonellales)
# controls: ctl279 (279-aa 2-TM protein, same contig as locus A), ctl90 (90-aa soluble, same contig), ctlDUF87 (364-aa DUF87+AAA_10
# protein elsewhere in genome A), ctlThrRS (556-aa threonyl-tRNA synthetase from the locus-A flank)
if __name__ == '__main__':
    print(len(MONO) + len(MONO2) + len(MONO3), 'monomers;', len(CX1) + len(CX2), 'complexes')
