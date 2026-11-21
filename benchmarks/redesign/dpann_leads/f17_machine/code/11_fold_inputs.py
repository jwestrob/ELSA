"""Fold inputs: monomers missing from the atlas set, plus complexes with controls; hydrophobic screen first."""
import duckdb, pandas as pd
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
ids = {'ATPase504_A': 'DAHWZL010000033.1_22', 'ATPase504_C': 'HLD60057.1', 'A2Ml794_A': 'DAHWZL010000033.1_27', 'A2Ml794_C': 'HLD60061.1',
       'wHTH244_A': 'DAHWZL010000033.1_25', 'wHTH244_C': 'HLD60060.1', 'TM277a_A': 'DAHWZL010000033.1_21', 'TM277a_C': 'HLD60056.1',
       'TM278b_A': 'DAHWZL010000033.1_20', 'TM132_A': 'DAHWZL010000033.1_24', 'TM132_C': 'HLD60059.1', 'p102_A': 'DAHWZL010000033.1_23',
       'p102_C': 'HLD60058.1', 'p49_A': 'DAHWZL010000033.1_26', 'pil129_A': 'DAHWZL010000033.1_12', 'MP4610_A': 'DAHWZL010000033.1_14',
       'p244b_A': 'DAHWZL010000033.1_13', 'Top6B_A': 'DAHWZL010000033.1_28', 'KH171_A': 'DAHWZL010000033.1_29', 'ArsR178_A': 'DAHWZL010000033.1_30',
       'ctlDUF87_A': 'DAHWZL010000041.1_15', 'ctlThrRS_A': 'DAHWZL010000033.1_10', 'ctlTop6A_A': 'DAHWZL010000015.1_19',
       'ctl279_A': 'DAHWZL010000033.1_6', 'ctlGT2_A': 'DAHWZL010000033.1_7', 'ctl90_A': 'DAHWZL010000033.1_32'}
q = pd.DataFrame({'name': list(ids), 'protein_id': list(ids.values())})
c.register('q', q)
d = c.sql("select q.name, q.protein_id, rtrim(p.sequence,'*') seq, p.sequence_length L from q join proteins p using(protein_id)").df()
KD = dict(zip('ARNDCQEGHILKMFPSTWYV', [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]))
def ntm(s, w=19, t=1.6):
    v = [sum(KD.get(x, 0) for x in s[i:i + w]) / w for i in range(max(1, len(s) - w + 1))]
    n, i = 0, 0
    while i < len(v):
        if v[i] >= t: n += 1; i += w
        else: i += 1
    return n
d['hyd'] = d.seq.map(lambda s: round(sum(x in 'ILVFMAWC' for x in s) / len(s), 3))
d['kd_tm'] = d.seq.map(ntm)
d.to_csv('data/fold_candidates.tsv', sep='\t', index=False)
print(d[['name', 'protein_id', 'L', 'hyd', 'kd_tm']].to_string())
