#!/usr/bin/env python
"""Write single chains of PDB files as <stem>_<chain>.pdb (gemmi), protein residues only."""
import sys
import gemmi
for spec in sys.argv[2:]:
    path, chains = spec.split(':')
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_ligands_and_waters()
    for ch in chains.split(','):
        s2 = st.clone()
        for m in s2:
            for c in list(m):
                if c.name != ch:
                    m.remove_chain(c.name)
        s2.write_pdb(f"{sys.argv[1]}/{path.split('/')[-1][:-4]}_{ch}.pdb")
