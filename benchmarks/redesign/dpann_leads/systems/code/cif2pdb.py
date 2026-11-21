#!/usr/bin/env python
"""Convert a downloaded mmCIF reference structure to PDB format (gemmi), protein chains only."""
import sys
import gemmi
st = gemmi.read_structure(sys.argv[1])
st.setup_entities()
st.remove_ligands_and_waters()
st.write_pdb(sys.argv[2])
print([(ch.name, len(ch)) for ch in st[0]])
