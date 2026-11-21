# Download reference sequences and feature JSON from UniProt by accession
cd /Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/f17_machine/refs
ACC="Q4JAQ5 Q4JB69 Q4JB68 Q4JB70 Q9YFN0 A3MU75 A3MU74 A3MU65 Q97WG8 Q9YFZ4 P17794 O50330 P17793 P17796 P17792 Q5G3P5 Q2L5X8 Q6AG11 A0A8F5C2T3 A0A8F5C2Y3 P76578 P76577 P01023 P46889 Q9YFN1 Q9YFM9"
: > refs.faa
for a in $ACC; do curl -s "https://rest.uniprot.org/uniprotkb/$a.fasta" >> refs.faa; curl -s "https://rest.uniprot.org/uniprotkb/$a.json" -o $a.json; done
grep -c '>' refs.faa
