#!/usr/bin/env python3
"""
Homology processor for functional categorization of MMseqs2 results.
Processes alignment data into functional relationship categories for GPT-5 analysis.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .mmseqs2_runner import MMseqs2Alignment
from .elsa_similarity_analyzer import ELSASimilarityAnalyzer

logger = logging.getLogger(__name__)


class HomologyRelationship(Enum):
    """Types of homology relationships."""
    STRONG_ORTHOLOG = "strong_ortholog"      # >70% identity, >80% coverage
    WEAK_ORTHOLOG = "weak_ortholog"          # >50% identity, >60% coverage  
    PARALOG = "paralog"                      # 30-70% identity, variable coverage
    DISTANT_HOMOLOG = "distant_homolog"      # 20-50% identity, low coverage
    NO_HOMOLOGY = "no_homology"              # <20% identity or no significant hit


@dataclass
class HomologyPair:
    """Container for processed homology relationship."""
    query_id: str
    target_id: str
    relationship: HomologyRelationship
    identity: float
    coverage_query: float
    coverage_target: float
    evalue: float
    bitscore: float
    functional_conservation: str
    pathway_role: Optional[str] = None
    pfam_domains_query: Optional[str] = None
    pfam_domains_target: Optional[str] = None


@dataclass
class FunctionalGroup:
    """Container for functional protein group."""
    group_id: str
    function_description: str
    pathway: Optional[str]
    query_proteins: List[str]
    target_proteins: List[str]
    homology_pairs: List[HomologyPair]
    conservation_level: str


@dataclass
class HomologyAnalysis:
    """Complete homology analysis results."""
    block_id: str
    total_query_proteins: int
    total_target_proteins: int
    total_alignments: int
    ortholog_pairs: List[HomologyPair]
    functional_groups: List[FunctionalGroup]
    pathway_conservation: Dict[str, Any]
    conservation_summary: Dict[str, Any]
    analysis_status: str = "success"  # success, null_result, pipeline_failure
    failure_reason: Optional[str] = None
    elsa_evidence: Optional[Dict[str, Any]] = None


class HomologyProcessor:
    """Process MMseqs2 results into functional relationship categories."""
    
    def __init__(self):
        """Initialize homology processor."""
        self.pathway_keywords = self._load_pathway_keywords()
        self.elsa_analyzer = ELSASimilarityAnalyzer()
        logger.info("HomologyProcessor initialized with ELSA similarity support")
    
    def _load_pathway_keywords(self) -> Dict[str, List[str]]:
        """Load pathway keywords for functional categorization."""
        return {
            "dTDP_rhamnose_synthesis": [
                "dTDP", "rhamnose", "rml", "glucose-1-phosphate", "thymidylyltransferase",
                "dTDP-glucose", "4,6-dehydratase", "3,5-epimerase", "4-reductase"
            ],
            "peptidoglycan_synthesis": [
                "peptidoglycan", "murein", "mur", "UDP", "N-acetylglucosamine",
                "N-acetylmuramic", "alanine", "glutamate", "lysine", "diaminopimelate"
            ],
            "lipopolysaccharide_biosynthesis": [
                "LPS", "lipopolysaccharide", "waa", "rfb", "O-antigen", "core oligosaccharide",
                "lipid A", "KDO", "heptose"
            ],
            "DNA_replication": [
                "DNA", "replication", "polymerase", "helicase", "primase", "ligase",
                "topoisomerase", "single-strand", "origin", "replication fork"
            ],
            "transcription": [
                "transcription", "RNA polymerase", "sigma", "promoter", "termination",
                "elongation", "transcript"
            ],
            "translation": [
                "translation", "ribosome", "ribosomal", "tRNA", "aminoacyl", "peptidyl",
                "elongation factor", "initiation factor", "release factor"
            ],
            "energy_metabolism": [
                "ATP", "ADP", "NADH", "NADPH", "cytochrome", "electron transport",
                "oxidative phosphorylation", "glycolysis", "TCA cycle", "citrate"
            ],
            "transport": [
                "transport", "transporter", "permease", "ABC", "antiporter", "symporter",
                "channel", "porin", "efflux"
            ]
        }
    
    def process_homology_data(self, alignments: List[MMseqs2Alignment],
                            query_proteins: Dict[str, str],
                            target_proteins: Dict[str, str],
                            block_id: str,
                            query_locus: Optional[str] = None,
                            target_locus: Optional[str] = None) -> HomologyAnalysis:
        """Process MMseqs2 alignments into functional homology analysis.
        
        Args:
            alignments: MMseqs2 alignment results
            query_proteins: Query protein ID to function mapping
            target_proteins: Target protein ID to function mapping
            block_id: Syntenic block identifier
            
        Returns:
            Complete homology analysis with functional categorization
        """
        try:
            logger.info(f"Processing homology data for block {block_id}: {len(alignments)} alignments")
            
            # Determine analysis status
            if len(alignments) == 0:
                analysis_status = "null_result"
                failure_reason = "No significant protein alignments detected (legitimate biological result)"
            else:
                analysis_status = "success"
                failure_reason = None
            
            # Categorize individual relationships
            homology_pairs = []
            for alignment in alignments:
                pair = self._categorize_relationship(alignment, query_proteins, target_proteins)
                homology_pairs.append(pair)
            
            # Filter for ortholog pairs
            ortholog_pairs = [p for p in homology_pairs 
                            if p.relationship in [HomologyRelationship.STRONG_ORTHOLOG, 
                                                HomologyRelationship.WEAK_ORTHOLOG]]
            
            # Group proteins by function
            functional_groups = self._group_by_function(ortholog_pairs, query_proteins, target_proteins)
            
            # Analyze pathway conservation
            pathway_conservation = self._analyze_pathway_conservation(functional_groups)
            
            # Generate conservation summary
            conservation_summary = self._generate_conservation_summary(
                len(query_proteins), len(target_proteins), homology_pairs, ortholog_pairs
            )
            
            # Run ELSA similarity analysis if loci are provided
            elsa_evidence = None
            if query_locus and target_locus:
                logger.info(f"Running ELSA similarity analysis for {query_locus} vs {target_locus}")
                elsa_result = self.elsa_analyzer.analyze_block_similarity(block_id, query_locus, target_locus)
                if elsa_result:
                    elsa_evidence = {
                        'conservation_level': elsa_result.conservation_level,
                        'average_conservation': elsa_result.average_conservation,
                        'conserved_window_count': elsa_result.conserved_window_count,
                        'total_window_pairs': elsa_result.total_window_pairs,
                        'elsa_summary': elsa_result.elsa_summary,
                        'window_similarities': [
                            {
                                'window_1': sim.window_id_1,
                                'window_2': sim.window_id_2,
                                'conservation_score': sim.conservation_score,
                                'cosine_similarity': sim.cosine_similarity
                            }
                            for sim in elsa_result.window_similarities[:5]  # Top 5
                        ]
                    }
                    logger.info(f"ELSA evidence: {elsa_result.conservation_level} conservation, "
                               f"{elsa_result.total_window_pairs} window pairs")
            
            analysis = HomologyAnalysis(
                block_id=block_id,
                total_query_proteins=len(query_proteins),
                total_target_proteins=len(target_proteins),
                total_alignments=len(alignments),
                ortholog_pairs=ortholog_pairs,
                functional_groups=functional_groups,
                pathway_conservation=pathway_conservation,
                conservation_summary=conservation_summary,
                analysis_status=analysis_status,
                failure_reason=failure_reason,
                elsa_evidence=elsa_evidence
            )
            
            logger.info(f"Homology analysis completed: {len(ortholog_pairs)} orthologs, {len(functional_groups)} functional groups")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to process homology data: {e}")
            # Return analysis with pipeline failure status
            return HomologyAnalysis(
                block_id=block_id,
                total_query_proteins=len(query_proteins),
                total_target_proteins=len(target_proteins),
                total_alignments=0,
                ortholog_pairs=[],
                functional_groups=[],
                pathway_conservation={},
                conservation_summary={'conservation_level': 'unknown', 'summary': 'Analysis failed'},
                analysis_status="pipeline_failure",
                failure_reason=f"Processing pipeline failed: {str(e)}",
                elsa_evidence=None
            )
    
    def _categorize_relationship(self, alignment: MMseqs2Alignment,
                               query_proteins: Dict[str, str],
                               target_proteins: Dict[str, str]) -> HomologyPair:
        """Categorize a single alignment into homology relationship type."""
        
        # Determine relationship type based on identity and coverage
        max_coverage = max(alignment.coverage_query, alignment.coverage_target)
        
        if alignment.identity >= 0.7 and max_coverage >= 0.8:
            relationship = HomologyRelationship.STRONG_ORTHOLOG
        elif alignment.identity >= 0.5 and max_coverage >= 0.6:
            relationship = HomologyRelationship.WEAK_ORTHOLOG
        elif alignment.identity >= 0.3 and alignment.identity < 0.7:
            relationship = HomologyRelationship.PARALOG
        elif alignment.identity >= 0.2:
            relationship = HomologyRelationship.DISTANT_HOMOLOG
        else:
            relationship = HomologyRelationship.NO_HOMOLOGY
        
        # Assess functional conservation
        query_function = query_proteins.get(alignment.query_id, "hypothetical protein")
        target_function = target_proteins.get(alignment.target_id, "hypothetical protein")
        
        functional_conservation = self._assess_functional_conservation(
            query_function, target_function, alignment.identity
        )
        
        # Identify pathway role
        pathway_role = self._identify_pathway_role(query_function, target_function)
        
        return HomologyPair(
            query_id=alignment.query_id,
            target_id=alignment.target_id,
            relationship=relationship,
            identity=alignment.identity,
            coverage_query=alignment.coverage_query,
            coverage_target=alignment.coverage_target,
            evalue=alignment.evalue,
            bitscore=alignment.bitscore,
            functional_conservation=functional_conservation,
            pathway_role=pathway_role,
            pfam_domains_query=query_function,
            pfam_domains_target=target_function
        )
    
    def _assess_functional_conservation(self, query_func: str, target_func: str, identity: float) -> str:
        """Assess level of functional conservation between proteins."""
        
        # If both are hypothetical, assess by identity only
        if "hypothetical" in query_func.lower() and "hypothetical" in target_func.lower():
            if identity >= 0.9:
                return "likely_conserved_function"
            elif identity >= 0.7:
                return "possibly_conserved_function"
            else:
                return "unknown_function"
        
        # Compare function descriptions
        query_terms = set(query_func.lower().split(';'))
        target_terms = set(target_func.lower().split(';'))
        
        # Calculate functional overlap
        if query_terms & target_terms:  # Has overlapping domains
            if identity >= 0.8:
                return "catalytic_core_identical"
            elif identity >= 0.6:
                return "functional_domains_conserved"
            else:
                return "partial_domain_conservation"
        else:
            if identity >= 0.9:
                return "sequence_conserved_function_unknown"
            elif identity >= 0.7:
                return "moderate_conservation_different_annotation"
            else:
                return "divergent_function"
    
    def _identify_pathway_role(self, query_func: str, target_func: str) -> Optional[str]:
        """Identify pathway role based on protein functions."""
        
        combined_function = f"{query_func} {target_func}".lower()
        
        for pathway, keywords in self.pathway_keywords.items():
            if any(keyword.lower() in combined_function for keyword in keywords):
                return pathway
        
        return None
    
    def _group_by_function(self, ortholog_pairs: List[HomologyPair],
                          query_proteins: Dict[str, str],
                          target_proteins: Dict[str, str]) -> List[FunctionalGroup]:
        """Group proteins by functional similarity."""
        
        # Group by pathway role first
        pathway_groups = defaultdict(list)
        unassigned_pairs = []
        
        for pair in ortholog_pairs:
            if pair.pathway_role:
                pathway_groups[pair.pathway_role].append(pair)
            else:
                unassigned_pairs.append(pair)
        
        functional_groups = []
        
        # Create pathway-based groups
        for pathway, pairs in pathway_groups.items():
            query_proteins_in_group = list(set(p.query_id for p in pairs))
            target_proteins_in_group = list(set(p.target_id for p in pairs))
            
            # Determine conservation level
            avg_identity = sum(p.identity for p in pairs) / len(pairs)
            if avg_identity >= 0.9:
                conservation_level = "very_high"
            elif avg_identity >= 0.7:
                conservation_level = "high"
            elif avg_identity >= 0.5:
                conservation_level = "moderate"
            else:
                conservation_level = "low"
            
            group = FunctionalGroup(
                group_id=f"{pathway}_group",
                function_description=f"Proteins involved in {pathway.replace('_', ' ')}",
                pathway=pathway,
                query_proteins=query_proteins_in_group,
                target_proteins=target_proteins_in_group,
                homology_pairs=pairs,
                conservation_level=conservation_level
            )
            functional_groups.append(group)
        
        # Group remaining pairs by functional conservation
        if unassigned_pairs:
            conservation_groups = defaultdict(list)
            for pair in unassigned_pairs:
                conservation_groups[pair.functional_conservation].append(pair)
            
            for conservation_type, pairs in conservation_groups.items():
                if len(pairs) >= 2:  # Only create groups with multiple members
                    query_proteins_in_group = list(set(p.query_id for p in pairs))
                    target_proteins_in_group = list(set(p.target_id for p in pairs))
                    
                    group = FunctionalGroup(
                        group_id=f"{conservation_type}_group",
                        function_description=f"Proteins with {conservation_type.replace('_', ' ')}",
                        pathway=None,
                        query_proteins=query_proteins_in_group,
                        target_proteins=target_proteins_in_group,
                        homology_pairs=pairs,
                        conservation_level="moderate"
                    )
                    functional_groups.append(group)
        
        return functional_groups
    
    def _analyze_pathway_conservation(self, functional_groups: List[FunctionalGroup]) -> Dict[str, Any]:
        """Analyze conservation at the pathway level."""
        
        pathway_analysis = {}
        
        for group in functional_groups:
            if group.pathway:
                # Calculate pathway completeness
                total_proteins = len(group.query_proteins) + len(group.target_proteins)
                ortholog_count = len(group.homology_pairs)
                
                # Assess conservation completeness
                if ortholog_count >= 3 and group.conservation_level in ["very_high", "high"]:
                    completeness = "complete"
                elif ortholog_count >= 2:
                    completeness = "partial"
                else:
                    completeness = "minimal"
                
                pathway_analysis[group.pathway] = {
                    "conservation_level": group.conservation_level,
                    "completeness": completeness,
                    "ortholog_count": ortholog_count,
                    "total_proteins": total_proteins,
                    "key_functions": [p.functional_conservation for p in group.homology_pairs[:3]]
                }
        
        # Overall pathway conservation summary
        if pathway_analysis:
            complete_pathways = sum(1 for p in pathway_analysis.values() if p["completeness"] == "complete")
            total_pathways = len(pathway_analysis)
            
            pathway_analysis["summary"] = {
                "total_pathways_detected": total_pathways,
                "complete_pathways": complete_pathways,
                "pathway_conservation_score": complete_pathways / total_pathways if total_pathways > 0 else 0
            }
        
        return pathway_analysis
    
    def _generate_conservation_summary(self, query_count: int, target_count: int,
                                     all_pairs: List[HomologyPair],
                                     ortholog_pairs: List[HomologyPair]) -> Dict[str, Any]:
        """Generate overall conservation summary statistics."""
        
        if not all_pairs:
            return {"conservation_level": "none", "summary": "No homologous proteins detected"}
        
        # Calculate conservation metrics
        ortholog_fraction = len(ortholog_pairs) / min(query_count, target_count)
        avg_identity = sum(p.identity for p in ortholog_pairs) / len(ortholog_pairs) if ortholog_pairs else 0
        
        # Categorize relationship types
        relationship_counts = defaultdict(int)
        for pair in all_pairs:
            relationship_counts[pair.relationship.value] += 1
        
        # Determine overall conservation level
        if ortholog_fraction >= 0.5 and avg_identity >= 0.8:
            overall_level = "very_high"
        elif ortholog_fraction >= 0.3 and avg_identity >= 0.6:
            overall_level = "high"
        elif ortholog_fraction >= 0.1 and avg_identity >= 0.4:
            overall_level = "moderate"
        else:
            overall_level = "low"
        
        return {
            "conservation_level": overall_level,
            "ortholog_fraction": ortholog_fraction,
            "average_identity": avg_identity,
            "total_relationships": len(all_pairs),
            "relationship_breakdown": dict(relationship_counts),
            "summary": f"{overall_level.replace('_', ' ').title()} conservation with {len(ortholog_pairs)} orthologous pairs"
        }


def test_homology_processor():
    """Test the homology processor with MMseqs2 results."""
    print("=== Testing HomologyProcessor ===")
    
    try:
        # Check if MMseqs2 results exist
        alignment_file = Path("test_homology_output/mmseqs2_alignments.tsv")
        
        if not alignment_file.exists():
            print("❌ MMseqs2 alignment file not found. Run MMseqs2 test first.")
            return False
        
        # Load alignments (simplified - in real use would import from mmseqs2_runner)
        import pandas as pd
        df = pd.read_csv(alignment_file, sep='\t', header=None, names=[
            'query_id', 'target_id', 'identity', 'alnlen', 'mismatch', 'gapopen',
            'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bitscore', 'qlen', 'tlen'
        ])
        
        # Convert to MMseqs2Alignment objects
        from .mmseqs2_runner import MMseqs2Alignment
        alignments = []
        for _, row in df.iterrows():
            alignment = MMseqs2Alignment(
                query_id=row['query_id'],
                target_id=row['target_id'],
                identity=row['identity'] / 100.0,
                alignment_length=row['alnlen'],
                mismatches=row['mismatch'],
                gap_opens=row['gapopen'],
                query_start=row['qstart'],
                query_end=row['qend'],
                target_start=row['tstart'],
                target_end=row['tend'],
                evalue=row['evalue'],
                bitscore=row['bitscore'],
                query_length=row['qlen'],
                target_length=row['tlen'],
                coverage_query=(row['qend'] - row['qstart'] + 1) / row['qlen'],
                coverage_target=(row['tend'] - row['tstart'] + 1) / row['tlen']
            )
            alignments.append(alignment)
        
        print(f"✓ Loaded {len(alignments)} alignments from MMseqs2 results")
        
        # Create mock protein function mappings
        query_proteins = {align.query_id: "hypothetical protein" for align in alignments}
        target_proteins = {align.target_id: "hypothetical protein" for align in alignments}
        
        # Add some realistic functional annotations
        query_proteins[alignments[0].query_id] = "dTDP-glucose 4,6-dehydratase;PF01370"
        target_proteins[alignments[0].target_id] = "dTDP-glucose 4,6-dehydratase;PF01370"
        
        if len(alignments) > 1:
            query_proteins[alignments[1].query_id] = "DNA polymerase III;PF00712"
            target_proteins[alignments[1].target_id] = "DNA polymerase III;PF00712"
        
        print(f"✓ Created function mappings: {len(query_proteins)} query, {len(target_proteins)} target")
        
        # Test homology processor
        processor = HomologyProcessor()
        analysis = processor.process_homology_data(
            alignments, query_proteins, target_proteins, "test_block_0"
        )
        
        print(f"✓ Homology analysis completed:")
        print(f"  - Total alignments: {analysis.total_alignments}")
        print(f"  - Ortholog pairs: {len(analysis.ortholog_pairs)}")
        print(f"  - Functional groups: {len(analysis.functional_groups)}")
        print(f"  - Conservation level: {analysis.conservation_summary['conservation_level']}")
        print(f"  - Average identity: {analysis.conservation_summary['average_identity']:.3f}")
        
        # Test relationship categorization
        strong_orthologs = [p for p in analysis.ortholog_pairs 
                          if p.relationship == HomologyRelationship.STRONG_ORTHOLOG]
        print(f"✓ Strong orthologs detected: {len(strong_orthologs)}")
        
        # Test functional grouping
        if analysis.functional_groups:
            sample_group = analysis.functional_groups[0]
            print(f"✓ Sample functional group: {sample_group.function_description}")
            print(f"  - Conservation level: {sample_group.conservation_level}")
            print(f"  - Proteins: {len(sample_group.query_proteins)} query, {len(sample_group.target_proteins)} target")
        
        # Test pathway conservation
        if analysis.pathway_conservation:
            print(f"✓ Pathway conservation analysis completed")
            if "summary" in analysis.pathway_conservation:
                summary = analysis.pathway_conservation["summary"]
                print(f"  - Pathways detected: {summary['total_pathways_detected']}")
                print(f"  - Complete pathways: {summary['complete_pathways']}")
        
        print("✅ All homology processor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_homology_processor()