#!/usr/bin/env python3
"""
GPT-5 powered syntenic block analysis using DSPy for structured biological interpretation.
"""

import dspy
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SyntenicBlockData:
    """Container for syntenic block analysis data."""
    block_id: str
    genomes: List[str]
    genes: List[Dict[str, Any]]
    coordinates: Dict[str, Any]
    similarity_score: float
    length: int
    pfam_domains: List[Dict[str, Any]]
    conservation_stats: Dict[str, Any]
    homology_analysis: Optional[Dict[str, Any]] = None


class SyntenicBlockAnalysis(dspy.Signature):
    """Analyze a syntenic block for biological significance and evolutionary patterns.
    
    CRITICAL CONSTRAINTS:
    - Only use data explicitly provided in the input fields
    - Do NOT invent specific domain names, E-values, or protein functions not provided
    - Mark any functional predictions as speculative when annotation data is missing
    - Base analysis on sequence conservation and synteny patterns when domain data unavailable
    
    Provides structured biological interpretation of conserved genomic regions,
    focusing on functional significance and evolutionary insights.
    """
    
    block_info = dspy.InputField(
        desc="Syntenic block metadata including coordinates, genes, similarity scores, and length"
    )
    pfam_domains = dspy.InputField(
        desc="PFAM domain annotations for all genes in the block with e-values and positions"
    )
    genome_context = dspy.InputField(
        desc="Source genome species, strains, and any available ecological/clinical context"
    )
    conservation_pattern = dspy.InputField(
        desc="Gene conservation statistics and variation patterns across genomes"
    )
    homology_data = dspy.InputField(
        desc="Detailed protein homology analysis including ortholog pairs, functional groups, and pathway conservation from MMseqs2 alignments"
    )
    
    functional_summary = dspy.OutputField(
        desc="2-3 sentence summary of syntenic block significance. If PFAM domains unavailable, focus on conservation patterns and explicitly state functional limitations. Do NOT invent specific protein functions."
    )
    conservation_rationale = dspy.OutputField(
        desc="1-2 sentences explaining conservation based ONLY on provided data (sequence identity, synteny). Mark speculative functional reasons as such."
    )
    key_genes = dspy.OutputField(
        desc="List genes by ID with conservation levels from provided data. Do NOT invent protein names or specific functions not explicitly provided."
    )
    homology_insights = dspy.OutputField(
        desc="Insights from actual homology data provided. If no MMseqs2 results available, explicitly state this limitation."
    )
    biological_significance = dspy.OutputField(
        desc="Evolutionary significance based on conservation patterns. Mark any clinical/biotechnological predictions as speculative when functional data is limited."
    )


class GPT5SyntenicAnalyzer:
    """GPT-5 powered analyzer for syntenic blocks using structured DSPy signatures."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        """Initialize the analyzer with GPT-5 high reasoning configuration.
        
        Args:
            api_key: OpenAI API key (if None, expects OPENAI_API_KEY env var)
            model: Model to use (defaults to gpt-5 with high reasoning effort)
        """
        self.api_key = api_key
        self.model = model
        self._setup_dspy()
        
    def _setup_dspy(self):
        """Configure DSPy with GPT-5 high reasoning effort for complex biological analysis."""
        try:
            # Configure OpenAI model with high reasoning using current DSPy API
            import os
            if self.api_key:
                os.environ["OPENAI_API_KEY"] = self.api_key
            
            # Try different DSPy API configurations (API has evolved)
            lm = None
            
            # Try new API first - GPT-5 requires specific parameters
            try:
                if self.model == "gpt-5":
                    # Sanity check: verify LiteLLM supports reasoning_effort for GPT-5
                    try:
                        from litellm import get_supported_openai_params
                        supported_params = get_supported_openai_params(model="gpt-5", custom_llm_provider="openai")
                        logger.debug(f"GPT-5 supported params: {supported_params}")
                        if "reasoning_effort" not in str(supported_params):
                            logger.warning("LiteLLM may not support reasoning_effort for GPT-5 - consider upgrading")
                    except Exception as e:
                        logger.debug(f"Could not check LiteLLM param support: {e}")
                    
                    # GPT-5 reasoning model requirements with high reasoning effort
                    # Using standard gpt-5 model with reasoning_effort parameter
                    lm = dspy.LM(
                        "openai/gpt-5",  # standard GPT-5 model
                        temperature=1.0,
                        max_tokens=40000,  # LiteLLM maps to max_completion_tokens for Chat
                        reasoning_effort="high",  # GPT-5 high reasoning effort
                        allowed_openai_params=['reasoning_effort']  # Tell LiteLLM to allow this param
                    )
                else:
                    # Other models (like GPT-4o)
                    lm = dspy.LM(
                        model=f"openai/{self.model}",
                        max_tokens=2000,
                        temperature=0.1,
                    )
                dspy.configure(lm=lm)
            except (AttributeError, TypeError):
                # Try older API patterns
                try:
                    # Alternative: direct model specification
                    lm = dspy.OpenAI(
                        model=self.model,
                        api_key=self.api_key,
                        max_tokens=2000,
                        temperature=0.1,
                    )
                    dspy.settings.configure(lm=lm)
                except AttributeError:
                    # Fallback to even older API
                    dspy.settings.configure(
                        lm=dspy.LM(
                            model=self.model,
                            api_key=self.api_key,
                            max_tokens=2000,
                            temperature=0.1,
                        )
                    )
            
            # Initialize the prediction module
            self.analyzer = dspy.Predict(SyntenicBlockAnalysis)
            logger.info(f"GPT-5 syntenic analyzer initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy with GPT-5: {e}")
            logger.error(f"Available DSPy attributes: {[attr for attr in dir(dspy) if not attr.startswith('_')]}")
            raise
    
    def analyze_block(self, block_data: SyntenicBlockData) -> Dict[str, Any]:
        """Analyze a syntenic block and return structured biological insights.
        
        Args:
            block_data: Complete syntenic block information
            
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            logger.info(f"Starting GPT-5 analysis of syntenic block: {block_data.block_id}")
            
            # Prepare structured input for DSPy
            block_info_str = self._format_block_info(block_data)
            pfam_domains_str = self._format_pfam_domains(block_data.pfam_domains)
            genome_context_str = self._format_genome_context(block_data.genomes)
            conservation_str = self._format_conservation_pattern(block_data.conservation_stats)
            homology_str = self._format_homology_data(block_data)
            
            # Run DSPy prediction with structured inputs
            result = self.analyzer(
                block_info=block_info_str,
                pfam_domains=pfam_domains_str,
                genome_context=genome_context_str,
                conservation_pattern=conservation_str,
                homology_data=homology_str
            )
            
            # Structure the response
            analysis_result = {
                "block_id": block_data.block_id,
                "analysis": {
                    "functional_summary": result.functional_summary,
                    "conservation_rationale": result.conservation_rationale, 
                    "key_genes": result.key_genes,
                    "homology_insights": result.homology_insights,
                    "biological_significance": result.biological_significance
                },
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "block_length": block_data.length,
                    "similarity_score": block_data.similarity_score,
                    "num_genomes": len(block_data.genomes),
                    "num_genes": len(block_data.genes),
                    "num_pfam_domains": len(block_data.pfam_domains)
                }
            }
            
            logger.info(f"GPT-5 analysis completed for block: {block_data.block_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"GPT-5 analysis failed for block {block_data.block_id}: {e}")
            return {
                "block_id": block_data.block_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_block_info(self, block_data: SyntenicBlockData) -> str:
        """Format syntenic block metadata for GPT-5 input."""
        # Show all genes, not just a subset
        total_genes = len(block_data.genes)
        genes_list = []
        for g in block_data.genes:  # SHOW ALL GENES
            gene_id = g.get('gene_id', g.get('id', 'unknown'))
            function = g.get('function')
            if function and function != 'hypothetical protein':
                genes_list.append(f"{gene_id} ({function})")
            else:
                genes_list.append(f"{gene_id} (no functional annotation)")
        
        # If too many genes, show first 50 with indication of total
        if len(genes_list) > 50:
            displayed_genes = genes_list[:50]
            displayed_genes.append(f"... and {len(genes_list) - 50} more genes")
            genes_display = ', '.join(displayed_genes)
        else:
            genes_display = ', '.join(genes_list)
        
        return f"""Syntenic Block {block_data.block_id}:
- Core Aligned Genes: {total_genes} genes (FOCUSED ON SYNTENIC REGIONS ONLY)
- Aligned Windows: {block_data.length} ELSA windows
- Similarity Score: {block_data.similarity_score:.3f} 
- Coordinates: {block_data.coordinates}
- Genes: {genes_display}
- Total Genomes: {len(block_data.genomes)}
- ANALYSIS SCOPE: Core aligned regions only, not extended genomic context"""
    
    def _format_pfam_domains(self, pfam_domains: List[Dict[str, Any]]) -> str:
        """Format PFAM domain annotations for GPT-5 input."""
        if not pfam_domains:
            return "No PFAM domains annotated for this block."
        
        # Group domains by frequency and significance
        domain_counts = {}
        significant_domains = []
        
        for domain in pfam_domains:
            domain_name = domain.get('hmm_name', 'unknown')
            if domain_name not in domain_counts:
                domain_counts[domain_name] = 0
            domain_counts[domain_name] += 1
            
            # Collect domains with good e-values
            if domain.get('evalue', 1.0) <= 1e-5:
                evalue_str = f"{domain.get('evalue'):.1e}" if domain.get('evalue') else "1e-5"
                significant_domains.append(f"{domain_name} (E={evalue_str})")
        
        # Format top domains
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        domain_summary = [f"{name} (×{count})" for name, count in top_domains]
        
        return f"""PFAM Domains ({len(pfam_domains)} total):
- Most frequent: {', '.join(domain_summary)}
- Significant hits: {', '.join(significant_domains[:5])}"""
    
    def _format_genome_context(self, genomes: List[str]) -> str:
        """Format genome context information for GPT-5 input."""
        # Try to infer species/context from genome names
        genome_info = []
        for genome in genomes[:5]:  # Limit to first 5 genomes
            # Basic heuristics for common naming patterns
            if 'coli' in genome.lower():
                genome_info.append(f"{genome} (E. coli)")
            elif any(x in genome.lower() for x in ['staph', 'aureus']):
                genome_info.append(f"{genome} (S. aureus)")
            elif any(x in genome.lower() for x in ['pseudomonas', 'aeruginosa']):
                genome_info.append(f"{genome} (P. aeruginosa)")
            else:
                genome_info.append(f"{genome} (bacterial strain)")
        
        return f"""Genome Context:
- Genomes: {', '.join(genome_info)}
- Total: {len(genomes)} genomes
- Domain: Bacteria (inferred)"""
    
    def _format_conservation_pattern(self, conservation_stats: Dict[str, Any]) -> str:
        """Format conservation statistics for GPT-5 input."""
        if not conservation_stats:
            return "Conservation statistics not available."
        
        return f"""Conservation Pattern:
- Core genes: {conservation_stats.get('core_genes', 'unknown')}
- Accessory genes: {conservation_stats.get('accessory_genes', 'unknown')}
- Average identity: {conservation_stats.get('avg_identity', 'unknown')}
- Conservation level: {conservation_stats.get('conservation_level', 'moderate')}"""
    
    def _format_homology_data(self, block_data: SyntenicBlockData) -> str:
        """Format homology analysis data for GPT-5 input."""
        if not block_data.homology_analysis:
            return "Detailed homology analysis not available for this block."
        
        homology = block_data.homology_analysis
        
        # Check analysis status and handle appropriately
        analysis_status = homology.get('analysis_status', 'unknown')
        failure_reason = homology.get('failure_reason')
        
        if analysis_status == 'pipeline_failure':
            return f"Homology analysis pipeline failure: {failure_reason}. Analysis based on PFAM domains and synteny only."
        elif analysis_status == 'null_result':
            status_note = f"Note: {failure_reason}. This represents a legitimate biological finding, not a technical failure."
        else:
            status_note = "Homology analysis completed successfully."
        
        # Format ortholog pairs
        ortholog_pairs = homology.get('ortholog_pairs', [])
        strong_orthologs = [p for p in ortholog_pairs if p.get('relationship') == 'strong_ortholog']
        
        ortholog_summary = []
        for pair in strong_orthologs[:5]:  # Show top 5
            query_id = pair.get('query_id', 'unknown')
            target_id = pair.get('target_id', 'unknown')
            identity = pair.get('identity', 0)
            conservation = pair.get('functional_conservation', 'unknown')
            ortholog_summary.append(f"{query_id} ↔ {target_id} ({identity:.1%} identity, {conservation})")
        
        # Format functional groups
        functional_groups = homology.get('functional_groups', [])
        group_summary = []
        for group in functional_groups[:3]:  # Show top 3 groups
            description = group.get('function_description', 'unknown function')
            conservation_level = group.get('conservation_level', 'unknown')
            protein_count = len(group.get('query_proteins', [])) + len(group.get('target_proteins', []))
            group_summary.append(f"{description} ({conservation_level} conservation, {protein_count} proteins)")
        
        # Format pathway conservation
        pathway_conservation = homology.get('pathway_conservation', {})
        pathway_summary = []
        for pathway, data in pathway_conservation.items():
            if pathway != 'summary' and isinstance(data, dict):
                completeness = data.get('completeness', 'unknown')
                conservation_level = data.get('conservation_level', 'unknown')
                pathway_summary.append(f"{pathway.replace('_', ' ')} ({completeness}, {conservation_level})")
        
        # Overall conservation summary
        conservation_summary = homology.get('conservation_summary', {})
        overall_level = conservation_summary.get('conservation_level', 'unknown')
        ortholog_fraction = conservation_summary.get('ortholog_fraction', 0)
        avg_identity = conservation_summary.get('average_identity', 0)
        
        # Add ELSA evidence if available
        elsa_section = ""
        elsa_evidence = homology.get('elsa_evidence')
        if elsa_evidence:
            elsa_level = elsa_evidence.get('conservation_level', 'unknown')
            elsa_windows = elsa_evidence.get('total_window_pairs', 0)
            elsa_conserved = elsa_evidence.get('conserved_window_count', 0)
            elsa_summary = elsa_evidence.get('elsa_summary', '')
            
            elsa_section = f"""

ELSA Windowed Embedding Analysis:
• Conservation level: {elsa_level}
• Window pairs analyzed: {elsa_windows}
• Highly conserved windows: {elsa_conserved}
• Summary: {elsa_summary}"""
        
        return f"""Detailed Homology Analysis:

Status: {status_note}

Ortholog Relationships ({len(ortholog_pairs)} total, {len(strong_orthologs)} strong):
{chr(10).join(f"• {summary}" for summary in ortholog_summary) if ortholog_summary else "• No strong orthologs detected"}

Functional Groups ({len(functional_groups)} identified):
{chr(10).join(f"• {summary}" for summary in group_summary) if group_summary else "• No functional groups identified"}

Pathway Conservation ({len(pathway_summary)} pathways):
{chr(10).join(f"• {summary}" for summary in pathway_summary) if pathway_summary else "• No conserved pathways detected"}

Overall Conservation:
• Level: {overall_level}
• Ortholog fraction: {ortholog_fraction:.1%}
• Average identity: {avg_identity:.1%}
• Summary: {conservation_summary.get('summary', 'No summary available')}{elsa_section}"""


def test_analyzer():
    """Test function for the GPT-5 analyzer."""
    # Create mock data for testing
    mock_block = SyntenicBlockData(
        block_id="test_001",
        genomes=["E.coli_K12", "E.coli_O157"],
        genes=[
            {"id": "gene_001", "name": "dnaA", "function": "chromosomal replication initiation protein"},
            {"id": "gene_002", "name": "dnaN", "function": "DNA polymerase III subunit beta"},
            {"id": "gene_003", "name": "recF", "function": "DNA replication and repair protein RecF"}
        ],
        coordinates={"start": 1000, "end": 4000},
        similarity_score=0.95,
        length=3,
        pfam_domains=[
            {"hmm_name": "PF00308", "evalue": 1e-50, "description": "Bacterial dnaA protein"},
            {"hmm_name": "PF00712", "evalue": 1e-45, "description": "DNA polymerase III beta subunit"}
        ],
        conservation_stats={"core_genes": 3, "avg_identity": 0.95, "conservation_level": "high"}
    )
    
    # Test without actual API call
    print("Mock syntenic block data created successfully")
    print(f"Block ID: {mock_block.block_id}")
    print(f"Genes: {len(mock_block.genes)}")
    print(f"PFAM domains: {len(mock_block.pfam_domains)}")


if __name__ == "__main__":
    test_analyzer()