#!/usr/bin/env python3
"""
Streamlit AI Analysis Panel for GPT-5 powered syntenic block interpretation.
Provides an interactive interface for biological analysis of conserved genomic regions.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AIAnalysisPanel:
    """Streamlit component for AI-powered syntenic block analysis."""
    
    def __init__(self):
        """Initialize the AI analysis panel."""
        self.session_key = "ai_analysis_results"
        
        # Initialize session state for caching analyses
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {}
    
    def render_analysis_trigger(self, block_id: str, block_data: Dict[str, Any]) -> bool:
        """Render the AI analysis trigger button for a syntenic block.
        
        Args:
            block_id: Identifier for the syntenic block
            block_data: Basic block information for display
            
        Returns:
            True if analysis was triggered, False otherwise
        """
        # Create a unique key for this block's button
        button_key = f"ai_analyze_{block_id}"
        
        # Show block info and analysis button inline
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**Block {block_id}** - "
                    f"Length: {block_data.get('length', 'N/A')} genes, "
                    f"Identity: {block_data.get('identity', 0):.1%}")
        
        with col2:
            # Check if we have cached analysis
            is_analyzed = block_id in st.session_state[self.session_key]
            if is_analyzed:
                if st.button("🔄 Re-analyze", key=f"reanalyze_{block_id}", 
                           help="Run AI analysis again"):
                    return True
            else:
                if st.button("🧠 AI Explain", key=button_key, 
                           help="Analyze this block with GPT-5"):
                    return True
        
        with col3:
            # Show analysis status
            if is_analyzed:
                st.success("✅ Analyzed")
            else:
                st.info("💭 Ready")
        
        return False
    
    def render_analysis_panel(self, block_id: Optional[str] = None) -> None:
        """Render the main AI analysis panel at the top of the page.
        
        Args:
            block_id: Currently selected block for analysis (if any)
        """
        st.markdown("---")
        st.subheader("🧠 AI Syntenic Analysis")
        
        # Show current analysis or selection prompt
        if block_id and block_id in st.session_state[self.session_key]:
            self._render_analysis_results(block_id)
        else:
            self._render_selection_prompt()
    
    def _render_analysis_results(self, block_id: str) -> None:
        """Render the AI analysis results for a block."""
        analysis = st.session_state[self.session_key][block_id]
        
        if "error" in analysis:
            self._render_error_state(block_id, analysis["error"])
            return
        
        # Analysis header
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### Analysis: Block {block_id}")
        with col2:
            if st.button("📋 Export", key=f"export_{block_id}"):
                self._export_analysis(block_id, analysis)
        
        # Main analysis content
        analysis_data = analysis.get("analysis", {})
        metadata = analysis.get("metadata", {})
        
        # Functional Summary (most prominent)
        if "functional_summary" in analysis_data:
            st.markdown("#### ✨ Summary")
            st.info(analysis_data["functional_summary"])
        
        # Key insights in expandable sections
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🔬 Key Genes", expanded=True):
                if "key_genes" in analysis_data:
                    st.write(analysis_data["key_genes"])
                else:
                    st.write("No key genes identified.")
            
            with st.expander("🧬 Conservation", expanded=False):
                if "conservation_rationale" in analysis_data:
                    st.write(analysis_data["conservation_rationale"])
                else:
                    st.write("Conservation analysis not available.")
        
        with col2:
            with st.expander("🎯 Biological Significance", expanded=True):
                if "biological_significance" in analysis_data:
                    st.write(analysis_data["biological_significance"])
                else:
                    st.write("Biological significance not determined.")
            
            with st.expander("📊 Analysis Metadata", expanded=False):
                if metadata:
                    st.json({
                        "Model": metadata.get("model", "unknown"),
                        "Timestamp": metadata.get("timestamp", "unknown"),
                        "Block Length": metadata.get("block_length", "unknown"),
                        "Similarity Score": metadata.get("similarity_score", "unknown"),
                        "Genomes": metadata.get("num_genomes", "unknown"),
                        "PFAM Domains": metadata.get("num_pfam_domains", "unknown")
                    })
                else:
                    st.write("Metadata not available.")
    
    def _render_error_state(self, block_id: str, error: str) -> None:
        """Render error state for failed analysis."""
        st.error(f"❌ Analysis failed for block {block_id}")
        st.write(f"**Error:** {error}")
        
        if st.button("🔄 Retry Analysis", key=f"retry_{block_id}"):
            # Clear the error from session state to allow retry
            if block_id in st.session_state[self.session_key]:
                del st.session_state[self.session_key][block_id]
            st.rerun()
    
    def _render_selection_prompt(self) -> None:
        """Render prompt to select a block for analysis."""
        st.info("👆 Click **🧠 AI Explain** on any syntenic block below to get biological insights powered by GPT-5.")
        
        # Show any cached analyses as quick access
        cached_analyses = st.session_state[self.session_key]
        if cached_analyses:
            st.markdown("**Recent Analyses:**")
            cols = st.columns(min(3, len(cached_analyses)))
            
            for i, (block_id, analysis) in enumerate(cached_analyses.items()):
                with cols[i % 3]:
                    if st.button(f"📖 Block {block_id}", key=f"view_{block_id}"):
                        st.session_state["selected_analysis"] = block_id
                        st.rerun()
    
    def store_analysis_result(self, block_id: str, analysis_result: Dict[str, Any]) -> None:
        """Store analysis result in session state for caching.
        
        Args:
            block_id: Block identifier
            analysis_result: Complete analysis result from GPT-5
        """
        st.session_state[self.session_key][block_id] = analysis_result
        logger.info(f"Stored analysis result for block: {block_id}")
    
    def get_cached_analysis(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result if available.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Cached analysis result or None
        """
        return st.session_state[self.session_key].get(block_id)
    
    def _export_analysis(self, block_id: str, analysis: Dict[str, Any]) -> None:
        """Export analysis results to downloadable format."""
        try:
            # Prepare export data
            export_data = {
                "block_id": block_id,
                "export_timestamp": datetime.now().isoformat(),
                "analysis": analysis
            }
            
            # Create formatted text version
            analysis_data = analysis.get("analysis", {})
            text_export = f"""ELSA Syntenic Block Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Block ID: {block_id}

FUNCTIONAL SUMMARY
{analysis_data.get('functional_summary', 'Not available')}

KEY GENES
{analysis_data.get('key_genes', 'Not available')}

CONSERVATION RATIONALE
{analysis_data.get('conservation_rationale', 'Not available')}

BIOLOGICAL SIGNIFICANCE
{analysis_data.get('biological_significance', 'Not available')}

ANALYSIS METADATA
{json.dumps(analysis.get('metadata', {}), indent=2)}
"""
            
            # Offer download options
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download as Text",
                    data=text_export,
                    file_name=f"elsa_analysis_{block_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                st.download_button(
                    label="📊 Download as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"elsa_analysis_{block_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.success("📥 Export options generated!")
            
        except Exception as e:
            st.error(f"Export failed: {e}")
            logger.error(f"Export failed for block {block_id}: {e}")
    
    def render_loading_state(self, block_id: str) -> None:
        """Render loading state while analysis is running."""
        st.markdown("---")
        st.subheader("🧠 AI Syntenic Analysis")
        
        with st.spinner(f"🤖 Analyzing block {block_id} with GPT-5..."):
            st.info("⏳ This may take 10-15 seconds depending on block complexity.")
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            for i in range(100):
                time.sleep(0.1)  # Simulate progress
                progress_bar.progress(i + 1)
                
                if i < 30:
                    status_text.text("🔍 Gathering syntenic block data...")
                elif i < 60:
                    status_text.text("🧬 Analyzing PFAM domains...")
                elif i < 90:
                    status_text.text("🤖 GPT-5 generating insights...")
                else:
                    status_text.text("✨ Finalizing analysis...")


def demo_analysis_panel():
    """Demo function to test the AI analysis panel."""
    st.title("ELSA AI Analysis Panel Demo")
    
    panel = AIAnalysisPanel()
    
    # Mock some analysis results for demo
    mock_analysis = {
        "block_id": "demo_001",
        "analysis": {
            "functional_summary": "This syntenic block contains essential DNA replication genes that are highly conserved across bacterial species, suggesting fundamental importance for cellular division.",
            "key_genes": "dnaA (replication initiation), dnaN (sliding clamp), polA (DNA polymerase I) - these form the core replication machinery.",
            "conservation_rationale": "High conservation (95% identity) reflects strong selective pressure to maintain DNA replication fidelity across bacterial lineages.",
            "biological_significance": "Critical for bacterial viability and potential target for antimicrobial drug development due to essential function."
        },
        "metadata": {
            "model": "gpt-4o",
            "timestamp": datetime.now().isoformat(),
            "block_length": 3,
            "similarity_score": 0.95,
            "num_genomes": 2,
            "num_pfam_domains": 5
        }
    }
    
    # Store mock analysis
    panel.store_analysis_result("demo_001", mock_analysis)
    
    # Render the panel
    panel.render_analysis_panel("demo_001")
    
    # Show trigger buttons
    st.markdown("---")
    st.subheader("Syntenic Blocks")
    
    mock_blocks = [
        {"id": "demo_001", "length": 3, "identity": 0.95},
        {"id": "demo_002", "length": 5, "identity": 0.87},
        {"id": "demo_003", "length": 7, "identity": 0.92}
    ]
    
    for block in mock_blocks:
        if panel.render_analysis_trigger(block["id"], block):
            st.success(f"Analysis triggered for block {block['id']}")


if __name__ == "__main__":
    demo_analysis_panel()