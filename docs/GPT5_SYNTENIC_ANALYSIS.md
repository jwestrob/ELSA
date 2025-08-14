# GPT-5 Syntenic Block Analysis Feature

## Overview

Add AI-powered analysis to the ELSA genome browser that uses GPT-5 with high reasoning to provide biological interpretations of syntenic blocks. Users can click an "🧠 AI Explain" button to get insights about the functional significance and evolutionary patterns of selected blocks.

## Architecture

### Frontend (Streamlit)
- **Location**: New panel at top of genome browser page, above genome diagrams
- **Trigger**: "🧠 AI Explain" button on individual syntenic block rows
- **Display**: Collapsible analysis panel with structured results

### Backend (DSPy + GPT-5)
- **DSPy Signatures**: Structured prompts for biological analysis
- **GPT-5 Integration**: High reasoning mode for detailed interpretation
- **Data Pipeline**: Gather block data, PFAM domains, genome context

### User Experience Flow
1. User clicks "🧠 AI Explain" button on a syntenic block
2. Loading indicator shows "Analyzing with AI..."
3. Analysis panel expands at top of page with summary results
4. User can expand sections for more detail

## Implementation Plan

### Phase 1: Core Infrastructure
**Files to create:**
- `genome_browser/analysis/gpt5_analyzer.py` - GPT-5 integration with DSPy
- `genome_browser/analysis/data_collector.py` - Gather syntenic block data
- `genome_browser/components/ai_analysis_panel.py` - Streamlit UI component

**Files to modify:**
- `genome_browser/app.py` - Add analysis panel and API endpoints
- `genome_browser/requirements.txt` - Add DSPy and OpenAI dependencies

### Phase 2: DSPy Signatures
```python
class SyntenicBlockAnalysis(dspy.Signature):
    """Analyze a syntenic block for biological significance."""
    block_info = dspy.InputField(desc="Syntenic block coordinates, genes, similarity scores")
    pfam_domains = dspy.InputField(desc="PFAM domain annotations for genes in block")
    genome_context = dspy.InputField(desc="Source genome species and metadata")
    
    functional_summary = dspy.OutputField(desc="2-3 sentence summary of block function")
    conservation_pattern = dspy.OutputField(desc="Why this block is conserved across genomes")
    biological_significance = dspy.OutputField(desc="Evolutionary or clinical relevance")
    key_genes = dspy.OutputField(desc="Most important genes and their roles")
```

### Phase 3: Data Collection
**Input data for GPT-5:**
- Syntenic block metadata (coordinates, similarity, length)
- Gene list with PFAM domain annotations
- Genome species information
- Conservation patterns (which genes are core vs accessory)

### Phase 4: UI Integration
**Analysis panel structure:**
- Header: "AI Analysis: Block [ID]" with loading/error states
- Summary card with key insights
- Expandable sections: Function, Conservation, Significance
- Subtle styling to distinguish from main genome browser

## Technical Specifications

### GPT-5 Configuration
```python
{
    "model": "gpt-5",
    "reasoning": True,  # High reasoning mode
    "temperature": 0.1,  # Focused analysis
    "max_tokens": 2000,  # Summary-length responses
}
```

### API Endpoint
```python
@st.cache_data
def analyze_syntenic_block(block_id: str) -> dict:
    """Analyze a syntenic block with GPT-5."""
    # Gather data → DSPy analysis → Return structured results
```

### Error Handling
- API key validation
- Rate limiting awareness
- Graceful fallbacks for API failures
- Clear error messages to users

## User Interface Mockup

```
[ELSA Genome Browser]

┌─────────────────────────────────────────────────────────┐
│ 🧠 AI Analysis: Block syn_001                          │
│                                                         │
│ ✨ Summary: This syntenic block contains core DNA      │
│ replication genes highly conserved across E. coli      │
│ strains, suggesting essential cellular function.       │
│                                                         │
│ 🔬 Key Genes: dnaA (replication initiation), dnaN     │
│ (sliding clamp), polA (DNA polymerase I)              │
│                                                         │
│ 🧬 Conservation: 95% identity suggests strong         │
│ selective pressure to maintain replication fidelity   │
│                                                         │
│ [Show Details] [Export Analysis]                       │
└─────────────────────────────────────────────────────────┘

[Existing genome diagrams below...]
```

## Dependencies

### New Requirements
```txt
dspy-ai>=2.0.0
openai>=1.0.0
streamlit>=1.28.0
```

### Configuration
- Users provide their own OpenAI API keys
- No cost management needed initially
- Caching analysis results per session

## Success Criteria

1. **Functional**: Users can click button and get meaningful biological insights
2. **Fast**: Analysis completes within 10-15 seconds
3. **Informative**: Provides actionable biological interpretation
4. **Integrated**: Seamlessly fits into existing genome browser UI
5. **Reliable**: Handles API errors gracefully

## Future Enhancements (Not in MVP)

- Multi-block comparative analysis
- Deep dive analysis mode
- Analysis export to PDF/markdown
- Batch analysis for multiple blocks
- Cost tracking and usage limits

## Risk Mitigation

- **API Failures**: Cache successful analyses, show clear error messages
- **Cost Concerns**: User-provided API keys, clear usage expectations
- **Analysis Quality**: Use structured DSPy signatures for consistent output
- **Performance**: Cache results and show loading indicators

## Implementation Timeline

- **Week 1**: Core infrastructure and DSPy signatures
- **Week 2**: Data collection and GPT-5 integration  
- **Week 3**: UI components and testing
- **Week 4**: Integration testing and polish

This feature will make ELSA's genome browser uniquely powerful for biological interpretation!