# Genome Browser: AI Navigation Improvements

## Current Pain Points for AI Agents

After exploring the genome browser as an AI agent (Claude), I identified several challenges:

### 1. No Deep Linking
- Tabs use Streamlit's native `st.tabs()` which doesn't support URL parameters
- Can't link directly to a specific block, cluster, or genome
- Every session starts from Dashboard

### 2. Session State Dependency
- Viewing a block requires: filter genomes → select block → click "View"
- State is lost on page reload
- No way to bookmark or share a specific view

### 3. Truncated Genome Names
- Sidebar shows `LC_01_R...`, `MancoS...` instead of full names
- Hard to identify which genome is which without expanding
- Borg color names (Olive_Borg, Blue_Borg) are more memorable than scaffold IDs

### 4. No Search Functionality
- Have to manually filter through dropdowns
- Can't search for specific genes, contigs, or patterns
- No "find similar blocks" feature

### 5. No Programmatic Access
- Everything requires visual interaction
- No REST API for querying data
- No export of current view state

---

## Proposed Improvements

### Priority 1: URL Query Parameters

Add URL-based navigation using `st.query_params`:

```python
# In app.py
params = st.query_params

# Tab navigation
if 'tab' in params:
    st.session_state.active_tab = params['tab']

# Deep linking
if 'block_id' in params:
    st.session_state.selected_block = int(params['block_id'])
    st.session_state.current_page = 'genome_viewer'

if 'cluster_id' in params:
    st.session_state.selected_cluster = int(params['cluster_id'])
    st.session_state.current_page = 'cluster_detail'

if 'genome' in params:
    st.session_state.genome_filter = [params['genome']]
```

**Examples:**
- `http://localhost:8501?tab=block_explorer`
- `http://localhost:8501?block_id=123`
- `http://localhost:8501?cluster_id=456&tab=cluster_explorer`
- `http://localhost:8501?genome=Olive_Borg&min_size=10`

### Priority 2: Better Genome Names

For Borg data, extract color names from genome IDs:

```python
def get_display_name(genome_id: str) -> str:
    """Extract human-readable name from genome ID."""
    # Pattern: ..._Color_Borg_##_##
    parts = genome_id.split('_')
    for i, part in enumerate(parts):
        if part == 'Borg' and i > 0:
            return f"{parts[i-1]}_Borg"
    return genome_id[:20]
```

Group genomes by class in sidebar:
```
📁 Borg_32 (5 genomes)
  ├─ Pink_Borg
  ├─ Red_Borg
  └─ ...
📁 Borg_33 (4 genomes)
  └─ ...
📁 Borg_34 (6 genomes)
  └─ ...
```

### Priority 3: Quick Navigation Panel

Add to sidebar:

```python
st.sidebar.markdown("### Quick Navigation")

# Jump to specific block
block_id = st.sidebar.number_input("Jump to Block ID", min_value=0, step=1)
if st.sidebar.button("Go to Block"):
    st.session_state.selected_block = block_id
    st.rerun()

# Search genes
gene_search = st.sidebar.text_input("Search genes/contigs")
if gene_search:
    # Filter blocks containing this gene/contig
    pass

# Find largest blocks
if st.sidebar.button("Show Largest Blocks"):
    st.session_state.sort_by = 'size'
    st.session_state.sort_desc = True
```

### Priority 4: Machine-Readable Summary Box

Add a collapsible section with parseable stats:

```python
with st.expander("📊 Summary (Machine Readable)"):
    summary = {
        "genomes": len(genomes_df),
        "blocks": len(blocks_df),
        "clusters": len(clusters_df),
        "genes": total_genes,
        "largest_block": {
            "id": largest_block.block_id,
            "size": largest_block.n_genes,
            "genomes": [largest_block.query_genome, largest_block.target_genome]
        }
    }
    st.code(json.dumps(summary, indent=2), language='json')
    st.button("Copy to Clipboard", on_click=lambda: pyperclip.copy(json.dumps(summary)))
```

### Priority 5: REST API (Future)

Add FastAPI endpoints alongside Streamlit:

```python
# api.py
from fastapi import FastAPI, Query
app = FastAPI()

@app.get("/api/blocks")
def get_blocks(
    genome: Optional[str] = None,
    min_size: int = 1,
    max_size: int = 1000,
    limit: int = 100
):
    """Query syntenic blocks."""
    ...

@app.get("/api/blocks/{block_id}")
def get_block(block_id: int):
    """Get specific block with gene details."""
    ...

@app.get("/api/clusters")
def get_clusters():
    """List all clusters."""
    ...

@app.get("/api/genes")
def search_genes(q: str = Query(...)):
    """Search genes by ID or annotation."""
    ...
```

---

## Quick Wins (Low Effort, High Impact)

1. **Show full genome names in dropdown** - Just increase width or use tooltip
2. **Add "Copy Block ID" button** - Easy clipboard access
3. **Sort by size by default** - Show interesting blocks first
4. **Add block count per genome pair** - In genome comparison matrix
5. **Keyboard shortcuts** - `n` for next block, `p` for previous

---

## Implementation Order

1. URL query params for tab navigation (30 min)
2. Better genome name display (15 min)
3. Quick navigation sidebar (1 hour)
4. Machine-readable summary (30 min)
5. REST API (separate project)

---

*Generated by Claude while exploring Borg genome synteny data, February 2026*
