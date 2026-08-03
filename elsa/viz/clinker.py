"""Clinker-style gene-neighbourhood rendering, independent of any UI framework.

These builders turn a locus/edge payload into a self-contained HTML+D3 document
and return it as a string. Nothing here imports Streamlit or touches a database,
so the same visualisation can be embedded in the browser, written to a file, or
served to an agent alongside the payload it was built from.

Payload shape (``build_multiloc_html``)::

    {"loci": [{"genome": str, "contig": str, "genes": [...]}, ...],
     "edges": [{"source": ..., "target": ..., "similarity": float}, ...]}
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def _pfam_color(label: str, alpha: float = 0.85) -> str:
    """Stable RGBA colour for a domain label.

    Hashing the label keeps a domain the same colour across every diagram and
    every run, which is what makes two clusters visually comparable.
    """
    seed = hashlib.md5(label.encode("utf-8")).hexdigest()
    r, g, b = (int(seed[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"

def comparative_to_multiloc(comp_data: Dict, block: Dict) -> Dict:
    """Convert comparative JSON (2-locus) to multiloc format for _render_multiloc_d3.

    Auto-detects inverted blocks by comparing strand directions of matched
    gene pairs. If most anchor pairs have opposite strands, the target track
    is auto-flipped so edges run parallel instead of crossing.
    """
    q_genes = comp_data.get('query_locus', [])
    t_genes = comp_data.get('target_locus', [])

    # Add fields expected by multiloc renderer
    for g in q_genes + t_genes:
        if 'aa' not in g:
            g['aa'] = max(1, abs(g.get('end', 0) - g.get('start', 0)) // 3)
        if 'contig' not in g:
            g['contig'] = ''

    def _locus(genes, name_key, genome_key, contig_key):
        return {
            'name': comp_data.get(name_key, name_key),
            'genome_id': block.get(genome_key, ''),
            'contig_id': block.get(contig_key, ''),
            'start': min((g['start'] for g in genes), default=0),
            'end': max((g['end'] for g in genes), default=0),
            'genes': genes,
        }

    loci = [
        _locus(q_genes, 'query_name', 'query_genome_id', 'query_contig_id'),
        _locus(t_genes, 'target_name', 'target_genome_id', 'target_contig_id'),
    ]

    # Convert edges: comparative {source, target, cos, pf} → multiloc {a, b, ai, bi, type, cos, pf}
    edges = []
    for e in comp_data.get('edges', []):
        cos_val = e.get('cos')
        edges.append({
            'a': 0, 'b': 1,
            'ai': int(e.get('source', 0)),
            'bi': int(e.get('target', 0)),
            'type': 'emb' if cos_val is not None else ('pfam' if int(e.get('pf', 0)) > 0 else 'anchor'),
            'cos': float(cos_val) if cos_val is not None else None,
            'pf': int(e.get('pf', 0)),
        })

    # Auto-detect inversion: compare strand directions of matched gene pairs.
    # If most edges connect genes with opposite strands, flip the target track.
    auto_flips = [False, False]
    same_strand = 0
    opp_strand = 0
    for e in edges:
        qi = e['ai']
        ti = e['bi']
        if qi < len(q_genes) and ti < len(t_genes):
            q_strand = q_genes[qi].get('strand', 1)
            t_strand = t_genes[ti].get('strand', 1)
            if q_strand == t_strand:
                same_strand += 1
            else:
                opp_strand += 1
    if opp_strand > same_strand:
        auto_flips[1] = True

    return {'loci': loci, 'edges': edges, 'auto_flips': auto_flips}


def consensus_from_data(data: Dict, min_cov: float) -> Tuple[List[Dict], List[Dict]]:
    """Derive consensus tokens/pairs from the loci used in the Clinker view."""

    loci = (data or {}).get('loci') or []
    total_tracks = len(loci)
    if total_tracks == 0:
        return [], []

    min_cov = max(0.0, min(1.0, float(min_cov)))

    # Union-find over genes across loci using clinker edges
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def _find(node: Tuple[int, int]) -> Tuple[int, int]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = _find(parent[node])
        return parent[node]

    def _union(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        if a not in parent or b not in parent:
            return
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # Initialise nodes
    for row_idx, locus in enumerate(loci):
        for gene_idx, _ in enumerate(locus.get('genes', [])):
            parent[(row_idx, gene_idx)] = (row_idx, gene_idx)

    # Merge via edges
    for edge in (data or {}).get('edges', []):
        a_row = int(edge.get('a', 0))
        b_row = int(edge.get('b', 0))
        a_idx = int(edge.get('ai', 0))
        b_idx = int(edge.get('bi', 0))
        node_a = (a_row, a_idx)
        node_b = (b_row, b_idx)
        if node_a in parent and node_b in parent:
            _union(node_a, node_b)

    components: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for node in parent:
        components[_find(node)].append(node)

    comp_infos: List[Dict] = []
    node_to_comp: Dict[Tuple[int, int], int] = {}
    for comp_id, (root, members) in enumerate(sorted(components.items(), key=lambda item: min(item[1]))):
        rows = set()
        pf_counts: Counter = Counter()
        orientation_votes: Dict[int, Counter] = defaultdict(Counter)
        for row_idx, gene_idx in members:
            node_to_comp[(row_idx, gene_idx)] = comp_id
            gene = loci[row_idx]['genes'][gene_idx]
            pf_tokens = [tok.strip() for tok in str(gene.get('pfam', '') or '').split(';') if tok.strip()]
            if pf_tokens:
                pf_counts.update([pf_tokens[0]])
            else:
                # Fallback to gene ID when PFAM missing
                pf_counts.update([f"~{gene.get('id', 'gene')}" ] )
            strand = gene.get('strand', 1)
            orientation_votes[row_idx].update([1 if int(strand) >= 0 else -1])
            rows.add(row_idx)

        orient_summary = {
            row: (1 if votes.get(1, 0) >= votes.get(-1, 0) else -1)
            for row, votes in orientation_votes.items()
        }

        comp_infos.append(
            {
                'id': comp_id,
                'members': members,
                'rows': rows,
                'pf_counts': pf_counts,
                'orientations': orient_summary,
            }
        )

    if not comp_infos:
        return [], []

    # Use first locus (highest-support ordering) as reference for token order
    reference_row = 0
    tokens: List[Dict] = []
    ordered_components: List[int] = []
    seen_components: Set[int] = set()
    ref_genes = loci[reference_row]['genes'] if loci[reference_row]['genes'] else []
    for gene_idx, gene in enumerate(ref_genes):
        comp_id = node_to_comp.get((reference_row, gene_idx))
        if comp_id is None or comp_id in seen_components:
            continue
        info = comp_infos[comp_id]
        coverage = len(info['rows']) / float(total_tracks)
        if coverage < min_cov:
            continue
        label_raw, _ = info['pf_counts'].most_common(1)[0] if info['pf_counts'] else (gene.get('id', f"gene_{gene_idx}"), 0)
        label = label_raw[1:] if label_raw.startswith('~') else label_raw
        color = _pfam_color(label)
        orient_values = info['orientations'].values()
        fwd_frac = (sum(1 for val in orient_values if val >= 0) / len(orient_values)) if orient_values else 0.0
        tokens.append(
            {
                'token': label,
                'color': color,
                'coverage': coverage,
                'fwd_frac': fwd_frac,
                'n_occ': len(info['members']),
                '_comp': comp_id,
            }
        )
        ordered_components.append(comp_id)
        seen_components.add(comp_id)

    if not tokens:
        return [], []

    # Compute directional consensus between adjacent tokens
    pairs: List[Dict] = []
    for idx in range(len(tokens) - 1):
        comp_a = comp_infos[tokens[idx]['_comp']]
        comp_b = comp_infos[tokens[idx + 1]['_comp']]
        shared_rows = comp_a['rows'] & comp_b['rows']
        if not shared_rows:
            continue
        same = 0
        total = 0
        for row in shared_rows:
            or_a = comp_a['orientations'].get(row)
            or_b = comp_b['orientations'].get(row)
            if or_a is None or or_b is None:
                continue
            if (or_a >= 0 and or_b >= 0) or (or_a < 0 and or_b < 0):
                same += 1
            total += 1
        if total:
            pairs.append({'i': idx, 'j': idx + 1, 'same_frac': same / total, 'support': total})

    for token in tokens:
        token.pop('_comp', None)

    return tokens, pairs


def build_comparative_html(data: Dict, width: int = 0, height: int = 500):
    """Render clinker-like comparative view using D3 in a Streamlit component.

    Notes:
    - Width is responsive to the container (100%). The `width` arg is ignored.
    - X positions are normalized per locus to fill available width while preserving
      intra-locus genomic spacing (bp-proportional within each track).
    """
    import json
    import base64
    payload = json.dumps(data)
    payload_b64 = base64.b64encode(payload.encode('utf-8')).decode('ascii')
    html = """
    <div id=\"cmp\" style=\"width:100%; height:__HEIGHT__px;\" data-payload=\"__PAYLOAD_B64__\"></div>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
    (function(){
      const container = document.getElementById('cmp');
      let data;
      try {
        const payloadB64 = container.dataset.payload || '';
        data = JSON.parse(atob(payloadB64));
      } catch (e) {
        container.innerHTML = '<div style="color:red; font-family:monospace">Failed to parse data: ' + (e && e.message ? e.message : e) + '</div>';
        return;
      }
      const fixedHeight = __HEIGHT__;
      const margin = {top: 20, right: 260, bottom: 100, left: 20};


      function arrowPath(x, y, w, h, strand) {
        if (strand >= 0) {
          return 'M' + x + ',' + y + ' h' + (w - h/2) + ' l' + (h/2) + ',' + (h/2) + ' l-' + (h/2) + ',' + (h/2) + ' h-' + (w - h/2) + ' z';
        } else {
          return 'M' + (x + w) + ',' + y + ' h-' + (w - h/2) + ' l-' + (h/2) + ',' + (h/2) + ' l' + (h/2) + ',' + (h/2) + ' h' + (w - h/2) + ' z';
        }
      }

      function render() {
        const width = container.clientWidth || 800;
        const height = fixedHeight;
        container.innerHTML = '';

        const svg = d3.select(container).append('svg')
          .attr('width', width)
          .attr('height', height)
          .style('background', 'transparent');

        const innerW = Math.max(10, width - margin.left - margin.right);
        const innerH = Math.max(10, height - margin.top - margin.bottom);

        const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        // Y anchors (leave space for labels and legend)
        const qY = 90;
        const tY = innerH - 70;
        const geneH = 12;
        const minGeneW = 6; // minimum pixel width for visibility

        // Scales: normalize bp per locus to full width
        const qMin = d3.min(data.query_locus, d => d.start) || 0;
        const qMax = d3.max(data.query_locus, d => d.end) || 1;
        const tMin = d3.min(data.target_locus, d => d.start) || 0;
        const tMax = d3.max(data.target_locus, d => d.end) || 1;
        const qX = d3.scaleLinear().domain([qMin, qMax]).range([0, innerW]);
        const tX = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);

        // Draw query genes (top)
        const q = g.append('g');
        try {
          q.selectAll('path.gene')
            .data(data.query_locus || [])
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = qX(d.start);
              const x1 = qX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1); // handle any reversed coords
              return arrowPath(x, qY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5)
            .append('title').text(d=> (d.id + '\nPFAM: ' + (d.pfam || 'None')));
        } catch (e) {
          q.append('text').attr('x',10).attr('y',qY).attr('fill','red').text('Render error (query): ' + e);
        }

        // Draw target genes (bottom)
        const t = g.append('g');
        try {
          t.selectAll('path.gene')
            .data(data.target_locus || [])
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = tX(d.start);
              const x1 = tX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1);
              return arrowPath(x, tY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5)
            .append('title').text(d=> (d.id + '\nPFAM: ' + (d.pfam || 'None')));
        } catch (e) {
          t.append('text').attr('x',10).attr('y',tY).attr('fill','red').text('Render error (target): ' + e);
        }

        // Edges (curves between centers)
        const edges = g.append('g').attr('opacity',0.7);
        try {
          edges.selectAll('path.edge')
            .data(data.edges || [])
            .enter().append('path')
            .attr('class','edge')
            .attr('fill','none')
            .attr('stroke', d=> d.score>0.8? '#2c7bb6' : d.score>0.5? '#abd9e9':'#fdae61')
            .attr('stroke-width', d=> 1 + 2*(d.score||0.5))
            .attr('d', d=> {
              const qs = data.query_locus[d.source] || {start: qMin, end: qMin};
              const ts = data.target_locus[d.target] || {start: tMin, end: tMin};
              const x1 = (qX(qs.start) + qX(qs.end)) / 2, y1 = qY + geneH/2;
              const x2 = (tX(ts.start) + tX(ts.end)) / 2, y2 = tY + geneH/2;
              const mx = (x1 + x2) / 2;
              return 'M' + x1 + ',' + y1 + ' C' + mx + ',' + (y1-60) + ' ' + mx + ',' + (y2+60) + ' ' + x2 + ',' + y2;
            })
          .append('title').text(d=> ('score=' + ((d.score != null) ? d.score : '')));
        } catch (e) {
          g.append('text').attr('x',10).attr('y',(qY+tY)/2).attr('fill','red').text('Render error (edges): ' + e);
        }
      }

      // Initial render and resize handling
      render();
      if ('ResizeObserver' in window) {
        const ro = new ResizeObserver(() => render());
        ro.observe(container);
      } else {
        window.addEventListener('resize', render);
      }
    })();
    </script>
    """
    html = html.replace('__PAYLOAD_B64__', payload_b64).replace('__HEIGHT__', str(height))
    return html


def build_comparative_html_v2(data: Dict, width: int = 0, height: int = 500):
    """Alternate renderer using inline JSON script tag (safer for parsing)."""
    import json
    payload = json.dumps(data)
    html = """
    <div id=\"cmp\" style=\"width:100%; height:__HEIGHT__px;\"></div>
    <script id=\"cmp-data\" type=\"application/json\">__PAYLOAD_JSON__</script>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
    (function(){
      try {
        const container = document.getElementById('cmp');
        const dataText = document.getElementById('cmp-data').textContent || '{}';
        const data = JSON.parse(dataText);
        const fixedHeight = __HEIGHT__;
        // Add extra bottom margin to make room for tooltips below bottom track
        const margin = {top: 20, right: 20, bottom: 100, left: 20};

        function arrowPath(x, y, w, h, strand) {
          if (strand >= 0) {
            return 'M' + x + ',' + y + ' h' + (w - h/2) + ' l' + (h/2) + ',' + (h/2) + ' l-' + (h/2) + ',' + (h/2) + ' h-' + (w - h/2) + ' z';
          } else {
            return 'M' + (x + w) + ',' + y + ' h-' + (w - h/2) + ' l-' + (h/2) + ',' + (h/2) + ' l' + (h/2) + ',' + (h/2) + ' h' + (w - h/2) + ' z';
          }
        }

        function render() {
          const width = container.clientWidth || 800;
          const height = fixedHeight;
          container.innerHTML = '';

          const svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('background', 'transparent');

          const innerW = Math.max(10, width - margin.left - margin.right);
          const innerH = Math.max(10, height - margin.top - margin.bottom);

          const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

          const qY = 40;
          const tY = innerH - 40;
          const geneH = 12;
          const minGeneW = 6;

          const qMin = d3.min(data.query_locus || [], d => d.start) || 0;
          const qMax = d3.max(data.query_locus || [], d => d.end) || 1;
          const tMin = d3.min(data.target_locus || [], d => d.start) || 0;
          const tMax = d3.max(data.target_locus || [], d => d.end) || 1;
          const qX = d3.scaleLinear().domain([qMin, qMax]).range([0, innerW]);
          const tX = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);

          // Tooltip
          container.style.position = 'relative';
          const tooltip = d3.select(container).append('div')
            .attr('class','cmp-tooltip')
            .style('position','absolute')
            .style('pointer-events','none')
            .style('background','#2a2a2a')
            .style('color','#ddd')
            .style('border','1px solid #555')
            .style('padding','6px 8px')
            .style('font','12px monospace')
            .style('border-radius','4px')
            .style('box-shadow','0 2px 6px rgba(0,0,0,0.15)')
            .style('display','none')
            .style('z-index','10');

          const qData = (data.query_locus || []).map((d,i)=>Object.assign({_idx:i}, d));
          const tData = (data.target_locus || []).map((d,i)=>Object.assign({_idx:i}, d));

          const q = g.append('g');
          const qGenes = q.selectAll('path.gene')
            .data(qData)
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = qX(d.start);
              const x1 = qX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1);
              return arrowPath(x, qY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5);

          qGenes
            .on('mouseover', function(event, d){
              const strand = (d.strand>=0? '+':'-');
              const len = Math.max(0, Math.round(Math.abs(d.end - d.start)));
              // Cosine similarity (max across connected edges)
              let simLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.source === d._idx);
                if (matches.length) {
                  const maxSim = d3.max(matches, ed => (ed.cos != null ? +ed.cos : -Infinity));
                  if (maxSim != null && isFinite(maxSim)) simLine = '<br>Cosine: ' + (+maxSim).toFixed(3);
                }
              } catch (e) {}
              // PFAM overlap (max across connected edges)
              let pfLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.source === d._idx && ed.pf != null);
                if (matches.length) {
                  const maxPf = d3.max(matches, ed => (+ed.pf || 0));
                  if (maxPf != null && isFinite(maxPf) && maxPf > 0) pfLine = '<br>PFAM overlap: ' + maxPf;
                }
              } catch (e) {}
              tooltip.style('display','block')
                     .html('<b>' + d.id + '</b><br>' +
                           'Length: ' + len.toLocaleString() + ' aa<br>' +
                           'Strand: ' + strand + '<br>' +
                           'PFAM: ' + (d.pfam || 'None') + simLine + pfLine);
              d3.select(this).attr('stroke-width',2);
              highlightFor('q', d._idx);
            })
            .on('mousemove', function(event){
              const pt = d3.pointer(event, container);
              // Keep tooltip within container bounds and away from labels
              const cw = container.clientWidth || 0;
              const ch = container.clientHeight || 0;
              const tw = tooltip.node().offsetWidth || 0;
              const th = tooltip.node().offsetHeight || 0;
              let left = pt[0] + 12;
              let top = pt[1] + 12;
              // Avoid overlapping top label band (~30px) and bottom label band (~30px)
              if (top < 30) top = 34;
              if (top > ch - 34 - th) top = ch - 34 - th;
              if (left + tw > cw - 4) left = Math.max(4, pt[0] - 12 - tw);
              if (top + th > ch - 4) top = Math.max(4, pt[1] - 12 - th);
              tooltip.style('left', left + 'px').style('top', top + 'px');
            })
            .on('mouseout', function(){
              tooltip.style('display','none');
              d3.select(this).attr('stroke-width',0.5);
              clearHighlight();
            });

          const t = g.append('g');
          const tGenes = t.selectAll('path.gene')
            .data(tData)
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = tX(d.start);
              const x1 = tX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1);
              return arrowPath(x, tY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5);

          tGenes
            .on('mouseover', function(event, d){
              const strand = (d.strand>=0? '+':'-');
              const len = Math.max(0, Math.round(Math.abs(d.end - d.start)));
              // Cosine similarity (max across connected edges)
              let simLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.target === d._idx);
                if (matches.length) {
                  const maxSim = d3.max(matches, ed => (ed.cos != null ? +ed.cos : -Infinity));
                  if (maxSim != null && isFinite(maxSim)) simLine = '<br>Cosine: ' + (+maxSim).toFixed(3);
                }
              } catch (e) {}
              // PFAM overlap (max across connected edges)
              let pfLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.target === d._idx && ed.pf != null);
                if (matches.length) {
                  const maxPf = d3.max(matches, ed => (+ed.pf || 0));
                  if (maxPf != null && isFinite(maxPf) && maxPf > 0) pfLine = '<br>PFAM overlap: ' + maxPf;
                }
              } catch (e) {}
              tooltip.style('display','block')
                     .html('<b>' + d.id + '</b><br>' +
                           'Length: ' + len.toLocaleString() + ' aa<br>' +
                           'Strand: ' + strand + '<br>' +
                           'PFAM: ' + (d.pfam || 'None') + simLine + pfLine);
              d3.select(this).attr('stroke-width',2);
              highlightFor('t', d._idx);
            })
            .on('mousemove', function(event){
              const pt = d3.pointer(event, container);
              // Keep tooltip within container bounds and away from labels
              const cw = container.clientWidth || 0;
              const ch = container.clientHeight || 0;
              const tw = tooltip.node().offsetWidth || 0;
              const th = tooltip.node().offsetHeight || 0;
              let left = pt[0] + 12;
              let top = pt[1] + 12;
              if (top < 30) top = 34;
              if (top > ch - 34 - th) top = ch - 34 - th;
              if (left + tw > cw - 4) left = Math.max(4, pt[0] - 12 - tw);
              if (top + th > ch - 4) top = Math.max(4, pt[1] - 12 - th);
              tooltip.style('left', left + 'px').style('top', top + 'px');
            })
            .on('mouseout', function(){
              tooltip.style('display','none');
              d3.select(this).attr('stroke-width',0.5);
              clearHighlight();
            });

          const edges = g.append('g').attr('opacity',0.7);
          const eSel = edges.selectAll('path.edge')
            .data(data.edges || [])
            .enter().append('path')
            .attr('class','edge')
            .attr('fill','none')
            .attr('stroke', d=> {
              if (d.type === 'both') return '#7b3294';       // purple
              if (d.type === 'cos') return '#2c7bb6';        // blue
              if (d.type === 'pfam') return '#fdae61';       // orange
              return '#999999';                               // fallback/grey
            })
            .attr('stroke-width', d=> {
              if (d.type === 'cos' || d.type === 'both') {
                const c = Math.max(0, Math.min(1, +d.cos || 0.5));
                return 2 + 4*c; // thicker scaling for cosine/both
              }
              if (d.type === 'pfam') {
                const p = Math.max(0, Math.min(3, +d.pf || 1));
                return 2 + 2*(p/3); // thicker PFAM lines
              }
              return 2.0; // fallback thickness
            })
            .attr('stroke-opacity', 0.6)
            .attr('d', d=> {
              const qs = data.query_locus[d.source] || {start: qMin, end: qMin};
              const ts = data.target_locus[d.target] || {start: tMin, end: tMin};
              const x1 = (qX(qs.start) + qX(qs.end)) / 2, y1 = qY + geneH/2;
              const x2 = (tX(ts.start) + tX(ts.end)) / 2, y2 = tY + geneH/2;
              // Straight line connection
              return 'M' + x1 + ',' + y1 + ' L' + x2 + ',' + y2;
            })
            .append('title').text(d=> {
              let parts = [];
              if (d.cos != null) parts.push('cosine=' + (+d.cos).toFixed(3));
              if (d.pf != null) parts.push('pfam_overlap=' + d.pf);
              if (!parts.length) parts.push('fallback');
              return parts.join(', ');
            });

          // Labels for loci
          const topLabel = (data.query_name || 'Query locus');
          const botLabel = (data.target_name || 'Target locus');
          g.append('text')
            .attr('x', 0)
            .attr('y', qY - 20)
            .attr('fill', '#ddd')
            .attr('font-size', 12)
            .attr('font-family', 'sans-serif')
            .text(topLabel);
          g.append('text')
            .attr('x', 0)
            .attr('y', tY + geneH + 18)
            .attr('fill', '#ddd')
            .attr('font-size', 12)
            .attr('font-family', 'sans-serif')
            .text(botLabel);

          // Legend (top-right)
          const legend = g.append('g');
          const lx = Math.max(0, innerW - 260), ly = 8;
          legend.append('rect').attr('x', lx-6).attr('y', ly-6).attr('width', 252).attr('height', 60)
                .attr('fill', 'rgba(30,30,30,0.85)').attr('stroke', '#555');
          legend.append('rect').attr('x', lx).attr('y', ly).attr('width', 14).attr('height', 14).attr('fill', '#2E8B57').attr('stroke','#666');
          legend.append('rect').attr('x', lx+18).attr('y', ly).attr('width', 14).attr('height', 14).attr('fill', '#FF6347').attr('stroke','#666');
          legend.append('text').attr('x', lx+36).attr('y', ly+11).attr('font-size', 12).attr('fill','#ddd').text('Gene arrows (strand color)');
          // Cosine edge
          legend.append('line').attr('x1', lx).attr('y1', ly+26).attr('x2', lx+36).attr('y2', ly+26).attr('stroke', '#2c7bb6').attr('stroke-width', 3);
          legend.append('text').attr('x', lx+44).attr('y', ly+29).attr('font-size', 12).attr('fill','#ddd').text('Cosine homology');
          // PFAM edge
          legend.append('line').attr('x1', lx).attr('y1', ly+42).attr('x2', lx+36).attr('y2', ly+42).attr('stroke', '#fdae61').attr('stroke-width', 3);
          legend.append('text').attr('x', lx+44).attr('y', ly+45).attr('font-size', 12).attr('fill','#ddd').text('PFAM homology');
          // Both edge
          legend.append('line').attr('x1', lx+130).attr('y1', ly+34).attr('x2', lx+166).attr('y2', ly+34).attr('stroke', '#7b3294').attr('stroke-width', 3);
          legend.append('text').attr('x', lx+174).attr('y', ly+37).attr('font-size', 12).attr('fill','#ddd').text('Both');

          // Cross-track highlighting
          function clearHighlight() {
            eSel.attr('stroke-opacity', 0.6).attr('stroke-width', d=> 1 + 2*(d.score||0.5));
            qGenes.attr('opacity', 1.0).attr('stroke-width', 0.5);
            tGenes.attr('opacity', 1.0).attr('stroke-width', 0.5);
          }

          function highlightFor(side, idx) {
            // Determine matching edges and counterpart indices
            const matches = [];
            (data.edges || []).forEach((ed, i) => {
              if (side === 'q' && ed.source === idx) matches.push({ed, i});
              if (side === 't' && ed.target === idx) matches.push({ed, i});
            });

            // Dim everything
            eSel.attr('stroke-opacity', 0.15);
            qGenes.attr('opacity', 0.4).attr('stroke-width', 0.5);
            tGenes.attr('opacity', 0.4).attr('stroke-width', 0.5);

            // Highlight hovered gene
            if (side === 'q') {
              qGenes.filter(d => d._idx === idx).attr('opacity', 1.0).attr('stroke-width', 2);
            } else {
              tGenes.filter(d => d._idx === idx).attr('opacity', 1.0).attr('stroke-width', 2);
            }

            // Highlight counterpart genes and edges
            matches.forEach(({ed}) => {
              eSel.filter(d => d.source === ed.source && d.target === ed.target)
                  .attr('stroke-opacity', 0.95)
                  .attr('stroke-width', d=> 2.5 + 2*(d.score||0.5));
              qGenes.filter(d => d._idx === ed.source).attr('opacity', 1.0).attr('stroke-width', 2);
              tGenes.filter(d => d._idx === ed.target).attr('opacity', 1.0).attr('stroke-width', 2);
            });
          }

        }

        render();
        if ('ResizeObserver' in window) {
          const ro = new ResizeObserver(() => render());
          ro.observe(container);
        } else {
          window.addEventListener('resize', render);
        }
      } catch (e) {
        const el = document.getElementById('cmp');
        if (el) {
          el.innerHTML = '<div style=\\"color:red; font-family:monospace\\">Component error: ' + (e && e.message ? e.message : e) + '</div>';
        }
      }
    })();
    </script>
    """
    # Escape closing tag sequences in JSON
    safe_json = payload.replace('</', '<\\/')
    html = html.replace('__PAYLOAD_JSON__', safe_json).replace('__HEIGHT__', str(height))
    return html


def build_multiloc_html(data: Dict, width: int = 0, height: int = 500,
                        initial_tau: float = 0.85, show_labels: bool = True,
                        scale_edge_width: bool = True):
    """Render stacked multi-locus clinker-like view connecting adjacent tracks by PFAM matches.

    Args:
        data: Dict with 'loci' and 'edges' from _build_cluster_multiloc_json.
        width: Fixed width (0 = responsive to container).
        height: Component height in pixels.
        initial_tau: Initial cosine similarity threshold for edge filtering.
        show_labels: Whether to show cosine similarity labels on edges.
        scale_edge_width: Whether to scale edge line width by similarity.
    """
    import json as _json
    if not data or not data.get('loci'):
        return ""
    html = """
    <div id=\"cluster-multiloc\" style=\"width:100%; position:relative; padding:8px 16px; box-sizing:border-box; background:#1a1a2e; color:#ddd; min-height:100%;\"></div>
    <style>
      /* The diagram and its controls were authored against a dark host. Carrying
         that background here keeps the control strip legible instead of drawing
         #ddd text onto whatever the embedding page happens to be. */
      html, body { margin:0; padding:0; background:#1a1a2e; }
      #cluster-multiloc input[type=range] { vertical-align:middle; accent-color:#7fb8a4; }
      #cluster-multiloc button { background:#2a2a3e; color:#ddd; border:1px solid #444;
                                 border-radius:4px; padding:2px 8px; cursor:pointer; }
      #cluster-multiloc button:hover { border-color:#7fb8a4; }
      #cluster-multiloc label { user-select:none; }
    </style>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
      const container = document.getElementById('cluster-multiloc');
      const data = __DATA__;
      const margin = {top: 10, right: 20, bottom: 20, left: 20};
      const W = __WIDTH__;
      const H = __HEIGHT__;
      const INITIAL_TAU = __INITIAL_TAU__;
      const SHOW_LABELS = __SHOW_LABELS__;
      const SCALE_EDGE_WIDTH = __SCALE_EDGE_WIDTH__;
      const baseInnerW = (W>0?W:container.clientWidth) - margin.left - margin.right;
      const minZoom = 0.6, maxZoom = 3.0;
      const rowH = 60;
      const rowGap = 32;
      const contentH = (rowH+rowGap)*data.loci.length;
      const svg = d3.select(container).append('svg')
          .attr('width', baseInnerW + margin.left + margin.right)
          .attr('height', margin.top + margin.bottom + contentH)
          .style('background', 'transparent');
      const root = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const panG = root.append('g').attr('class','panzoom');

      // Tooltip similar to block explorer
      const tooltip = d3.select(container).append('div')
        .style('position','absolute')
        .style('pointer-events','none')
        .style('background','#2a2a2a')
        .style('color','#ddd')
        .style('border','1px solid #555')
        .style('padding','6px 8px')
        .style('font','12px monospace')
        .style('border-radius','4px')
        .style('box-shadow','0 2px 6px rgba(0,0,0,0.15)')
        .style('display','none')
        .style('z-index','10');

      function colorPF(p) { let h=0; for (let i=0;i<p.length;i++) { h=(h*31 + p.charCodeAt(i))>>>0; } const r=(h&0xFF), g2=((h>>8)&0xFF), b=((h>>16)&0xFF); return `rgba(${r},${g2},${b},0.9)`; }

      // Controls and state
      const controls = d3.select(container).append('div').style('margin','6px 0').style('font-size','14px').style('color','#ddd');
      let offsets = data.loci.map(() => 0);
      // Initialize flips from auto_flips (e.g. auto-flip inverted blocks) or default to false
      let flips = data.loci.map((_, i) => (data.auto_flips && data.auto_flips[i]) || false);
      let tau = INITIAL_TAU;
      let showLabels = SHOW_LABELS;
      let scaleWidth = SCALE_EDGE_WIDTH;
      let view = d3.zoomIdentity.scale(1.25);
      controls.append('button').text('Reset alignment').on('click', () => { offsets = offsets.map(() => 0); redraw(); });
      controls.append('button').text('Reset view').style('margin-left','8px').on('click', () => { svg.transition().duration(150).call(zoomBehavior.transform, d3.zoomIdentity.scale(1.25)); });
      controls.append('button').text('Flip all to forward').style('margin-left','8px').on('click', () => { const refOri=rowOrientation(0); flips = flips.map((f,i)=> rowOrientation(i)===refOri?false:true); redraw(); });
      controls.append('button').text('Export SVG').style('margin-left','8px').on('click', () => {
        // Clone SVG, add white background for readability, set viewBox, serialize
        const svgNode = svg.node();
        const clone = svgNode.cloneNode(true);
        // Set explicit dimensions and dark background
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        clone.style.background = '#1a1a2e';
        // Add a background rect as first child
        const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bgRect.setAttribute('width', '100%');
        bgRect.setAttribute('height', '100%');
        bgRect.setAttribute('fill', '#1a1a2e');
        clone.insertBefore(bgRect, clone.firstChild);
        const serializer = new XMLSerializer();
        const svgStr = serializer.serializeToString(clone);
        const blob = new Blob([svgStr], {type: 'image/svg+xml;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'clinker_alignment.svg';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
      controls.append('button').text('Zoom -').style('margin-left','8px').on('click', () => { const k = Math.max(minZoom, +(view.k-0.1).toFixed(2)); svg.transition().duration(120).call(zoomBehavior.scaleTo, k); });
      controls.append('button').text('Zoom +').style('margin-left','4px').on('click', () => { const k = Math.min(maxZoom, +(view.k+0.1).toFixed(2)); svg.transition().duration(120).call(zoomBehavior.scaleTo, k); });
      const zoomLbl = controls.append('span').style('margin-left','8px').style('color','#ddd').text(() => ` (${(view.k).toFixed(2)}x)`);
      controls.append('span').style('margin-left','12px').style('color','#ddd').style('font-weight','600').text('Cosine \\u03C4:');
      // Span the slider over the cosines actually present. PLM embedding space
      // is anisotropic, so anchored proteins routinely sit above 0.98 and a
      // fixed 0.50-0.99 range leaves every edge drawn at every position — the
      // control appears to work while changing nothing.
      const cosVals = (data.edges || []).map(e => e.cos).filter(v => v != null);
      const cosLo = cosVals.length ? Math.min.apply(null, cosVals) : 0.50;
      const cosHi = cosVals.length ? Math.max.apply(null, cosVals) : 0.99;
      const pad = Math.max(0.002, (cosHi - cosLo) * 0.05);
      const tauMin = Math.max(0, cosLo - pad), tauMax = Math.min(1, cosHi + pad);
      const tauStep = Math.max(0.0005, (tauMax - tauMin) / 100);
      if (tau < tauMin || tau > tauMax) tau = tauMin;
      const tauInput = controls.append('input')
        .attr('type','range').attr('min', tauMin.toFixed(4)).attr('max', tauMax.toFixed(4))
        .attr('step', tauStep.toFixed(4)).attr('value', tau.toFixed(4))
        .on('input', function(){ tau=+this.value; tauLbl.text(` ${tau.toFixed(3)}`); drawEdges(); });
      const tauLbl = controls.append('span').style('margin-left','6px').style('color','#ddd').text(` ${tau.toFixed(3)}`);
      controls.append('span').style('margin-left','6px').style('color','#8a8f98').style('font-size','12px')
        .text(`(data ${cosLo.toFixed(3)}\u2013${cosHi.toFixed(3)})`);

      // Label toggle (in-component, mirrors the Streamlit checkbox)
      const labelChk = controls.append('label').style('margin-left','16px').style('color','#ddd').style('cursor','pointer');
      labelChk.append('input').attr('type','checkbox').property('checked', showLabels)
        .on('change', function(){ showLabels = this.checked; drawEdges(); });
      labelChk.append('span').style('margin-left','4px').text('Labels');

      // Width-scaling toggle
      const widthChk = controls.append('label').style('margin-left','12px').style('color','#ddd').style('cursor','pointer');
      widthChk.append('input').attr('type','checkbox').property('checked', scaleWidth)
        .on('change', function(){ scaleWidth = this.checked; drawEdges(); });
      widthChk.append('span').style('margin-left','4px').text('Scale width');

      // Legend (edge meaning)
      const legend = d3.select(container).append('div').style('margin','4px 0 8px 0').style('font','14px sans-serif').style('color','#ddd');
      function legendItem(color, label){
        const item = legend.append('span').style('display','inline-flex').style('align-items','center').style('margin-right','14px');
        item.append('span').style('display','inline-block').style('width','18px').style('height','3px').style('margin-right','6px').style('background', color).style('border','1px solid rgba(255,255,255,0.2)');
        item.append('span').text(label);
      }
      legendItem('#2c7bb6', 'Cosine-only');
      legendItem('#fdae61', 'PFAM-only');
      legendItem('#7b3294', 'Both');
      try {
        const emb = (data.embeddings || {});
        const msg = (emb.enabled && emb.mapped>0) ? `Embeddings ON (${emb.mapped.toLocaleString()} genes mapped)` : 'Embeddings OFF (PFAM-only)';
        legend.append('span').style('margin-left','14px').style('opacity','0.7').text(msg);
      } catch (e) {}
      legend.append('div')
        .style('margin-top','6px')
        .style('opacity','0.7')
        .text('Tips: drag background to pan | wheel or buttons to zoom | click a gene to center homologs | double-click a gene or click row label to flip | drag a row to nudge | use Cosine \\u03C4 to filter edges.');

      function rowOrientation(idx){ const genes = data.loci[idx].genes||[]; let s=0; genes.forEach(g=>{ s += (g.strand>=0?1:-1); }); return s>=0?1:-1; }

      const rowsG = [];
      const gapPx = 16; // fixed inter-gene gap in pixels
      const minGenePx = 3; // minimal visible width per gene
      let pxPerAA = 1; // computed from max row extent
      data.loci.forEach((loc, idx) => {
        const y0 = idx*(rowH+rowGap);
        const minX = d3.min(loc.genes, d => d.start) || 0;
        const maxX = d3.max(loc.genes, d => d.end) || 1;
        const baseX = d3.scaleLinear().domain([minX, maxX]).range([0, baseInnerW]);
        // Stacked label, one line per '  |  ' segment. Earlier this rendered
        // only the first two and silently dropped the rest, so a label carrying
        // lineage, accession and contig lost the contig without any sign.
        const labelParts = (loc.name||'').split('  |  ').filter(t => t !== '');
        const labelG = panG.append('text').attr('x', 0)
          .attr('y', y0 - 12 - Math.max(0, labelParts.length - 2) * 11)
          .attr('fill','#ddd').style('cursor','pointer')
          .on('click', () => { flips[idx] = !flips[idx]; redraw(); });
        if (labelParts.length >= 2) {
          labelG.append('tspan').text(labelParts[0])
            .attr('font-size','13px').attr('font-weight','bold').attr('x', 0);
          labelParts.slice(1).forEach((part, n) => {
            labelG.append('tspan').text(part)
              .attr('font-size','10px')
              .attr('fill', n === labelParts.length - 2 ? '#8a8f98' : '#aab2bd')
              .attr('x', 0).attr('dy', n === 0 ? '12' : '11');
          });
        } else {
          labelG.text(loc.name).attr('font-size','12px');
        }
        const rowSel = panG.append('g').attr('transform', `translate(0,${y0})`);
        const row = rowSel.node();
        row._baseX = baseX; rowsG.push(row);
      });

      // Precompute per-row gene layouts in pixels with a global px-per-aa so equal protein lengths have equal widths across rows
      function computeLayouts(){
        // determine global pxPerAA from the row with max total aa length
        let maxSumAA = 1;
        let maxGenes = 1;
        data.loci.forEach(loc => {
          const aaSum = (loc.genes||[]).reduce((s,g)=> s + Math.max(1, +g.aa || 1), 0);
          if (aaSum > maxSumAA) maxSumAA = aaSum;
          const n = (loc.genes||[]).length;
          if (n > maxGenes) maxGenes = n;
        });
        pxPerAA = Math.max(0.05, (baseInnerW - gapPx * Math.max(0, maxGenes-1)) / maxSumAA);
        // assign layout per row
        data.loci.forEach((loc, idx) => {
          const row = rowsG[idx];
          let x = 0;
          const lay = [];
          (loc.genes||[]).forEach(g => {
            const w = Math.max(minGenePx, (Math.max(1, +g.aa || 1)) * pxPerAA);
            lay.push({start:x, w:w});
            x += w + gapPx;
          });
          row._layout = lay;
          row._rowWidth = x > 0 ? (x - gapPx) : 0;
        });
      }
      computeLayouts();

      // Compute initial horizontal offsets to align matched genes between adjacent tracks
      (function alignTracks(){
        // For each adjacent pair, find the best-cosine edge and align on it
        const adjEdges = (data.edges||[]).filter(e => e.b === e.a + 1 && e.cos != null && isFinite(e.cos));
        // Group edges by (a,b) pair, pick highest cosine per pair
        const bestByPair = new Map();
        adjEdges.forEach(e => {
          const key = e.a + '-' + e.b;
          if (!bestByPair.has(key) || e.cos > bestByPair.get(key).cos) bestByPair.set(key, e);
        });
        // Propagate offsets from track 0 (anchor at 0)
        for (let k = 0; k < data.loci.length - 1; k++) {
          const key = k + '-' + (k+1);
          const e = bestByPair.get(key);
          if (!e) continue;
          const rowA = rowsG[k], rowB = rowsG[k+1];
          const layA = (rowA._layout||[])[e.ai], layB = (rowB._layout||[])[e.bi];
          if (!layA || !layB) continue;
          const centerA = layA.start + layA.w/2 + offsets[k];
          const centerB = layB.start + layB.w/2 + offsets[k+1];
          offsets[k+1] += (centerA - centerB);
        }
      })();

      function rowScale(idx){ const base = rowsG[idx]._baseX; const dom=base.domain().slice(); return d3.scaleLinear().domain(flips[idx]?dom.reverse():dom).range([0, baseInnerW]); }

      const lociIndexMaps = data.loci.map(loc => { const m=new Map(); loc.genes.forEach((g,i)=>m.set(i,g)); return {map:m}; });
      const edgesG = panG.append('g').attr('class','edges');
      // Render edges beneath gene glyphs
      try { edgesG.lower(); } catch (e) {}

      // Background overlay for pan/zoom capture
      const overlay = root.append('rect')
        .attr('x',0).attr('y',0).attr('width', baseInnerW).attr('height', contentH)
        .style('fill','transparent')
        .style('pointer-events','none')
        .style('cursor','grab');

      const zoomBehavior = d3.zoom().scaleExtent([minZoom, maxZoom]).on('zoom', (event) => {
        view = event.transform;
        panG.attr('transform', view.toString());
        zoomLbl.text(` (${(view.k).toFixed(2)}x)`);
      });
      svg.call(zoomBehavior).on('dblclick.zoom', null);
      svg.call(zoomBehavior.transform, view);

      function geneCenterX(rowIdx, geneIdx){
        const row = rowsG[rowIdx];
        const lay = (row._layout||[])[geneIdx] || {start:0,w:0};
        const rowW = row._rowWidth || 0;
        const local = flips[rowIdx] ? (rowW - (lay.start + lay.w/2)) : (lay.start + lay.w/2);
        return local + offsets[rowIdx];
      }

      function edgeColor(d) {
        const hasCos=(d.cos!=null&&isFinite(d.cos));
        const hasPf=(+d.pf||0)>0;
        if (hasCos && hasPf) return '#7b3294';
        if (hasCos) return '#2c7bb6';
        if (hasPf) return '#fdae61';
        return '#999';
      }
      function edgeWidth(d) {
        if (!scaleWidth) return 1.2;
        const hasCos = (d.cos != null && isFinite(d.cos));
        if (hasCos) {
          // Scale from 0.8 at tau to 3.5 at 1.0
          const t = Math.max(0, Math.min(1, (+d.cos - 0.5) / 0.5));
          return 0.8 + t * 2.7;
        }
        return 1.0;
      }
      function edgeOpacity(d) {
        const hasCos = (d.cos != null && isFinite(d.cos));
        if (hasCos) {
          // Scale opacity: 0.35 at threshold, 0.9 at 1.0
          const t = Math.max(0, Math.min(1, (+d.cos - 0.5) / 0.5));
          return 0.35 + t * 0.55;
        }
        return 0.5;
      }

      function drawEdges(){
        const allEdges = data.edges || [];
        const E = allEdges.filter(ed => {
          const hasCos = (ed.cos != null && isFinite(ed.cos));
          if (hasCos) return (+ed.cos) >= tau;
          // keep PFAM-only edges regardless of tau if they have overlap
          if ((+ed.pf || 0) > 0) return true;
          // always show fallback/anchor edges (no cosine, no PFAM)
          return !hasCos;
        });

        // --- Edge lines ---
        const esel = edgesG.selectAll('line.edge').data(E, d=> `${d.a}-${d.b}-${d.ai}-${d.bi}`);
        esel.exit().remove();
        const enter = esel.enter().append('line').attr('class','edge')
          .on('mouseover', function(event, ed){
            d3.select(this).attr('stroke-width', edgeWidth(ed) + 1.5);
            const cos=(ed.cos!=null&&isFinite(ed.cos))?(+ed.cos).toFixed(3):'n/a';
            const pf=(ed.pf!=null)?ed.pf:'n/a';
            tooltip.style('display','block').html(
              '<b>Match: '+(ed.type||'anchor')+'</b><br>'+
              'Cosine: '+cos+'<br>'+
              'PFAM overlap: '+pf
            );
          })
          .on('mousemove', function(event){
            const cw=container.clientWidth||0,ch=container.clientHeight||0;
            const tw=tooltip.node().offsetWidth||0, th=tooltip.node().offsetHeight||0;
            const pt=d3.pointer(event,container);
            let left=pt[0]+12, top=pt[1]+12;
            if (left+tw>cw-4) left=Math.max(4, pt[0]-12-tw);
            if (top+th>ch-4) top=Math.max(4, pt[1]-12-th);
            tooltip.style('left', left+'px').style('top', top+'px');
          })
          .on('mouseout', function(event, ed){
            d3.select(this).attr('stroke-width', edgeWidth(ed));
            tooltip.style('display','none');
          });
        enter.merge(esel)
          .attr('stroke', edgeColor)
          .attr('stroke-width', edgeWidth)
          .attr('stroke-opacity', edgeOpacity)
          .attr('x1', d => geneCenterX(d.a, d.ai))
          .attr('y1', d => d.a*(rowH+rowGap) + rowH/2)
          .attr('x2', d => geneCenterX(d.b, d.bi))
          .attr('y2', d => d.b*(rowH+rowGap) + rowH/2);

        // --- Similarity labels on edges ---
        // Only show labels for edges that have a cosine value
        const labelData = showLabels ? E.filter(d => d.cos != null && isFinite(d.cos)) : [];
        const lsel = edgesG.selectAll('text.edge-label').data(labelData, d=> `lbl-${d.a}-${d.b}-${d.ai}-${d.bi}`);
        lsel.exit().remove();
        const lEnter = lsel.enter().append('text').attr('class','edge-label')
          .attr('font-size','9px')
          .attr('font-family','monospace')
          .attr('text-anchor','middle')
          .attr('pointer-events','none')
          // White outline behind text for readability over edges/genes
          .attr('stroke','#fff')
          .attr('stroke-width','3px')
          .attr('paint-order','stroke');
        lEnter.merge(lsel)
          .attr('x', d => (geneCenterX(d.a, d.ai) + geneCenterX(d.b, d.bi)) / 2)
          .attr('y', d => {
            const y1 = d.a*(rowH+rowGap) + rowH/2;
            const y2 = d.b*(rowH+rowGap) + rowH/2;
            // Small vertical jitter based on gene index to reduce overlap
            const jitter = ((d.ai * 7 + d.bi * 3) % 5 - 2) * 3;
            return (y1 + y2) / 2 - 2 + jitter;
          })
          .attr('fill', edgeColor)
          .attr('font-weight', 'bold')
          .attr('opacity', 0.9)
          .text(d => (+d.cos).toFixed(2));
      }

      function redraw(){
        // Keep fixed SVG size; pan/zoom handled by zoomBehavior on panG
        zoomLbl.text(` (${(view.k).toFixed(2)}x)`);
        data.loci.forEach((loc, idx) => {
          const rowNode = rowsG[idx]; const row = d3.select(rowNode); const yMid=rowH/2;
          row.attr('transform', `translate(${offsets[idx]},${idx*(rowH+rowGap)})`);
          const genesSel = row.selectAll('path.gene').data(loc.genes, d=>d.id);
          genesSel.enter().append('path').attr('class','gene'); genesSel.exit().remove();
          row.selectAll('path.gene')
            .attr('d', (d,i) => { const lay=(rowNode._layout||[])[i]||{start:0,w:0}; const rowW=rowNode._rowWidth||0; const w=lay.w; const x0 = flips[idx] ? (rowW - (lay.start + lay.w)) : lay.start; const dir=d.strand>=0?1:-1; const head=6; const x1=x0, x2=x0+w; return dir>=0? `M ${x1},${yMid-8} L ${x2-head},${yMid-8} L ${x2},${yMid} L ${x2-head},${yMid+8} L ${x1},${yMid+8} Z` : `M ${x2},${yMid-8} L ${x1+head},${yMid-8} L ${x1},${yMid} L ${x1+head},${yMid+8} L ${x2},${yMid+8} Z`; })
            .attr('fill', d => { const pf=(d.pfam||'').split(';')[0]||''; return pf ? colorPF(pf) : (d.strand>=0 ? '#2ecc71' : '#e74c3c'); })
            .attr('stroke','rgba(255,255,255,0.6)')
            .attr('stroke-width',1.0)
            .style('cursor','pointer')
            .on('click', (event,d)=> centerOn(idx,d))
            .on('dblclick', ()=> { flips[idx] = !flips[idx]; redraw(); })
            .on('mouseover', function(event, d){ const strand=(d.strand>=0? '+':'-'); const len=Math.max(0, Math.round(Math.abs(d.end-d.start))); tooltip.style('display','block').html('<b>'+d.id+'</b><br>Length: '+len.toLocaleString()+' aa<br>Strand: '+strand+'<br>PFAM: '+(d.pfam||'None')); })
            .on('mousemove', function(event){ const cw=container.clientWidth||0,ch=container.clientHeight||0; const tw=tooltip.node().offsetWidth||0, th=tooltip.node().offsetHeight||0; const pt=d3.pointer(event,container); let left=pt[0]+12, top=pt[1]+12; if (left+tw>cw-4) left=Math.max(4, pt[0]-12-tw); if (top+th>ch-4) top=Math.max(4, pt[1]-12-th); tooltip.style('left', left+'px').style('top', top+'px'); })
            .on('mouseout', ()=> tooltip.style('display','none'));
          let base=row.selectAll('line.base').data([0]); base=base.enter().append('line').attr('class','base').merge(base); const rowW=rowNode._rowWidth||0; base.attr('x1',0).attr('x2',rowW).attr('y1',yMid).attr('y2',yMid).attr('stroke','#ccc').attr('stroke-width',0.5);
        });
        drawEdges();
      }

      function cosine(a,b){ if (!a||!b||!a.emb||!b.emb) return null; let s=0,na=0,nb=0; for (let i=0;i<a.emb.length && i<b.emb.length;i++){ s+=a.emb[i]*b.emb[i]; na+=a.emb[i]*a.emb[i]; nb+=b.emb[i]*b.emb[i]; } if (na<=0||nb<=0) return null; return s/Math.sqrt(na*nb); }
      function pfOverlap(a,b){ const sA=new Set((a.pfam||'').split(';').map(x=>x.trim()).filter(Boolean)); const sB=new Set((b.pfam||'').split(';').map(x=>x.trim()).filter(Boolean)); let c=0; sA.forEach(x=>{ if(sB.has(x)) c++; }); return c; }
      function centerOn(rowIndex, gene){
        const targetX = baseInnerW/2;
        // find index of clicked gene in its row
        const seedIdx = (data.loci[rowIndex].genes||[]).findIndex(g => g.id === gene.id);
        if (seedIdx < 0) { return; }
        const matches = {}; // row -> gene index
        matches[rowIndex] = seedIdx;
        // helper to select best edge match between adjacent rows, preferring highest cosine
        function bestEdge(fromRow, fromIdx, toRow){
          let bestCos = null, bestCosScore = -Infinity;
          let bestPf = null, bestPfScore = -Infinity;
          (data.edges||[]).forEach(ed => {
            if ((ed.a === fromRow && ed.b === toRow && ed.ai === fromIdx) ||
                (ed.b === fromRow && ed.a === toRow && ed.bi === fromIdx)){
              const toIdx = (ed.a === fromRow) ? ed.bi : ed.ai;
              const cos = (ed.cos != null && isFinite(ed.cos)) ? +ed.cos : -Infinity;
              const pf  = (+ed.pf || 0);
              if (isFinite(cos) && cos > bestCosScore){ bestCosScore = cos; bestCos = {row: toRow, idx: toIdx, score: cos}; }
              if (!isFinite(cos) && pf > bestPfScore){ bestPfScore = pf; bestPf = {row: toRow, idx: toIdx, score: pf}; }
            }
          });
          if (bestCos && bestCos.score >= tau) return bestCos; // prefer cosine meeting tau
          if (bestPf && bestPf.score > 0) return bestPf; // fallback on PFAM when cosine missing or below tau
          return null;
        }
        // propagate downwards (towards row 0)
        for (let r = rowIndex; r > 0; r--){
          const prev = bestEdge(r, matches[r], r-1);
          if (!prev) break;
          matches[r-1] = prev.idx;
        }
        // propagate upwards (towards last row)
        for (let r = rowIndex; r < data.loci.length-1; r++){
          const nxt = bestEdge(r, matches[r], r+1);
          if (!nxt) break;
          matches[r+1] = nxt.idx;
        }
        // apply absolute offsets for matched rows to center the chosen genes
        const halfW = baseInnerW/2;
        function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
        Object.keys(matches).forEach(k => {
          const ri = +k; const gi = matches[ri];
          const row = rowsG[ri]; const lay = (row._layout||[])[gi] || {start:0,w:0}; const rowW=row._rowWidth||0;
          const centerLocal = flips[ri] ? (rowW - (lay.start + lay.w/2)) : (lay.start + lay.w/2);
          // Solve: (centerLocal + offset)*view.k + view.x = targetX
          const desired = (targetX - view.x)/view.k - centerLocal;
          offsets[ri] = clamp(desired, -halfW*2, halfW*2);
        });
        redraw();
      }

      // Enable per-row drag to manually align rows
      function attachRowDrag(row, idx){
        row.style('cursor','grab')
          .call(d3.drag()
            .on('start', function(){ d3.select(this).style('cursor','grabbing'); })
            .on('drag', function(event){ offsets[idx] += event.dx; redraw(); })
            .on('end', function(){ d3.select(this).style('cursor','grab'); })
          );
      }

      redraw();

      // Attach drag after first draw
      rowsG.forEach((row, idx) => attachRowDrag(d3.select(row), idx));
    </script>
    """
    # Escape closing-tag sequences before the payload is inlined: gene and PFAM
    # identifiers come from FASTA headers and annotation files, so a value
    # containing "</script>" would otherwise terminate the script block early.
    safe_payload = _json.dumps(data).replace('</', '<\\/')
    html = (html
        .replace('__DATA__', safe_payload)
        .replace('__WIDTH__', str(width if width else 0))
        .replace('__HEIGHT__', str(height))
        .replace('__INITIAL_TAU__', str(float(initial_tau)))
        .replace('__SHOW_LABELS__', 'true' if show_labels else 'false')
        .replace('__SCALE_EDGE_WIDTH__', 'true' if scale_edge_width else 'false')
    )
    return html


def build_consensus_html(
    tokens: List[Dict],
    pairs: Optional[List[Dict]] = None,
    *,
    height: int = 190,
    title: str = "",
    interactive: bool = True,
) -> str:
    """Draw a consensus architecture as a row of gene arrows.

    ``tokens`` come from :func:`consensus_from_data` in locus order. Each becomes
    an arrow pointing in the direction the gene usually runs, coloured by the
    same domain hash the alignment view uses so a slot is recognisable between
    the two. Arrow opacity carries coverage — how many of the cluster's loci hold
    that slot — because a consensus drawn at uniform weight implies every
    position is equally attested when they are not.

    ``pairs`` supply adjacency confidence between neighbouring slots and are
    drawn as the connector between arrows.
    """
    if not tokens:
        return ""

    pairs = pairs or []
    adjacency = {int(p.get("i", -1)): float(p.get("same_frac", 0.0)) for p in pairs}

    pad, gap, arrow_h = 18, 12, 42
    body_w = 128
    width = pad * 2 + len(tokens) * body_w + max(0, len(tokens) - 1) * gap
    top = 54 if title else 30

    parts: List[str] = [
        f'<svg id="consensus-svg" xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" data-w="{width}" data-h="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:#1a1a2e;font-family:'
        f'ui-sans-serif,-apple-system,sans-serif;display:block">'
    ]
    if title:
        parts.append(
            f'<text x="{pad}" y="26" fill="#e8e9ec" font-size="14" '
            f'font-weight="600">{_escape(title)}</text>'
        )

    for index, token in enumerate(tokens):
        x = pad + index * (body_w + gap)
        coverage = max(0.0, min(1.0, float(token.get("coverage", 0.0))))
        forward = float(token.get("fwd_frac", 1.0)) >= 0.5
        colour = token.get("color") or _pfam_color(str(token.get("token", "")))
        # Coverage drives opacity: a slot present in a third of loci must not
        # read the same as one present in all of them.
        opacity = 0.30 + 0.70 * coverage
        head = 22
        y, h = top, arrow_h
        if forward:
            points = (f"{x},{y} {x + body_w - head},{y} {x + body_w},{y + h / 2} "
                      f"{x + body_w - head},{y + h} {x},{y + h}")
        else:
            points = (f"{x + body_w},{y} {x + head},{y} {x},{y + h / 2} "
                      f"{x + head},{y + h} {x + body_w},{y + h}")
        parts.append(
            f'<polygon points="{points}" fill="{colour}" fill-opacity="{opacity:.2f}" '
            f'stroke="#0f1016" stroke-width="1"><title>{_escape(str(token.get("token","")))}'
            f' — in {coverage * 100:.0f}% of loci, {token.get("n_occ", 0)} occurrences'
            f'</title></polygon>'
        )
        label = str(token.get("token", ""))
        if len(label) > 16:
            label = label[:15] + "…"
        parts.append(
            f'<text x="{x + body_w / 2}" y="{y + h / 2 + 4}" fill="#f2f3f5" '
            f'font-size="11" text-anchor="middle" pointer-events="none">'
            f'{_escape(label)}</text>'
        )
        parts.append(
            f'<text x="{x + body_w / 2}" y="{y + h + 16}" fill="#9aa0a8" '
            f'font-size="10" text-anchor="middle">{coverage * 100:.0f}%</text>'
        )
        if index in adjacency and index + 1 < len(tokens):
            confidence = adjacency[index]
            cx = x + body_w
            parts.append(
                f'<line x1="{cx}" y1="{y + h / 2}" x2="{cx + gap}" y2="{y + h / 2}" '
                f'stroke="#7fb8a4" stroke-width="{0.8 + 2.4 * confidence:.1f}" '
                f'stroke-opacity="{0.25 + 0.65 * confidence:.2f}">'
                f'<title>adjacent in {confidence * 100:.0f}% of loci</title></line>'
            )

    parts.append("</svg>")
    svg = "".join(parts)
    if not interactive:
        return svg

    # A consensus of any length is unreadable scaled to fit — 29 slots squeezes
    # the labels to nothing. Draw at natural size inside a scrolling pane and
    # let the reader zoom, rather than choosing a scale for them.
    return f"""
<div style="background:#1a1a2e;font-family:ui-sans-serif,-apple-system,sans-serif">
  <div style="padding:6px 10px;color:#ddd;font-size:13px;display:flex;
              gap:6px;align-items:center;flex-wrap:wrap">
    <button data-z="out">&minus;</button>
    <button data-z="in">+</button>
    <button data-z="fit">Fit</button>
    <button data-z="reset">1:1</button>
    <span id="zlbl" style="color:#8a8f98"></span>
    <span style="color:#8a8f98;margin-left:auto">scroll to pan &middot; \u2318/ctrl+scroll to zoom</span>
  </div>
  <div id="pane" style="overflow:auto;max-width:100%;cursor:grab">{svg}</div>
</div>
<style>
  html,body {{ margin:0; background:#1a1a2e; }}
  #pane button, div button {{ background:#2a2a3e; color:#ddd; border:1px solid #444;
    border-radius:4px; padding:2px 9px; cursor:pointer; font:inherit; }}
  div button:hover {{ border-color:#7fb8a4; }}
</style>
<script>
  (function() {{
    const svg = document.getElementById('consensus-svg');
    const pane = document.getElementById('pane');
    const lbl = document.getElementById('zlbl');
    const W = +svg.dataset.w, H = +svg.dataset.h;
    let z = 1;
    function apply() {{
      svg.setAttribute('width', Math.round(W * z));
      svg.setAttribute('height', Math.round(H * z));
      lbl.textContent = Math.round(z * 100) + '%';
    }}
    function fit() {{ z = Math.min(4, Math.max(0.1, pane.clientWidth / W)); apply(); }}
    document.querySelectorAll('button[data-z]').forEach(b => b.onclick = () => {{
      const k = b.dataset.z;
      if (k === 'in')  z = Math.min(6, z * 1.25);
      if (k === 'out') z = Math.max(0.1, z / 1.25);
      if (k === 'reset') z = 1;
      if (k === 'fit') return fit();
      apply();
    }});
    pane.addEventListener('wheel', e => {{
      if (!e.ctrlKey && !e.metaKey) return;      // plain scroll still pans
      e.preventDefault();
      z = Math.min(6, Math.max(0.1, z * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
      apply();
    }}, {{passive: false}});
    let drag = null;
    pane.addEventListener('mousedown', e => {{
      drag = {{x: e.clientX, y: e.clientY, l: pane.scrollLeft, t: pane.scrollTop}};
      pane.style.cursor = 'grabbing'; e.preventDefault();
    }});
    window.addEventListener('mouseup', () => {{ drag = null; pane.style.cursor = 'grab'; }});
    window.addEventListener('mousemove', e => {{
      if (!drag) return;
      pane.scrollLeft = drag.l - (e.clientX - drag.x);
      pane.scrollTop  = drag.t - (e.clientY - drag.y);
    }});
    apply();
    if (W > pane.clientWidth) fit();            // start readable, not clipped
  }})();
</script>
"""


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))
