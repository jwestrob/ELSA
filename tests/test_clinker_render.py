"""The clinker renderers build HTML from a payload and nothing else.

These diagrams previously lived inside the Streamlit app and could only run
under a live session, so they were never covered. Extracted, they are ordinary
functions: payload in, document out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from elsa.viz import clinker

_MODULE = Path(clinker.__file__)


def _payload():
    return {
        "loci": [
            {"genome": "g1", "contig": "c1", "genes": [
                {"gene_id": "g1_0", "start": 0, "end": 900, "strand": 1,
                 "pfam": "Ribosomal_L2"},
                {"gene_id": "g1_1", "start": 1000, "end": 1800, "strand": 1,
                 "pfam": "Ribosomal_S19"},
            ]},
            {"genome": "g2", "contig": "c1", "genes": [
                {"gene_id": "g2_0", "start": 0, "end": 880, "strand": 1,
                 "pfam": "Ribosomal_L2"},
                {"gene_id": "g2_1", "start": 990, "end": 1790, "strand": -1,
                 "pfam": "Ribosomal_S19"},
            ]},
        ],
        "edges": [
            {"source": "g1_0", "target": "g2_0", "similarity": 0.98},
            {"source": "g1_1", "target": "g2_1", "similarity": 0.95},
        ],
    }


def test_module_is_free_of_ui_framework_imports():
    """The point of the split: these diagrams must not require Streamlit."""
    source = _MODULE.read_text()
    assert "streamlit" not in source
    assert "components.html" not in source


def test_multiloc_builds_a_self_contained_document():
    html = clinker.build_multiloc_html(_payload(), height=400)
    assert isinstance(html, str) and len(html) > 1000
    assert "cluster-multiloc" in html
    assert "d3" in html


def test_all_placeholders_are_substituted():
    """An unsubstituted token renders as a broken diagram rather than an error."""
    html = clinker.build_multiloc_html(_payload(), height=333, initial_tau=0.7,
                                       show_labels=False, scale_edge_width=False)
    for token in ("__DATA__", "__WIDTH__", "__HEIGHT__", "__INITIAL_TAU__",
                  "__SHOW_LABELS__", "__SCALE_EDGE_WIDTH__"):
        assert token not in html, f"{token} left unsubstituted"
    assert "333" in html
    assert "0.7" in html


def test_payload_values_reach_the_document():
    html = clinker.build_multiloc_html(_payload(), height=400)
    assert "Ribosomal_L2" in html
    assert "g1_0" in html and "g2_1" in html


def test_empty_payload_returns_empty_string_not_none():
    """Callers embed the result directly, so the empty case must stay a string."""
    assert clinker.build_multiloc_html({}) == ""
    assert clinker.build_multiloc_html({"loci": []}) == ""


def test_closing_script_tags_in_payload_cannot_break_out():
    """Gene identifiers are data; they must not be able to terminate the script."""
    payload = _payload()
    payload["loci"][0]["genes"][0]["gene_id"] = "</script><script>alert(1)</script>"
    html = clinker.build_multiloc_html(payload, height=400)
    assert "<script>alert(1)</script>" not in html


def test_comparative_v2_builds_and_substitutes():
    data = {
        "query_locus": [{"gene_id": "q0", "start": 0, "end": 500, "strand": 1}],
        "target_locus": [{"gene_id": "t0", "start": 0, "end": 500, "strand": 1}],
        "links": [{"source": "q0", "target": "t0", "identity": 0.9}],
    }
    html = clinker.build_comparative_html_v2(data, height=300)
    assert isinstance(html, str)
    if html:
        assert "__PAYLOAD_JSON__" not in html
        assert "__HEIGHT__" not in html


def test_consensus_runs_and_labels_by_domain():
    """Regression: extraction left ``defaultdict``, ``Counter``, ``Set`` and the
    ``_pfam_color`` helper behind in app.py, so this path raised NameError at
    runtime while the module imported fine."""
    payload = {
        "loci": [
            {"genome": "g1", "contig": "c1", "genes": [
                {"id": "g1_0", "start": 0, "end": 9, "strand": 1, "pfam": "Ribosomal_L2"},
                {"id": "g1_1", "start": 20, "end": 29, "strand": 1, "pfam": "Ribosomal_S19"},
            ]},
            {"genome": "g2", "contig": "c1", "genes": [
                {"id": "g2_0", "start": 0, "end": 9, "strand": 1, "pfam": "Ribosomal_L2"},
                {"id": "g2_1", "start": 20, "end": 29, "strand": 1, "pfam": "Ribosomal_S19"},
            ]},
        ],
        "edges": [
            {"a": 0, "ai": 0, "b": 1, "bi": 0, "cos": 0.99},
            {"a": 0, "ai": 1, "b": 1, "bi": 1, "cos": 0.98},
        ],
    }
    tokens, pairs = clinker.consensus_from_data(payload, 0.5)
    labels = {t["token"] for t in tokens}
    assert "Ribosomal_L2" in labels and "Ribosomal_S19" in labels
    for token in tokens:
        assert 0.0 <= token["coverage"] <= 1.0
        assert token["color"].startswith("rgba(")


def test_consensus_coverage_threshold_excludes_rare_slots():
    payload = {
        "loci": [
            {"genome": "g1", "contig": "c1", "genes": [
                {"id": "a", "start": 0, "end": 9, "strand": 1, "pfam": "Common"}]},
            {"genome": "g2", "contig": "c1", "genes": [
                {"id": "b", "start": 0, "end": 9, "strand": 1, "pfam": "Common"}]},
            {"genome": "g3", "contig": "c1", "genes": [
                {"id": "c", "start": 0, "end": 9, "strand": 1, "pfam": "Rare"}]},
        ],
        "edges": [{"a": 0, "ai": 0, "b": 1, "bi": 0, "cos": 0.99}],
    }
    permissive = clinker.consensus_from_data(payload, 0.1)[0]
    strict = clinker.consensus_from_data(payload, 0.9)[0]
    assert len(strict) < len(permissive)


def test_every_public_function_executes():
    """Guards the extraction as a whole rather than one code path.

    Moving these functions out of app.py left ``defaultdict``, ``Counter``,
    ``Set`` and the ``_pfam_color`` helper behind. The module still imported
    cleanly, so nothing failed until a request reached the consensus path in
    production. Calling each entry point here turns that into a test failure.
    """
    payload = {
        "loci": [
            {"genome": "g1", "contig": "c1", "genes": [
                {"id": "a", "start": 0, "end": 9, "strand": 1, "pfam": "X"}]},
            {"genome": "g2", "contig": "c1", "genes": [
                {"id": "b", "start": 0, "end": 9, "strand": -1, "pfam": "X"}]},
        ],
        "edges": [{"a": 0, "ai": 0, "b": 1, "bi": 0, "cos": 0.99}],
    }
    comparative = {
        "query_locus": [{"gene_id": "q", "start": 0, "end": 9, "strand": 1}],
        "target_locus": [{"gene_id": "t", "start": 0, "end": 9, "strand": 1}],
        "links": [{"source": "q", "target": "t", "identity": 0.9}],
    }

    assert isinstance(clinker.build_multiloc_html(payload, height=300), str)
    assert isinstance(clinker.build_comparative_html(comparative, height=300), str)
    assert isinstance(clinker.build_comparative_html_v2(comparative, height=300), str)
    tokens, pairs = clinker.consensus_from_data(payload, 0.5)
    assert isinstance(tokens, list) and isinstance(pairs, list)
    converted = clinker.comparative_to_multiloc(comparative, {"block_id": 1})
    assert isinstance(converted, dict)


def test_tau_slider_range_follows_the_data():
    """A slider spanning 0.50-0.99 filters nothing when every anchor sits above
    0.98, which is the norm in an anisotropic embedding space."""
    payload = {
        "loci": [
            {"genome": "g1", "contig": "c1", "genes": [
                {"id": "a", "start": 0, "end": 9, "strand": 1, "pfam": "X"}]},
            {"genome": "g2", "contig": "c1", "genes": [
                {"id": "b", "start": 0, "end": 9, "strand": 1, "pfam": "X"}]},
        ],
        "edges": [{"a": 0, "ai": 0, "b": 1, "bi": 0, "cos": 0.994}],
    }
    html = clinker.build_multiloc_html(payload, height=400)
    assert "cosVals" in html and "tauMin" in html
    assert "'0.50'" not in html, "hard-coded slider floor must be gone"


def _consensus_tokens():
    return [
        {"token": "Ribosomal_L2", "color": "rgba(120,80,200,0.85)",
         "coverage": 0.92, "fwd_frac": 1.0, "n_occ": 11},
        {"token": "Ribosomal_S19", "color": "rgba(80,180,140,0.85)",
         "coverage": 0.40, "fwd_frac": 0.1, "n_occ": 5},
    ]


def test_consensus_draws_one_arrow_per_slot():
    html = clinker.build_consensus_html(_consensus_tokens())
    assert html.count("<polygon") == 2
    assert "Ribosomal_L2" in html and "Ribosomal_S19" in html


def test_consensus_arrow_direction_follows_orientation():
    """A slot usually read on the reverse strand must point the other way."""
    forward, reverse = _consensus_tokens()
    fwd_only = clinker.build_consensus_html([forward])
    rev_only = clinker.build_consensus_html([reverse])
    assert fwd_only.count("<polygon") == rev_only.count("<polygon") == 1
    assert fwd_only != rev_only, "orientation must change the drawn shape"


def test_consensus_opacity_encodes_coverage():
    """Uniform weight would imply every slot is equally attested."""
    import re
    html = clinker.build_consensus_html(_consensus_tokens())
    opacities = [float(v) for v in re.findall(r'fill-opacity="([0-9.]+)"', html)]
    assert len(opacities) == 2
    assert opacities[0] > opacities[1], "0.92 coverage must read stronger than 0.40"


def test_consensus_labels_are_escaped():
    html = clinker.build_consensus_html(
        [{"token": "<script>x</script>", "coverage": 1.0, "fwd_frac": 1.0, "n_occ": 1}]
    )
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_consensus_adjacency_links_are_drawn_between_slots():
    html = clinker.build_consensus_html(
        _consensus_tokens(), [{"i": 0, "j": 1, "same_frac": 0.8, "support": 10}]
    )
    assert "<line" in html
    assert "adjacent in 80% of loci" in html


def test_empty_consensus_returns_empty_string():
    assert clinker.build_consensus_html([]) == ""


def test_consensus_is_drawn_at_natural_width_not_squashed_to_fit():
    """29 slots scaled to a viewport makes every label unreadable; the SVG keeps
    its own width and the pane scrolls instead."""
    tokens = [{"token": f"Dom_{i}", "coverage": 0.9, "fwd_frac": 1.0, "n_occ": 9}
              for i in range(29)]
    html = clinker.build_consensus_html(tokens)
    assert 'width="100%"' not in html
    assert "data-w=" in html and "data-h=" in html


def test_consensus_exposes_zoom_and_pan_controls():
    tokens = [{"token": "A", "coverage": 1.0, "fwd_frac": 1.0, "n_occ": 2}]
    html = clinker.build_consensus_html(tokens)
    for control in ('data-z="in"', 'data-z="out"', 'data-z="fit"', 'data-z="reset"'):
        assert control in html
    assert "wheel" in html and "grabbing" in html


def test_consensus_non_interactive_mode_is_bare_svg():
    """Composable form for embedding elsewhere: no chrome, no script."""
    tokens = [{"token": "A", "coverage": 1.0, "fwd_frac": 1.0, "n_occ": 2}]
    svg = clinker.build_consensus_html(tokens, interactive=False)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<script" not in svg and "data-z" not in svg
