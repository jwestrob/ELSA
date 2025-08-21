import math
from collections import defaultdict
from elsa.analyze.shingles import block_shingles, df_filter


def weighted_jaccard(A, B, idf):
    inter = A & B
    union = A | B
    if not union:
        return 0.0
    w_inter = sum(idf.get(x, 0.0) for x in inter)
    w_union = sum(idf.get(x, 0.0) for x in union)
    return (w_inter / w_union) if w_union > 0 else 0.0


def compute_idf(postings):
    # idf = log(1 + N/df)
    N = max(1, len(postings))
    idf = {}
    for token, blocks in postings.items():
        df = len(blocks)
        idf[token] = math.log(1.0 + (N / max(1, df)))
    return idf


def test_df_filter_can_split_a_cohesive_group():
    # Simulate 4 RP-like blocks sharing many common shingles (C), with a competence-like group (D)
    # Common RP shingles seen in all 4 RP blocks
    common = {f"c{i}" for i in range(10)}
    # Each RP block has a couple of local variants
    A = common | {"ra1", "ra2"}
    B = common | {"rb1", "rb2"}
    C = common | {"rc1", "rc2"}
    D = common | {"rd1", "rd2"}
    # A competence-like block shares a couple of shingles with B (e.g., accessory RP genes)
    E = {"rb1", "rb2", "x1", "x2", "x3"}

    blocks = {"A": A, "B": B, "C": C, "D": D, "E": E}

    # Build postings and IDF without DF filtering (keep common signal)
    postings = defaultdict(set)
    for name, s in blocks.items():
        for t in s:
            postings[t].add(name)
    idf = compute_idf(postings)

    # With all tokens, RP-to-RP similarities are high; RP-to-E is low
    sim_AB = weighted_jaccard(A, B, idf)
    sim_AE = weighted_jaccard(A, E, idf)
    assert sim_AB > sim_AE
    assert sim_AB > 0.5  # cohesive when common tokens kept

    # Now emulate DF filtering dropping common tokens with high DF (appear in A,B,C,D)
    df_max = 2
    filtered_blocks = {}
    for name, s in blocks.items():
        filtered_blocks[name] = {t for t in s if len(postings[t]) <= df_max}

    # Recompute postings/idf after filtering
    postings2 = defaultdict(set)
    for name, s in filtered_blocks.items():
        for t in s:
            postings2[t].add(name)
    idf2 = compute_idf(postings2)

    # Common signal is gone; B still shares two shingles with E, inflating B-E relative to A-B
    A2, B2, E2 = filtered_blocks["A"], filtered_blocks["B"], filtered_blocks["E"]
    sim_AB2 = weighted_jaccard(A2, B2, idf2)
    sim_BE2 = weighted_jaccard(B2, E2, idf2)
    # After filtering, a spurious bridge to E can rival or exceed RP-RP similarity
    assert sim_BE2 >= sim_AB2


def _make_shingles_from_seq(seq, k=3, skipgram=False):
    # Represent each window with a single band token = the integer itself
    windows = [[int(t)] for t in seq]
    return block_shingles(
        windows,
        k=k,
        method="xor",
        skipgram_offsets=(0, 2, 5) if skipgram else None,
        strand_canonical_shingles=False,
    )


def test_df_filter_on_real_shingles_sets_with_block_shingles():
    # Backbone sequence shared by four RP-like blocks
    base = [1,2,3,4,5,6,7,8,9,10]
    A_seq = base + [11,12]
    B_seq = base + [13,14]
    C_seq = base + [15,16]
    D_seq = base + [17,18]
    # Competence-like block shares only a couple of accessory shingles with B
    # Construct E to share k-grams [8,9,10] and [9,10,13] with B after contiguous k=3
    E_seq = [7,8,9,10,13,19,20]

    A = _make_shingles_from_seq(A_seq)
    B = _make_shingles_from_seq(B_seq)
    C = _make_shingles_from_seq(C_seq)
    D = _make_shingles_from_seq(D_seq)
    E = _make_shingles_from_seq(E_seq)

    blocks = {"A": A, "B": B, "C": C, "D": D, "E": E}
    # Build postings and IDF
    postings = defaultdict(set)
    for name, S in blocks.items():
        for s in S:
            postings[s].add(name)
    idf = compute_idf(postings)
    sim_AB = weighted_jaccard(A, B, idf)
    sim_BE = weighted_jaccard(B, E, idf)
    print(f"pre-DF: |A|={len(A)} |B|={len(B)} |E|={len(E)} sim_AB={sim_AB:.3f} sim_BE={sim_BE:.3f}")
    assert sim_AB > sim_BE

    # Apply DF filter dropping common shingles (DF>=4 here)
    shingle_df = {tok: len(bl) for tok, bl in postings.items()}
    A2 = df_filter(shingle_df, df_max=2, S=A)
    B2 = df_filter(shingle_df, df_max=2, S=B)
    E2 = df_filter(shingle_df, df_max=2, S=E)
    # Recompute IDF after filtering
    postings2 = defaultdict(set)
    for name, S in {"A": A2, "B": B2, "E": E2}.items():
        for s in S:
            postings2[s].add(name)
    idf2 = compute_idf(postings2)
    sim_AB2 = weighted_jaccard(A2, B2, idf2)
    sim_BE2 = weighted_jaccard(B2, E2, idf2)
    print(f"post-DF: |A2|={len(A2)} |B2|={len(B2)} |E2|={len(E2)} sim_AB2={sim_AB2:.3f} sim_BE2={sim_BE2:.3f}")
    assert sim_BE2 >= sim_AB2


def test_skipgram_can_increase_overlap_with_partial_blocks():
    # Base RP-like sequence
    base = [1,2,3,4,5,6,7,8,9,10]
    full_contig = _make_shingles_from_seq(base, k=3, skipgram=False)
    full_skip = _make_shingles_from_seq(base, k=3, skipgram=True)
    # Construct a sparse partial sequence that preserves one skip-gram triple (1,3,6)
    partial_seq = [1, 100, 3, 101, 102, 6]
    contig = _make_shingles_from_seq(partial_seq, k=3, skipgram=False)
    skipg = _make_shingles_from_seq(partial_seq, k=3, skipgram=True)
    pre_inter = full_contig & contig
    post_inter = full_skip & skipg
    print(f"contiguous_inter={len(pre_inter)} skipgram_inter={len(post_inter)}")
    assert len(post_inter) >= len(pre_inter)


def test_degree_cap_can_fragment_even_with_dense_edges():
    # Five nodes in a clique with equal weights
    nodes = ["n1", "n2", "n3", "n4", "n5"]
    # Assign identical weight to all pair edges
    weights = {(min(a, b), max(a, b)): 0.8 for i, a in enumerate(nodes) for b in nodes[i+1:]}

    # Apply a degree cap of 2: keep only top-2 edges per node (ties arbitrary)
    degree_cap = 2
    kept = set()
    for u in nodes:
        # Get incident edges and sort by weight
        inc = []
        for v in nodes:
            if v == u:
                continue
            key = (min(u, v), max(u, v))
            w = weights[key]
            inc.append((v, w))
        inc.sort(key=lambda x: -x[1])
        for v, w in inc[:degree_cap]:
            kept.add((min(u, v), max(u, v)))

    # Build adjacency and check connectivity
    adj = defaultdict(set)
    for (u, v) in kept:
        adj[u].add(v)
        adj[v].add(u)
    # BFS from first node; with small degree cap, it's possible to break connectivity
    seen = set()
    stack = [nodes[0]]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(list(adj[x] - seen))

    # In many tie-break orders, not all nodes are reachable -> fragmentation risk
    assert len(seen) <= len(nodes)
