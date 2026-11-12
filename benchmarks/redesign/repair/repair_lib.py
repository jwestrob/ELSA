"""Shared helpers for workstream repair."""


def resolve_moves(nxt, base_tok, size_of_token):
    """Final token of every moved gene.

    nxt: {moved gene: gene it follows (representative gene of its best hit)}. A gene that is not a key of nxt keeps
    its base token. A moved gene takes the final token of the gene it follows; a cycle of moved genes takes the base
    token of its member with the largest base token (ties: lower token id). Returns (final, n_chains, n_cycles).
    """
    final, n_chain, n_cycle = {}, 0, 0
    for g0 in nxt:
        if g0 in final:
            continue
        path, seen, g = [], {}, g0
        while g in nxt and g not in final and g not in seen:
            seen[g] = len(path)
            path.append(g)
            g = nxt[g]
        if g in final:
            f = final[g]
        elif g in seen:
            cyc = path[seen[g]:]
            n_cycle += 1
            f = sorted((-size_of_token[int(base_tok[c])], int(base_tok[c])) for c in cyc)[0][1]
            for c in cyc:
                final[c] = f
            path = path[:seen[g]]
        else:
            f = int(base_tok[g])
        for c in path:
            final[c] = f
        n_chain += int(len(path) > 1)
    return final, n_chain, n_cycle


if __name__ == "__main__":
    import numpy as np
    # genes 0..9; tokens: 0:{0,1} 1:{2,3} 2:{4} 3:{5,6,7} 4:{8} 5:{9}
    base = np.array([0, 0, 1, 1, 2, 3, 3, 3, 4, 5])
    size = {0: 2, 1: 2, 2: 1, 3: 3, 4: 1, 5: 1}
    # 4 -> 2 (token 1); 2 -> 5 (token 3, stable): chain, 4 ends in token 3
    # 0 -> 3 (token 1); 3 -> 1 (token 0): cycle-free? 3 -> 1, 1 not moved -> 3 ends in token 0; 0 follows 3 -> token 0 (own token)
    # 8 -> 9, 9 -> 8: cycle of singletons -> token min(4,5) by (-1, id) -> 4
    nxt = {4: 2, 2: 5, 0: 3, 3: 1, 8: 9, 9: 8}
    final, nc, ny = resolve_moves(nxt, base, size)
    assert final == {4: 3, 2: 3, 0: 0, 3: 0, 8: 4, 9: 4}, final
    assert ny == 1 and nc == 2, (nc, ny)
    # cycle with a tail: 1 -> 2, 2 -> 3? 3 is same token as 2 here; make 2 -> 0, 0 -> 2 (cycle tokens 1 and 0, sizes 2, 2 -> token 0)
    final, nc, ny = resolve_moves({1: 2, 2: 0, 0: 2}, base, size)
    assert final == {0: 0, 2: 0, 1: 0} and ny == 1, final
    print("resolve_moves ok", final)
