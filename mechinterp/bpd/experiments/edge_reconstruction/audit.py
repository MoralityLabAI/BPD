"""Independent atlas/VF2 check and exact minimum sufficient subdeck statistics."""
from collections import Counter, defaultdict
from itertools import product
import json
import networkx as nx
from oracle import canonical, deck


def audit():
    atlas = nx.graph_atlas_g()
    rows = []
    for n in range(4, 7):
        graphs = [g for g in atlas if len(g) == n]
        groups = defaultdict(list)
        for g in graphs:
            raw = tuple(sorted(tuple(sorted(e)) for e in g.edges))
            d = deck(n, raw)
            ref = Counter()
            for e in g.edges:
                card = g.copy()
                card.remove_edge(*e)
                matches = [h for h in graphs if h.size() == card.size()
                           and nx.is_isomorphic(h, card)]
                assert len(matches) == 1
                h = matches[0]
                ref[canonical(n, tuple(sorted(tuple(sorted(f)) for f in h.edges)))] += 1
            assert d == ref
            groups[g.size()].append(d)
        histogram = Counter()
        ambiguous = total = 0
        for ds in groups.values():
            for d in ds:
                total += 1
                if sum(not (d - other) for other in ds) != 1:
                    ambiguous += 1
                    continue
                best = sum(d.values())
                keys = list(d)
                for counts in product(*(range(d[k]+1) for k in keys)):
                    size = sum(counts)
                    if size >= best:
                        continue
                    partial = Counter(dict(zip(keys, counts)))
                    if sum(not (partial - other) for other in ds) == 1:
                        best = size
                histogram[best] += 1
        rows.append(dict(n=n, graphs=total, ambiguous_full=ambiguous,
                         min_subdeck_histogram=dict(sorted(histogram.items())),
                         cards=sum(g.size() for g in graphs)))
    return rows


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
