"""Offline checks for compare.py on made-up genes. Run: python test_compare.py"""
import random

import numpy as np

from compare import feature_table, pca, signature_distance, signature_table, window_table
from sequence import translate

random.seed(1)


def fake_gene(name, bases, n=5000):
    cds = "ATG" + "GCT" * 200 + "TAA"
    region = "".join(random.choices(bases, k=n)) + cds  # CDS sits at the end of the region
    return {"symbol": name, "mrna": cds, "cds_start": 0, "cds_end": len(cds),
            "protein": translate(cds[:-3]), "region": region}


genes = [fake_gene(f"AT{i}", "AAATTTGC") for i in range(3)] + [fake_gene(f"GC{i}", "GGGCCCAT") for i in range(3)]

f = feature_table(genes)
assert f.loc["AT0", "protein_aa"] == 201 and f.loc["AT0", "utr5_bp"] == 0
assert f.loc["GC0", "gc_region_%"] > 60 > f.loc["AT0", "gc_region_%"]

w = window_table(genes, window=1000)
assert len(w) == 6 * (len(genes[0]["region"]) // 1000)  # full windows only

d = signature_distance(signature_table(genes))
assert np.allclose(np.diag(d), 0) and np.allclose(d, d.T)
assert d.loc["AT0", "AT1"] < 50  # same made-up composition -> near-identical signatures

scores, var = pca(f)  # PC1 should split the AT-rich genes from the GC-rich ones
pc1 = scores["PC1"]
assert (pc1.filter(like="AT") > 0).all() != (pc1.filter(like="GC") > 0).all()
assert 0 < var[1] <= var[0] <= 100
print("all checks passed")
