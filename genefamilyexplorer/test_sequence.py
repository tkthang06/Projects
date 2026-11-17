"""Check sequence.py against Biopython. Run: python test_sequence.py"""
import random
from collections import Counter

from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

from sequence import (cpg_islands, cpg_obs_exp, find_orfs, gc_content, kmer_counts,
                      parse_fasta, reverse_complement, transcribe, translate)


def bio_orfs(seq, min_aa):
    """Reference ORFs from Biopython's translation: in each frame, split on stops, take from the first M."""
    found = []
    for strand, s in (("+", Seq(seq)), ("-", Seq(seq).reverse_complement())):
        for frame in range(3):
            sub = s[frame:]
            chunks = str(sub[:len(sub) // 3 * 3].translate()).split("*")
            for chunk in chunks[:-1]:  # last chunk has no stop codon
                if "M" in chunk and len(chunk) - chunk.index("M") >= min_aa:
                    found.append((strand, chunk[chunk.index("M"):]))
    return Counter(found)


random.seed(0)
for trial in range(50):
    seq = "".join(random.choices("ACGT", k=random.randint(500, 3000)))
    assert reverse_complement(seq) == str(Seq(seq).reverse_complement())
    assert transcribe(seq) == str(Seq(seq).transcribe())
    assert translate(seq) == str(Seq(seq[:len(seq) // 3 * 3]).translate())
    assert abs(gc_content(seq) - gc_fraction(seq) * 100) < 1e-9

    orfs = find_orfs(seq, min_aa=20)
    assert Counter((o["strand"], o["protein"]) for o in orfs) == bio_orfs(seq, 20)
    for o in orfs:  # coordinates point at the right bases on the forward strand
        piece = seq[o["start"]:o["end"]]
        if o["strand"] == "-":
            piece = reverse_complement(piece)
        assert translate(piece) == o["protein"] + "*"

assert parse_fasta(">a\nac\ngt\n\n>b x\nNN\n") == [("a", "ACGT"), ("b x", "NN")]
assert parse_fasta("acgt\nTT\n") == [("", "ACGTTT")]                     # pasted, no header
assert parse_fasta("1 atgcc gtaa\n11 ttg\n") == [("", "ATGCCGTAATTG")]   # NCBI numbered view
iupac = "ACGTRYKMBVDHSWN" * 3
assert reverse_complement(iupac) == str(Seq(iupac).reverse_complement())
assert translate("ATGNNNTAA") == "MX*"
assert kmer_counts("ACGNACG", 2) == Counter({"AC": 2, "CG": 2})

def naive_island_windows(seq, w=200):
    """Window starts passing the island test, recomputed from scratch with str.count."""
    return [i for i in range(len(seq) - w + 1)
            if (win := seq[i:i + w]).count("C") and win.count("G")
            and gc_content(win) >= 50 and cpg_obs_exp(win) >= 0.6]


# AT-rich background with one planted CpG-rich stretch at 1000..1600
seq = ("".join(random.choices("AATTCG", k=1000)) + "".join(random.choices("CG", k=600))
       + "".join(random.choices("AATTCG", k=1000)))
starts = naive_island_windows(seq)
islands = cpg_islands(seq)
assert islands == [(starts[0], starts[-1] + 200)], islands  # one merged island spanning every passing window
assert islands[0][0] <= 1000 and islands[0][1] >= 1600
assert cpg_islands("AT" * 500) == []
print("all checks passed")
