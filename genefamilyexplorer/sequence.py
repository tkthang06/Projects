"""Core sequence functions for Gene Family Explorer. Pure Python, no dependencies.

Functions expect uppercase DNA; parse_fasta uppercases, so send input through it.
"""
from collections import Counter

# Includes IUPAC ambiguity codes (R=A/G pairs with Y=C/T, etc.) so mixed bases flip correctly
COMPLEMENT = str.maketrans("ACGTRYKMBVDHSWN", "TGCAYRMKVBHDSWN")

# Standard genetic code, codons ordered TTT, TTC, TTA, TTG, TCT, ...
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {a + b + c: _AAS[16 * i + 4 * j + k]
               for i, a in enumerate(_BASES)
               for j, b in enumerate(_BASES)
               for k, c in enumerate(_BASES)}


def parse_fasta(text):
    """Return a list of (header, sequence) records from FASTA text.

    A raw pasted sequence with no ">" line becomes one record with an empty
    header. Spaces, digits and other non-letters in sequence lines are dropped,
    so sequence copied from NCBI's numbered GenBank view also works.
    """
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append([line[1:], []])
        else:
            if not records:
                records.append(["", []])
            records[-1][1].append("".join(ch for ch in line if ch.isalpha()))
    return [(h, "".join(s).upper()) for h, s in records]


def gc_content(seq):
    """GC percent over A/C/G/T only, so N's don't pull it down. NaN if no ACGT."""
    acgt = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / acgt * 100 if acgt else float("nan")


def sliding_gc(seq, window=200, step=1):
    """(positions, GC percents) for each window start."""
    positions = list(range(0, len(seq) - window + 1, step))
    return positions, [gc_content(seq[i:i + window]) for i in positions]


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


def transcribe(seq):
    return seq.replace("T", "U")


def translate(seq):
    """Translate codon by codon; unknown codons (e.g. containing N) become X."""
    return "".join(CODON_TABLE.get(seq[i:i + 3], "X") for i in range(0, len(seq) - 2, 3))


def find_orfs(seq, min_aa=30):
    """Six-frame ORFs: first ATG after a stop, up to and including the next stop.

    Returns dicts with forward-strand coordinates (0-based, end exclusive), so
    plus and minus strand ORFs can be drawn on the same map. ORFs that run off
    the end of the sequence without a stop codon are skipped.
    """
    L = len(seq)
    orfs = []
    for strand, s in (("+", seq), ("-", reverse_complement(seq))):
        for frame in range(3):
            start = None
            for i in range(frame, L - 2, 3):
                codon = s[i:i + 3]
                if start is None and codon == "ATG":
                    start = i
                elif start is not None and CODON_TABLE.get(codon) == "*":
                    end = i + 3
                    if (i - start) // 3 >= min_aa:  # length check before translating
                        protein = translate(s[start:i])
                        fs, fe = (start, end) if strand == "+" else (L - end, L - start)
                        orfs.append({"strand": strand, "frame": frame, "start": fs,
                                     "end": fe, "length_nt": end - start, "protein": protein})
                    start = None
    return sorted(orfs, key=lambda o: -o["length_nt"])


def cpg_obs_exp(seq):
    """Observed/expected CpG ratio: CG count * length / (C count * G count)."""
    c, g = seq.count("C"), seq.count("G")
    return seq.count("CG") * len(seq) / (c * g) if c and g else 0.0


def cpg_islands(seq, window=200, min_gc=50, min_oe=0.6, min_len=200):
    """CpG islands (Gardiner-Garden & Frommer defaults) as (start, end) pairs.

    Every window with GC >= min_gc % and CpG obs/exp >= min_oe is marked, and
    overlapping marked windows merge into one island. Prefix sums keep it O(n)
    so a 50 kb gene region is instant. Stricter Takai-Jones: 500, 55, 0.65.
    """
    c, g, cg = [0], [0], [0]
    for i, b in enumerate(seq):
        c.append(c[-1] + (b == "C"))
        g.append(g[-1] + (b == "G"))
        cg.append(cg[-1] + (seq[i:i + 2] == "CG"))
    islands = []
    for i in range(len(seq) - window + 1):
        C, G = c[i + window] - c[i], g[i + window] - g[i]
        CG = cg[i + window - 1] - cg[i]  # CG dinucleotides that start and end inside the window
        if C and G and (C + G) * 100 >= min_gc * window and CG * window >= min_oe * C * G:
            if islands and i <= islands[-1][1]:
                islands[-1][1] = i + window
            else:
                islands.append([i, i + window])
    return [(s, e) for s, e in islands if e - s >= min_len]


def kmer_counts(seq, k=3):
    """Counts of each k-mer made only of A/C/G/T."""
    return Counter(kmer for kmer in (seq[i:i + k] for i in range(len(seq) - k + 1))
                   if set(kmer) <= set("ACGT"))
