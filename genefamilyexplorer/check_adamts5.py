"""Step 2 check on real data: does find_orfs recover a gene's annotated CDS, and where are its CpG islands?

Run: python check_adamts5.py [GENE]   (default ADAMTS5; data comes from ncbi.fetch_gene)
"""
import sys

from ncbi import fetch_gene
from sequence import cpg_islands, find_orfs, gc_content

g = fetch_gene(sys.argv[1] if len(sys.argv) > 1 else "ADAMTS5")

# 1. mRNA: the longest ORF should be exactly the annotated coding sequence
longest = find_orfs(g["mrna"], min_aa=100)[0]
print(f"{g['symbol']} {g['mrna_id']}: {len(g['mrna'])} bp mRNA")
print(f"  annotated CDS : {g['cds_start']}..{g['cds_end']}  ({len(g['protein'])} aa)")
print(f"  longest ORF   : {longest['start']}..{longest['end']}  ({len(longest['protein'])} aa, strand {longest['strand']})")
assert longest["strand"] == "+"
assert (longest["start"], longest["end"]) == (g["cds_start"], g["cds_end"])
assert longest["protein"] == g["protein"]
print("  MATCH: longest ORF = annotated CDS, protein identical")

# 2. Gene region (exons + introns, given 5'->3' in the gene's direction)
gene = g["region"]
big_orfs = find_orfs(gene, min_aa=100)
print(f"\nGene region {g['chrom']}:{g['start']}-{g['end']} ({g['strand']}): {len(gene)} bp, GC {gc_content(gene):.1f}%")
print(f"  coding share  : {(g['cds_end'] - g['cds_start']) / len(gene) * 100:.1f}% of the region is CDS")
print(f"  ORFs >= 100 aa in the gene region: {len(big_orfs)}, longest {len(big_orfs[0]['protein']) if big_orfs else 0} aa "
      f"(the real protein is {len(g['protein'])} aa, split across exons)")
islands = cpg_islands(gene)
print(f"  CpG islands   : {len(islands)}")
for s, e in islands:
    print(f"    {s:>6}..{e:<6} {e - s:>5} bp  GC {gc_content(gene[s:e]):.0f}%")
