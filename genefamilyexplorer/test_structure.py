"""Offline checks for structure.py. Run: python test_structure.py"""
from structure import find_motif, residue_plddt

# ADAMTS1's real active site, embedded in filler; GAPDH-like filler has none
assert find_motif("MKVLAHELGHVFNMPHDDAKL") == [(6, 16)]
assert find_motif("MGKVKVGVNGFGRIGRLVTRAAF") == []
assert find_motif("HEIGHLLGLSH" + "A" * 10 + "HELGHVFNMPH") == [(1, 11), (22, 32)]

pdb = "\n".join([
    "ATOM      1  N   MET A   1      11.104   6.134  -6.504  1.00 45.10           N",
    "ATOM      2  CA  MET A   1      11.639   6.071  -5.147  1.00 45.10           C",
    "ATOM      3  CA  HIS A   2      13.012   7.511  -4.300  1.00 97.62           C",
    "HETATM    4  CA  ZN  A 900      10.000  10.000  10.000  1.00  0.00          ZN",
])
assert residue_plddt(pdb) == {1: 45.10, 2: 97.62}  # CA atoms only, B-factor column
print("all checks passed")
