"""Protein structure for a human gene: AlphaFold model via UniProt, plus active-site motif search.

Cached in data/<SYMBOL>_alphafold.pdb and data/<SYMBOL>_alphafold.json.
"""
import json
import os
import re
import urllib.parse
import urllib.request

from ncbi import DATA

# Zinc-binding active site of metalloproteases: three histidines (positions 1, 5, 11)
# hold the zinc and the E is the catalytic base. The textbook HExxHxxGxxH also
# requires a G at position 8, but only ADAMTS5 of the six ADAMTS genes here has
# it (the others read HELGHVFNMPH...), so the pattern keeps just the shared core.
ZINC_MOTIF = r"HE..H.....H"


def _json(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


def fetch_structure(symbol):
    """Return {uniprot, sequence, mean_plddt, pdb} for the gene's reviewed human UniProt entry."""
    symbol = symbol.strip().upper()
    path = os.path.join(DATA, symbol + "_alphafold")
    if not (os.path.exists(path + ".pdb") and os.path.exists(path + ".json")):
        query = urllib.parse.quote(f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true")
        hits = _json(f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession&format=json")["results"]
        if not hits:
            raise ValueError(f"No reviewed human UniProt entry for '{symbol}'")
        acc = hits[0]["primaryAccession"]
        model = _json(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")[0]
        pdb = urllib.request.urlopen(model["pdbUrl"], timeout=120).read().decode()
        if not pdb.lstrip().startswith(("HEADER", "ATOM", "REMARK", "TITLE")):
            raise RuntimeError(f"AlphaFold returned no model for {acc}")
        os.makedirs(DATA, exist_ok=True)
        with open(path + ".pdb", "w", encoding="utf-8") as f:
            f.write(pdb)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"uniprot": acc, "sequence": model["uniprotSequence"],
                       "mean_plddt": model["globalMetricValue"]}, f)
    with open(path + ".json", encoding="utf-8") as f:
        info = json.load(f)
    with open(path + ".pdb", encoding="utf-8") as f:
        info["pdb"] = f.read()
    return info


def residue_plddt(pdb):
    """{residue number: pLDDT} read from the B-factor column of each CA atom (AlphaFold stores confidence there)."""
    return {int(line[22:26]): float(line[60:66]) for line in pdb.splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"}


def find_motif(protein, pattern=ZINC_MOTIF):
    """1-based (start, end) residue numbers of every match of a regex motif."""
    return [(m.start() + 1, m.end()) for m in re.finditer(pattern, protein)]


if __name__ == "__main__":
    import sys
    from ncbi import fetch_gene

    for s in sys.argv[1:] or ["ADAMTS5"]:
        info, gene = fetch_structure(s), fetch_gene(s)
        plddt = residue_plddt(info["pdb"])
        print(f"{s}: UniProt {info['uniprot']}, {len(info['sequence'])} aa, mean pLDDT {info['mean_plddt']:.0f}, "
              f"same as RefSeq protein: {info['sequence'] == gene['protein']}, "
              f"zinc motif at {find_motif(info['sequence'])}, residues in model {len(plddt)}")
