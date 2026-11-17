"""Fetch a human gene's mRNA (RefSeq Select) and genomic region from NCBI, cached in data/."""
import json
import os
import time
import urllib.parse
import urllib.request

from Bio import SeqIO

from sequence import parse_fasta

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _get(endpoint, **params):
    time.sleep(0.4)  # NCBI allows 3 requests/second without an API key
    params["tool"] = "gene_family_explorer"
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
    return urllib.request.urlopen(url, timeout=60).read().decode()


def _download(symbol):
    """Look the gene up on NCBI and save mRNA (.gb), gene region (.fna) and coordinates (.json)."""
    term = (f'{symbol}[Gene Name] AND "Homo sapiens"[Organism] '
            "AND refseq_select[filter] AND biomol_mrna[prop]")
    ids = json.loads(_get("esearch.fcgi", db="nuccore", term=term, retmode="json"))["esearchresult"]["idlist"]
    if not ids:
        raise ValueError(f"No human RefSeq Select mRNA found for gene '{symbol}'")
    gb = _get("efetch.fcgi", db="nuccore", id=ids[0], rettype="gb", retmode="text")

    gene_ids = json.loads(_get("esearch.fcgi", db="gene", term=f"{symbol}[sym] AND human[orgn]",
                               retmode="json"))["esearchresult"]["idlist"]
    info = json.loads(_get("esummary.fcgi", db="gene", id=gene_ids[0], retmode="json"))
    loc = info["result"][gene_ids[0]]["genomicinfo"][0]
    a, b = loc["chrstart"], loc["chrstop"]  # 0-based; start > stop means minus strand
    coords = {"chrom": loc["chraccver"], "start": min(a, b) + 1, "end": max(a, b) + 1,
              "strand": "-" if a > b else "+"}
    fna = _get("efetch.fcgi", db="nuccore", id=coords["chrom"], seq_start=coords["start"],
               seq_stop=coords["end"], strand=2 if a > b else 1, rettype="fasta")

    # NCBI can answer with an error message instead of data; never cache that
    if not gb.startswith("LOCUS") or not fna.startswith(">"):
        raise RuntimeError(f"NCBI returned no sequence for {symbol}")
    os.makedirs(DATA, exist_ok=True)
    for ext, text in ((".gb", gb), (".fna", fna), (".json", json.dumps(coords))):
        with open(os.path.join(DATA, symbol + ext), "w", encoding="utf-8") as f:
            f.write(text)


def fetch_gene(symbol):
    """Return a dict with the gene's mRNA, annotated CDS, protein and genomic region.

    Downloads on first use; afterwards reads data/<SYMBOL>.gb/.fna/.json offline.
    """
    symbol = symbol.strip().upper()
    path = os.path.join(DATA, symbol)
    if not all(os.path.exists(path + ext) for ext in (".gb", ".fna", ".json")):
        _download(symbol)
    with open(path + ".gb", encoding="utf-8") as f:
        record = SeqIO.read(f, "genbank")
    with open(path + ".fna", encoding="utf-8") as f:
        _, region = parse_fasta(f.read())[0]
    with open(path + ".json", encoding="utf-8") as f:
        coords = json.load(f)
    cds = next(feat for feat in record.features if feat.type == "CDS")
    return {"symbol": symbol, "mrna_id": record.id, "mrna": str(record.seq).upper(),
            "cds_start": int(cds.location.start), "cds_end": int(cds.location.end),
            "protein": cds.qualifiers["translation"][0], "region": region, **coords}
