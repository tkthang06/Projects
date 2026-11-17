"""Fetch a human gene's mRNA (RefSeq Select) and genomic region from NCBI, cached in data/."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from Bio import SeqIO

from sequence import parse_fasta

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HUMAN_CODING = '"Homo sapiens"[orgn] AND alive[prop] AND "genetype protein coding"[prop]'


def _get(endpoint, **params):
    params["tool"] = "gene_family_explorer"
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
    for wait in (0.4, 1, 2):  # NCBI allows 3 requests/second without an API key
        time.sleep(wait)
        try:
            return urllib.request.urlopen(url, timeout=60).read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429:  # 429 = too many requests: back off and retry
                raise
    time.sleep(4)
    return urllib.request.urlopen(url, timeout=60).read().decode()


def _gene_summaries(gene_ids):
    """NCBI Gene summaries: symbol, full name, cytoband and genomic location for each ID."""
    result = json.loads(_get("esummary.fcgi", db="gene", id=",".join(gene_ids), retmode="json"))["result"]
    return [result[i] for i in gene_ids if i in result]


def search_genes(query, limit=25):
    """Search NCBI Gene for human protein-coding genes, most-studied first.

    One word is a gene symbol or symbol prefix ('MMP', 'ADAMTS*', 'TP53'); several words
    search gene and protein names ('matrix metallopeptidase'). A single word that matches
    no symbol ('collagen') falls back to a name search.
    """
    q = query.strip().replace('"', "")
    if not q:
        return []
    word = q.rstrip("*")
    terms = ([f"({word}[PREF] OR {word}*[PREF])", f"{word}[TITL]"] if " " not in q else [f"{q}[TITL]"])
    for term in terms:
        ids = json.loads(_get("esearch.fcgi", db="gene", term=f"({term}) AND {HUMAN_CODING}", retmode="json",
                              retmax=limit))["esearchresult"]["idlist"]
        if ids:
            return [{"symbol": s["name"], "name": s["description"], "band": s["maplocation"], "gene_id": s["uid"]}
                    for s in _gene_summaries(ids)]
    return []


def _gene_info(symbol):
    """The gene's NCBI ID, full name, cytoband and coordinates (start > stop means minus strand)."""
    ids = json.loads(_get("esearch.fcgi", db="gene", term=f"{symbol}[sym] AND {HUMAN_CODING}",
                          retmode="json"))["esearchresult"]["idlist"]
    if not ids:
        raise ValueError(f"'{symbol}' isn't a human protein-coding gene symbol on NCBI Gene")
    s = _gene_summaries(ids[:1])[0]
    loc = s["genomicinfo"][0]
    a, b = loc["chrstart"], loc["chrstop"]  # 0-based
    return {"gene_id": s["uid"], "name": s["description"], "band": s["maplocation"],
            "chrom": loc["chraccver"], "start": min(a, b) + 1, "end": max(a, b) + 1, "strand": "-" if a > b else "+"}


def _download(symbol):
    """Look the gene up on NCBI and save mRNA (.gb), gene region (.fna) and gene info (.json)."""
    info = _gene_info(symbol)
    term = f'{symbol}[Gene Name] AND "Homo sapiens"[Organism] AND refseq_select[filter] AND biomol_mrna[prop]'
    ids = json.loads(_get("esearch.fcgi", db="nuccore", term=term, retmode="json"))["esearchresult"]["idlist"]
    if not ids:
        raise ValueError(f"No human RefSeq Select mRNA found for gene '{symbol}'")
    gb = _get("efetch.fcgi", db="nuccore", id=ids[0], rettype="gb", retmode="text")
    fna = _get("efetch.fcgi", db="nuccore", id=info["chrom"], seq_start=info["start"], seq_stop=info["end"],
               strand=2 if info["strand"] == "-" else 1, rettype="fasta")

    # NCBI can answer with an error message instead of data; never cache that
    if not gb.startswith("LOCUS") or not fna.startswith(">"):
        raise RuntimeError(f"NCBI returned no sequence for {symbol}")
    os.makedirs(DATA, exist_ok=True)
    for ext, text in ((".gb", gb), (".fna", fna), (".json", json.dumps(info))):
        with open(os.path.join(DATA, symbol + ext), "w", encoding="utf-8") as f:
            f.write(text)


def fetch_gene(symbol):
    """Return a dict with the gene's mRNA, annotated CDS, protein, genomic region and NCBI info.

    Downloads on first use; afterwards reads data/<SYMBOL>.gb/.fna/.json offline.
    """
    symbol = symbol.strip().upper()
    path = os.path.join(DATA, symbol)
    if not all(os.path.exists(path + ext) for ext in (".gb", ".fna", ".json")):
        _download(symbol)
    with open(path + ".json", encoding="utf-8") as f:
        info = json.load(f)
    if "gene_id" not in info:  # cached before gene IDs were stored: look it up once
        info = _gene_info(symbol)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f)
    with open(path + ".gb", encoding="utf-8") as f:
        record = SeqIO.read(f, "genbank")
    with open(path + ".fna", encoding="utf-8") as f:
        _, region = parse_fasta(f.read())[0]
    cds = next(feat for feat in record.features if feat.type == "CDS")
    return {"symbol": symbol, "mrna_id": record.id, "mrna": str(record.seq).upper(),
            "cds_start": int(cds.location.start), "cds_end": int(cds.location.end),
            "protein": cds.qualifiers["translation"][0], "region": region, **info}
