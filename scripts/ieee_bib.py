"""Make paper/refs.bib IEEE-style and re-verify every DOI.

* journal names -> IEEE abbreviations (IEEE Reference Guide / ISO 4 forms)
* DOIs printed via `note = {doi: ...}` (IEEEtran.bst does not print the doi field)
* every DOI resolved against the Crossref API: title similarity and year compared with the bib entry
  -> paper/reference_check.csv

    python scripts/ieee_bib.py
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
BS = chr(92)

ABBREV = {
    "Econometrica": "Econometrica",
    "Journal of Financial Economics": "J. Financial Econ.",
    "The Journal of Finance": "J. Finance",
    "The Review of Financial Studies": "Rev. Financial Stud.",
    "The Journal of Portfolio Management": "J. Portfolio Manage.",
    "Notices of the American Mathematical Society": "Notices Amer. Math. Soc.",
    "Journal of the Royal Statistical Society: Series B (Methodological)": "J. Roy. Statist. Soc. Ser. B (Methodol.)",
    "Biometrics": "Biometrics",
    "Journal of Business \\& Economic Statistics": "J. Bus. Econ. Statist.",
    "Nature Machine Intelligence": "Nature Mach. Intell.",
    "Nature Human Behaviour": "Nature Human Behav.",
    "Social Psychological and Personality Science": "Social Psychol. Personality Sci.",
    "Journal of the American Statistical Association": "J. Amer. Statist. Assoc.",
    "Journal of Statistical Software": "J. Statist. Softw.",
    "Proceedings of the National Academy of Sciences": "Proc. Nat. Acad. Sci. USA",
    "Monthly Weather Review": "Monthly Weather Rev.",
    "Review of Quantitative Finance and Accounting": "Rev. Quant. Finance Account.",
    "OR Spectrum": "OR Spectr.",
    "IEEE Access": "IEEE Access",
    "Applied Intelligence": "Appl. Intell.",
    "Artificial Intelligence Review": "Artif. Intell. Rev.",
}
BOOKTITLE = {"Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining":
             "Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining"}


def crossref(doi: str) -> dict:
    req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"),
                                 headers={"User-Agent": "ieee-ref-check/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))["message"]


def plain(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\\[a-zA-Z]+|[{}\\'\"~^`]", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() and " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def main():
    src = BIB.read_text(encoding="utf-8")
    for full, ab in ABBREV.items():
        src = src.replace("journal = {" + full + "}", "journal = {" + ab + "}")
    for full, ab in BOOKTITLE.items():
        src = src.replace("booktitle = {" + full + "}", "booktitle = {" + ab + "}")
    # Harvey-Liu-Zhu: keep the ellipsis attached ("...and the cross-section of expected returns")
    src = src.replace("title = {{" + BS + "ldots} and the Cross-Section of Expected Returns}",
                      "title = {{" + BS + "ldots}and the Cross-Section of Expected Returns}")
    rows = []
    chunks = re.split(r"\n(?=@\w+\{)", src)
    entries = [(c.strip(), re.match(r"@\w+\{(\w+),", c.strip()).group(1)) for c in chunks if re.match(r"\s*@\w+\{", c)]
    for body, key in entries:
        m = re.search(r"doi = \{([^}]+)\}", body)
        title = re.search(r"\btitle = \{(.+?)\},\s*\n?\s*(journal|booktitle|publisher|author)", body, re.S)
        year = re.search(r"year = \{(\d{4})\}", body)
        t_bib = title.group(1) if title else ""
        y_bib = year.group(1) if year else ""
        if not m:
            rows.append({"key": key, "doi": "", "status": "NO DOI (book) - not machine-verifiable", "bib_year": y_bib,
                         "crossref_year": "", "title_similarity": ""})
            continue
        doi = m.group(1)
        try:
            cr = crossref(doi)
            t_cr = (cr.get("title") or [""])[0]
            years = {str(cr.get(k, {}).get("date-parts", [[None]])[0][0]) for k in ("published-print", "published-online", "issued")}
            sim = difflib.SequenceMatcher(None, plain(t_bib), plain(t_cr)).ratio()
            ok = sim > 0.85 and y_bib in years
            rows.append({"key": key, "doi": doi, "status": "VERIFIED" if ok else "CHECK", "bib_year": y_bib,
                         "crossref_year": "/".join(sorted(y for y in years if y != "None")), "title_similarity": f"{sim:.2f}"})
        except Exception as e:                      # network or resolution failure
            rows.append({"key": key, "doi": doi, "status": f"UNRESOLVED ({type(e).__name__})", "bib_year": y_bib,
                         "crossref_year": "", "title_similarity": ""})
        if "note = {doi:" not in body:
            new = body.replace("doi = {" + doi + "}", "doi = {" + doi + "},\n  note = {doi: " + BS + "url{" + doi + "}}")
            src = src.replace(body, new)
    BIB.write_text(src, encoding="utf-8")
    with open(ROOT / "paper" / "reference_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['status'][:40]:40s} {r['key']:20s} bib {r['bib_year']} crossref {r['crossref_year']:10s} sim {r['title_similarity']}")


if __name__ == "__main__":
    main()
