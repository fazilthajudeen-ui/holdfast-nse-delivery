"""Camera-ready checks for paper/main.pdf (IEEE PDF eXpress-style, run without poppler):

  fonts      every font on every page and inside every embedded figure (Form XObject): type, embedded,
             subset, Type 3 (equivalent of `pdffonts`)
  pdf        version, encryption, bookmarks/outlines, link annotations, page count, page size
  text       abstract word count, Index Terms order, figure/table citation-before-appearance and order,
             British spellings, double spaces, forbidden claims, key-number consistency across sections

    python scripts/check_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "main.pdf"
problems = []


def fonts_in(res, seen, out, where):
    if res is None:
        return
    res = res.get_object()
    for name, ref in (res.get("/Font") or {}).items():
        f = ref.get_object()
        key = id(f)
        if key in seen:
            continue
        seen.add(key)
        sub = f.get("/Subtype")
        base = str(f.get("/BaseFont", "(none)"))
        desc = f.get("/FontDescriptor")
        if sub == "/Type0":
            d = f["/DescendantFonts"][0].get_object()
            desc = d.get("/FontDescriptor")
            sub = f"Type0/{d.get('/Subtype')}"
        emb = False
        if desc is not None:
            desc = desc.get_object()
            emb = any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))
        if sub == "/Type3":
            emb = True                                       # glyphs are inline procedures
        subset = bool(re.match(r"/?[A-Z]{6}\+", base.lstrip("/")))
        out.append({"font": base.lstrip("/"), "type": str(sub).lstrip("/"), "embedded": emb, "subset": subset,
                    "where": where})
    for name, ref in (res.get("/XObject") or {}).items():
        x = ref.get_object()
        if x.get("/Subtype") == "/Form":
            fonts_in(x.get("/Resources"), seen, out, where + " (figure)")


def main():
    r = pypdf.PdfReader(str(PDF))
    head = PDF.read_bytes()[:8].decode("latin-1")
    fonts, seen = [], set()
    for i, p in enumerate(r.pages, 1):
        fonts_in(p.get("/Resources"), seen, fonts, f"p{i}")
    uniq = {}
    for f in fonts:
        uniq.setdefault((f["font"], f["type"], f["embedded"], f["subset"]), []).append(f["where"])
    print(f"PDF header {head.strip()} | encrypted: {r.is_encrypted} | pages: {len(r.pages)} | "
          f"page size: {float(r.pages[0].mediabox.width):.0f} x {float(r.pages[0].mediabox.height):.0f} pt")
    print(f"bookmarks/outline entries: {len(r.outline)} | link annotations: "
          f"{sum(1 for p in r.pages for a in (p.get('/Annots') or []) if a.get_object().get('/Subtype') == '/Link')}")
    print("\nname                                   type            emb  sub  first seen")
    for (n, t, e, s), w in sorted(uniq.items()):
        print(f"{n[:38]:38s} {t[:15]:15s} {'yes' if e else 'NO ':3s}  {'yes' if s else 'no ':3s}  {w[0]}")
        if not e:
            problems.append(f"font not embedded: {n}")
        if "Type3" in t:
            problems.append(f"Type 3 font: {n}")
        if not s:
            problems.append(f"font not subset: {n}")
    if r.is_encrypted:
        problems.append("PDF is encrypted")
    if float(head[5:8]) < 1.4:
        problems.append(f"PDF version {head}")

    pages = [p.extract_text() for p in r.pages]
    text = "\n".join(pages)
    flat = re.sub(r"\s+", " ", text)
    # abstract
    a = flat[flat.find("Abstract") + len("Abstract"):flat.find("Index Terms")]
    nwords = len(re.findall(r"[A-Za-z0-9][\w.,%+\-−/()']*", a))
    print(f"\nabstract words: {nwords} (limit 250)")
    if nwords > 250:
        problems.append("abstract over 250 words")
    it = flat[flat.find("Index Terms") + len("Index Terms") + 1:flat.find("I. I")].strip(" —-")
    terms = [t.strip().rstrip(".") for t in it.split(",")]
    print("index terms:", terms, "| alphabetical:", [t.lower() for t in terms] == sorted(t.lower() for t in terms))
    if [t.lower() for t in terms] != sorted(t.lower() for t in terms):
        problems.append("index terms not alphabetical")
    # floats: cited before appearing, numbered in order of first citation
    refs_page = next(i for i, p in enumerate(pages, 1) if "REFERENCES" in p)
    for kind, pat_cap, pat_ref in [("Fig.", r"Fig\. (\d+)\. ", r"Fig\. (\d+)(?!\. [A-Z])"),
                                   ("TABLE", r"TABLE ([IVX]+)\n", r"Table ([IVX]+)")]:
        caps, firsts = {}, {}
        for i, p in enumerate(pages, 1):
            for m in re.finditer(pat_cap, p):
                caps.setdefault(m.group(1), i)
            for m in re.finditer(pat_ref, p):
                firsts.setdefault(m.group(1), i)
        order = sorted(firsts, key=lambda k: (firsts[k], text.find(f"{'Fig. ' if kind == 'Fig.' else 'Table '}{k}")))
        print(f"{kind}: float page {caps} | first cited page {firsts} | citation order {order}")
        for k, pg in caps.items():
            if k not in firsts:
                problems.append(f"{kind} {k} never cited")
            elif firsts[k] > pg:
                problems.append(f"{kind} {k} appears (p{pg}) before its first citation (p{firsts[k]})")
            if pg > refs_page or (pg == refs_page and pages[pg - 1].find(f"{k}.") > pages[pg - 1].find("REFERENCES") >= 0
                                  and kind == "Fig."):
                problems.append(f"{kind} {k} floats into/after the references")
    # language and claims
    brit = re.findall(r"\b\w*(?:favour|colour|centred|labelled|modelling|behaviour|analys(?:e|ed|ing)\b|"
                      r"synchronis|organis|optimis|summaris|recognis)\w*", flat)
    brit = [w for w in brit if w not in ("Behav",)]
    print("British spellings:", sorted(set(brit)) or "none")
    if [w for w in brit if not w.startswith("Behaviour")]:
        problems.append(f"British spelling: {sorted(set(brit))}")
    body = flat[:flat.find("REFERENCES")]
    dbl = len(re.findall(r"[A-Za-z,.;] {2,}[A-Za-z]", body))
    print("double spaces in body text:", dbl)
    for pat, why in [(r"real[- ]time", "real-time claim"), (r"(?<!not )measure[sd]? (?:order-flow )?toxicity", "toxicity measured"),
                     (r"(?<!No hypothesis is )(?<!None is )(?<!not )\bsupported\b", "a hypothesis called supported")]:
        hits = [m.group(0) for m in re.finditer(pat, body, re.I)]
        print(f"{why}: {hits or 'none'}")
        if hits:
            problems.append(f"{why}: {hits}")
    # key numbers in each section
    sec = {"abstract": a, "results": flat[flat.find("RESULTS A. Exploratory"):flat.find("VI. D")],
           "discussion": flat[flat.find("VI. D"):flat.find("VII. C")],
           "conclusion": flat[flat.find("VII. C"):flat.find("Data and Code") if "Data and Code" in flat else flat.find("DATA AND")]}
    expect = {"H1 -0.04 pp": "−0.04", "H1 p 0.987": "0.987", "H2 -2.81 pp": "−2.81", "H2 q 0.286": "0.286",
              "H2 no-pilot -5.65": "−5.65", "no-pilot p 0.026": "0.026", "no-pilot q 0.128": "0.128",
              "AUC 0.460": "0.460", "AUC 0.526": "0.526", "MDE 6.26": "6.26", "MDE 5.71": "5.71",
              "break-even 8.8": "8.8 pp", "Nifty +9.7%": "+9.7%"}
    print("\nkey numbers by section:")
    for k, v in expect.items():
        where = [s for s, t in sec.items() if v in t]
        print(f"  {k:20s} -> {where}")
    stale = [s for s in ["−2.8 pp", "−5.6 pp", "5.83 million"] if s in flat]
    if stale:
        problems.append(f"stale values present: {stale}")
    print("\nPROBLEMS:", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
