"""
calibrate_guardrail.py
----------------------
Empirically calibrate the off-domain guardrail thresholds
(``retrieval_min_dense`` / ``retrieval_min_lexical`` in settings.py).

The guardrail declines a query deterministically only when BOTH the top dense
cosine AND the top BM25 score fall below their thresholds (see
``RAGAgent._off_domain``). This script measures those two top scores for a panel
of *in-domain* queries (real lab-ops questions the corpus should answer) and
*out-of-domain* queries (cooking, weather, general knowledge — the corpus must
decline these), then reports thresholds that separate the two sets.

Run against the fixture corpus (offline, deterministic)::

    python scripts/calibrate_guardrail.py

Both score signals are now corpus-size stable (dense cosine is inherently so;
BM25 is normalised to [0, 1] in lexical_index.py, which cancels the idf/log(N)
drift), so thresholds should hold as the corpus grows. Still worth re-running
against the REAL Google Drive corpus to confirm the in/out separation holds on
real content — pass --docs to point at a folder of markdown, or wire in the
Drive source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from document_source import LocalMarkdownSource          # noqa: E402
from embeddings import FastEmbedProvider                 # noqa: E402
from lexical_index import BM25Index                       # noqa: E402
from retrieval import DocumentStore                      # noqa: E402
from vector_store import ChromaVectorStore               # noqa: E402

FIXTURE_DOCS = REPO_ROOT / "tests" / "fixtures" / "docs"

# Real lab-ops questions the corpus SHOULD answer (must NOT be declined).
# Includes the welcome-screen prompts and a few paraphrases / other languages.
IN_DOMAIN = [
    "How do I clean the centrifuge rotor after each session?",
    "What's the recommended way to clean a laptop screen in the lab?",
    "How do I book an instrument and check its availability?",
    "How do I request access to an internal application?",
    "How do I report an IT incident?",
    "create an incident in ServiceNow",
    "How do I check the current sample stock?",
    "What is the decontamination procedure after a sample spill?",
    "my virtual session keeps disconnecting, what do I do?",
    "biological spill cleanup procedure",
    "Wie reinige ich die Zentrifuge?",                    # German paraphrase
    "comment réserver un instrument ?",                   # French paraphrase
]

# Off-domain questions the corpus has no business answering (SHOULD be declined).
OUT_OF_DOMAIN = [
    "How do I bake a chocolate cake?",
    "What's the weather in Madrid tomorrow?",
    "Who won the World Cup in 2018?",
    "Write me a Python function to reverse a linked list.",
    "What is the capital of Australia?",
    "Recommend a good science fiction novel.",
    "How do I change a tire on my car?",
    "What's a good recipe for pasta carbonara?",
    "Translate 'good morning' into Japanese.",
    "What are the rules of chess?",
]


def _build_store(docs_path: Path) -> DocumentStore:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="calib_chroma_"))
    store = ChromaVectorStore(path=str(tmp), collection_name="calibration")
    docs = DocumentStore(
        source=LocalMarkdownSource(docs_path),
        embedder=FastEmbedProvider(),
        vector_store=store,
        manifest_path=str(tmp / "manifest.json"),
        lexical_index=BM25Index(),          # hybrid mode (dense + BM25)
    )
    report = docs.ingest()
    print(f"Ingested {report.documents_seen} docs, "
          f"{report.chunks_written} chunks from {docs_path}\n")
    return docs


def _measure(docs: DocumentStore, queries: list[str]) -> list[tuple[str, float, float]]:
    rows = []
    for q in queries:
        r = docs.retrieve_scored(q, k=4)
        rows.append((q, r.max_dense, r.max_lexical))
    return rows


def _print_block(title: str, rows: list[tuple[str, float, float]]) -> None:
    print(title)
    print(f"  {'max_dense':>9}  {'max_lexical':>11}   query")
    for q, d, l in rows:
        print(f"  {d:>9.3f}  {l:>11.2f}   {q[:60]}")
    print()


def _suggest(in_rows, out_rows) -> None:
    in_dense = [d for _, d, _ in in_rows]
    in_lex = [l for _, _, l in in_rows]
    out_dense = [d for _, d, _ in out_rows]
    out_lex = [l for _, _, l in out_rows]

    print("=" * 64)
    print("SUMMARY")
    print(f"  in-domain  dense: min={min(in_dense):.3f}  max={max(in_dense):.3f}")
    print(f"  out-domain dense: min={min(out_dense):.3f}  max={max(out_dense):.3f}")
    print(f"  in-domain  lex:   min={min(in_lex):.2f}  max={max(in_lex):.2f}")
    print(f"  out-domain lex:   min={min(out_lex):.2f}  max={max(out_lex):.2f}")
    print()

    # The two signals play DIFFERENT roles, so they are calibrated differently:
    #
    #  * Dense is the primary discriminator. It typically separates in/out of
    #    corpus with a clean gap (in-domain cosine high, off-domain low). Pick
    #    the midpoint of that gap so neither set is misclassified.
    #
    #  * Lexical (BM25) is NOT a reliable off-domain discriminator — non-English
    #    in-domain queries score ~0 against an English corpus, while off-domain
    #    English queries pick up a few points from common tokens. Its only job is
    #    to RESCUE a low-dense query that has a strong exact-keyword hit (a part
    #    number / SOP code the embedder underweights). So set it as a bar a clear
    #    margin ABOVE the highest off-domain BM25 score, not in a (non-existent)
    #    gap. Below that bar, dense alone decides.
    dense_gap = min(in_dense) > max(out_dense)
    sugg_dense = round((max(out_dense) + min(in_dense)) / 2, 2) if dense_gap else 0.40
    # Rescue bar: a clear margin above the highest off-domain lexical score.
    # max_lexical is normalised to [0, 1], so use a fractional margin and clamp.
    sugg_lex = round(min(max(out_lex) + 0.10, 1.0), 2)
    if not dense_gap:
        print("!! WARNING: dense scores do NOT separate in/out cleanly on this "
              "corpus — inspect the table above before trusting min_dense.\n")
    print("SUGGESTED THRESHOLDS (decline only if BOTH below):")
    print(f"  retrieval_min_dense   = {sugg_dense}")
    print(f"  retrieval_min_lexical = {sugg_lex}")
    print()

    # Verify the suggestion against the panel.
    def declined(d, l):
        return d < sugg_dense and l < sugg_lex

    in_wrong = [q for q, d, l in in_rows if declined(d, l)]
    out_kept = [q for q, d, l in out_rows if not declined(d, l)]
    print("VERIFICATION at suggested thresholds:")
    print(f"  in-domain wrongly DECLINED:  {len(in_wrong)}/{len(in_rows)}")
    for q in in_wrong:
        print(f"     ! {q}")
    print(f"  out-of-domain NOT declined:  {len(out_kept)}/{len(out_rows)}")
    for q in out_kept:
        print(f"     ~ {q}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=FIXTURE_DOCS,
                    help="folder of markdown docs to ingest (default: test fixtures)")
    args = ap.parse_args()

    docs = _build_store(args.docs)
    in_rows = _measure(docs, IN_DOMAIN)
    out_rows = _measure(docs, OUT_OF_DOMAIN)
    _print_block("IN-DOMAIN (must be answered):", in_rows)
    _print_block("OUT-OF-DOMAIN (must be declined):", out_rows)
    _suggest(in_rows, out_rows)


if __name__ == "__main__":
    main()
