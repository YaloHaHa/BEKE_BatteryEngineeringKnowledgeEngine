"""Chunk statistics — compute and compare length distributions across 2A / 2B / 2C.

Day-3 deliverable: run this module to produce chunks_*.jsonl and print the
comparison table that goes into chunking_comparison.md.
"""

import json
import statistics
from pathlib import Path

from .schema import Chunk

from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------------

# compute_stats: summarise one set of chunks with length distribution + quality checks.
# Input:  chunks — list[Chunk] from any chunker
# Output: dict with count, length percentiles, pct_boundary_crossing, pct_empty


def compute_stats(chunks: list[Chunk]) -> dict:
	"""Return summary statistics for a list of Chunks.

	Metrics returned
	----------------
	count                : total number of chunks
	min / mean / median / max : character-length distribution
	p25 / p75            : lower and upper quartiles of chunk length
	pct_boundary_crossing: % of chunks spanning more than one ParsedSection
	pct_empty            : % of chunks with no non-whitespace text
	"""
	if not chunks:
		return {"count": 0}

	# Step 1: collect character lengths — one int per chunk
	lengths = [len(c.text) for c in chunks]                     # (easy)

	n = len(lengths)
	sorted_lengths = sorted(lengths)

	# Step 2: quartile indices — p25 is 1/4 of the way through the sorted list
	# integer division gives the nearest index; no external library needed.
	p25 = sorted_lengths[round(n*0.25)]                           # (think) index for 25th percentile
	p75 = sorted_lengths[round(n*0.75)]                           # (easy)  index for 75th percentile

	# Step 3: boundary-crossing chunks
	# A chunk whose section_idxs has more than one element spans multiple
	# ParsedSections — a sign that the chunker crossed a structural boundary.
	# For 2A and 2B this should always be 0; for 2C it may be non-zero.
	pct_crossing = sum(1 for c in chunks if len(c.section_idxs) >1) / n * 100                                 # (think) % of chunks where len(section_idxs) > 1

	# Step 4: empty chunks — text that is blank after stripping whitespace
	pct_empty = sum(1 for c in chunks if not c.text or not c.text.strip()) / n * 100                                   # (easy)

	return {
		"count":                n,
		"min":                  sorted_lengths[0],
		"p25":                  p25,
		"mean":                 round(statistics.mean(lengths), 1),
		"median":               statistics.median(lengths),
		"p75":                  p75,
		"max":                  sorted_lengths[-1],
		"pct_boundary_crossing": round(pct_crossing, 1),
		"pct_empty":            round(pct_empty, 1),
	}


# ---------------------------------------------------------------------------
# I/O helpers  (given — no blanks)
# ---------------------------------------------------------------------------

def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
	"""Write one Chunk JSON object per line."""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as fh:
		for chunk in chunks:
			fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
	print(f"wrote {len(chunks):,} chunks → {path}")


def load_chunks_jsonl(path: Path) -> list[Chunk]:
	"""Load chunks from a JSONL file produced by write_chunks_jsonl."""
	chunks = []
	with path.open(encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if line:
				chunks.append(Chunk(**json.loads(line)))
	return chunks


# ---------------------------------------------------------------------------
# Comparison table  (given — no blanks)
# ---------------------------------------------------------------------------

def print_comparison(stats_by_approach: dict[str, dict]) -> None:
	"""Print a side-by-side stats table for all chunking approaches."""
	keys = ["count", "min", "p25", "mean", "median", "p75", "max",
	        "pct_boundary_crossing", "pct_empty"]
	col_w = 14
	label_w = 26

	header = f"{'metric':<{label_w}}" + "".join(
		f"{name:>{col_w}}" for name in stats_by_approach
	)
	print(header)
	print("-" * len(header))
	for key in keys:
		row = f"{key:<{label_w}}"
		for stats in stats_by_approach.values():
			val = stats.get(key, "n/a")
			row += f"{val:>{col_w}}" if isinstance(val, str) else f"{val:>{col_w}.1f}" if isinstance(val, float) else f"{val:>{col_w}}"
		print(row)


# ---------------------------------------------------------------------------
# Smoke test  (run: python -m src.chunking.stats)
# Prerequisite: place a .docx at tests/fixtures/sample.docx
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from ..ingestion.loaders import load_docx_1b
	from .chunkers import chunk_recursive, chunk_semantic, chunk_parent_document

	doc = load_docx_1b(Path("tests/fixtures/sample.docx"))
	embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # wire up your HuggingFace embeddings to test this
	chunks_2a = chunk_recursive(doc, chunk_size=500, chunk_overlap=80)
	chunks_2b = chunk_semantic(doc, embeddings=embeddings)  # wire up your HuggingFace embeddings to test this
	chunks_2c = chunk_parent_document(doc, parent_size=1000, child_size=200, child_overlap=30)

	stats = {
		"2a_recursive": compute_stats(chunks_2a),
		"2b_semantic":  compute_stats(chunks_2b),
		"2c_parent":    compute_stats(chunks_2c),
	}
	print_comparison(stats)
	# expected shape (numbers vary by file):
	# metric                    2a_recursive     2b_semantic     2c_parent
	# --------------------------------------------------
	# count                               15            28
	# min                                  4             4
	# ...
	# pct_boundary_crossing              0.0           0.0
	# pct_empty                          0.0           0.0

	# Write to JSONL (paths match project plan Day-3 deliverables)
	write_chunks_jsonl(chunks_2a, Path("chunks_recursive.jsonl"))
	write_chunks_jsonl(chunks_2b, Path("chunks_semantic.jsonl"))
	write_chunks_jsonl(chunks_2c, Path("chunks_hierarchical.jsonl"))
	# Note: 2B (chunk_semantic) requires an embeddings model — wire it up
	# separately once your HuggingFace embeddings are configured.


# ---- HINTS (uncover only if stuck > 5 min) ----
# blank 1 — lengths:
#   You want the character count of each chunk's text field.
#   Attribute is c.text; built-in for string length is len().
#
# blank 2 — p25 index:
#   The 25th percentile of n sorted items is at position n/4.
#   Use integer division so it's a valid index.
#
# blank 3 — p75 index:
#   Same logic as p25 but 3/4 of the way through.
#
# blank 4 — pct_boundary_crossing:
#   A chunk crosses a boundary when its section_idxs list has more than one element.
#   Express as a percentage: (count of such chunks / total) * 100.
#
# blank 5 — pct_empty:
#   A chunk is empty when c.text.strip() is falsy.
#   Same percentage pattern as blank 4.


# ---- Reflection question ----
# Q5: pct_boundary_crossing is 0.0 for all three of your chunkers as currently
#     written — because every chunker splits within a section, never across one.
#     Describe a chunking strategy where boundary crossing would be non-zero
#     and explain whether that would be a bug or an intentional design choice.
