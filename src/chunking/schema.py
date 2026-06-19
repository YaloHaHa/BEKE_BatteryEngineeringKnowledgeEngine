"""Data schemas for chunking outputs.

A Chunk is what a chunker (2A / 2B / 2C) produces from a ParsedDocument's
sections. Every chunker MUST emit Chunks of this shape so the bake-off
eval can compare them on the same axes (length distribution, % crossing
section boundaries, retrieval Hit@k, etc.).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
	"""One retrieval-ready text unit and its provenance metadata.

	Provenance fields (source, section_idxs, heading_path, page_idx, slide_idx)
	are what enable retrieval citations later — without them, an answer can't
	point back to 'page 12 of thesis_chapter3.pdf, in the Methods section.'
	"""

	# ---- core payload ------------------------------------------------------
	text: str                         # the actual text payload
	source: str                       # absolute path to source file (= ParsedDocument.source)
	chunk_idx: int                    # 0-based ordinal of this chunk WITHIN its source document
	chunker: str                      # which approach produced it: "2a" | "2b" | "2c"

	# ---- provenance back to ParsedSection(s) ------------------------------
	# Singleton list for 2A (per-section split). Possibly multi-element for 2C
	# if a parent-document chunk spans multiple sections.
	#
	# ─── Algorithmic blank: dataclass mutable-default idiom ─────────────────
	# Why this is a blank: Python dataclasses reject `= []` as a default because
	# every instance would share the SAME list (a classic shared-state bug).
	# The dataclasses module ships `field(default_factory=...)` — pass it a
	# zero-arg callable that produces a fresh default each instance. For a
	# list-typed field, the zero-arg callable IS just the type itself.
	# Don't peek at the worked examples below until you've attempted this.
	section_idxs: list[int] = field(default_factory=list)                                # (think) algorithmic — mutable default idiom

	# ---- GIVEN worked examples (same pattern, different factories) --------
	heading_path: list[str] = field(default_factory=list)
	page_idx: Optional[int] = None
	slide_idx: Optional[int] = None
	meta: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable mapping for this chunk."""
		return asdict(self)


# ---------------------------------------------------------------------------
# Smoke test  (run: python -m src.chunking.schema)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	c = Chunk(
		text="Lithium-air batteries face challenges with...",
		source="/abs/path/to/thesis_chapter3.pdf",
		chunk_idx=0,
		chunker="2a",
		section_idxs=[3],
		heading_path=["Chapter 3", "3.1 Background"],
		page_idx=12,
	)
	print(f"chunk_idx     : {c.chunk_idx}")
	print(f"chunker       : {c.chunker}")
	print(f"section_idxs  : {c.section_idxs}")
	print(f"heading_path  : {c.heading_path}")
	print(f"meta default  : {c.meta}")
	# expected shape:
	# chunk_idx     : 0
	# chunker       : 2a
	# section_idxs  : [3]
	# heading_path  : ['Chapter 3', '3.1 Background']
	# meta default  : {}

	# Mutable-default sanity check — each instance MUST get its own meta dict.
	c2 = Chunk(text="x", source="y", chunk_idx=1, chunker="2a", section_idxs=[0])
	c.meta["touched_by_c"] = True
	assert "touched_by_c" not in c2.meta, \
		"BUG: instances share a meta dict — your default_factory isn't returning a fresh value."
	print("mutable-default isolation: OK")


# ---- HINTS (uncover only if stuck > 5 min) ----
# Algo blank — section_idxs default:
#   Concept:  dataclass mutable-default idiom (you flagged this as shaky in prefs)
#   Hint:     the type is list[int]. The zero-arg callable that creates a fresh
#             empty list of any element type IS just `list` itself (calling
#             `list()` returns []). Mirror the heading_path / meta lines.
#   Answer:   field(default_factory=list)


# ---- Reflection question ----
# Q1: Why does `section_idxs: list[int] = []` fail at class-definition time
#     (you'll get `ValueError: mutable default <class 'list'> for field
#     section_idxs is not allowed`)? Specifically: what would go wrong if
#     Python silently accepted it and just used the same list object as the
#     default? Sketch a 2-line example showing the bug it would create.
