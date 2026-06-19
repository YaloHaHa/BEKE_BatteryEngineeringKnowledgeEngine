"""Data schemas for parsed ingestion outputs."""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ParsedSection:
	"""One extracted text unit and its provenance metadata."""

	text: str
	section: Optional[str] = None
	heading_path: list[str] = field(default_factory=list)
	page_idx: Optional[int] = None
	slide_idx: Optional[int] = None
	speaker_notes: Optional[str] = None
	meta: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable mapping for this section."""

		return asdict(self)


@dataclass
class ParsedDocument:
	"""Parsed representation of one source file."""

	source: str
	doc_type: str
	title: Optional[str] = None
	sections: list[ParsedSection] = field(default_factory=list)
	meta: dict[str, Any] = field(default_factory=dict)

	def add_section(self, section: ParsedSection) -> None:
		"""Append a parsed section while preserving source order."""

		self.sections.append(section)

	@property
	def full_text(self) -> str:
		"""Join all non-empty section text into one text payload."""

		return "\n\n".join(section.text for section in self.sections if section.text.strip())

	@property
	def section_count(self) -> int:
		"""Return the number of extracted sections."""

		return len(self.sections)

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable mapping for this document."""

		return asdict(self)
	