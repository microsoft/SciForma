"""
latex_parser.py — Extract methodology sections and figure captions from LaTeX.

Supports:
  - Single .tex file
  - Directory of .tex files (searches all, merges methodology sections)
  - ArXiv tar.gz (auto-extract)

Usage:
    from generate.latex_parser import LatexParser
    parser = LatexParser("paper.tex")
    sections = parser.get_methodology_sections()  # list of (title, text)
    captions = parser.get_figure_captions()        # list of str
"""
from __future__ import annotations
import re, os, tarfile, glob
from pathlib import Path


# Section headings we treat as "methodology"
_METHOD_KEYWORDS = re.compile(
    r"(method|approach|model|framework|architecture|system|pipeline|proposed|our\s+method)",
    re.IGNORECASE,
)

_SECTION_RE = re.compile(
    r"\\(?:sub)*section\*?\{([^}]+)\}", re.IGNORECASE
)
_CAPTION_RE = re.compile(
    r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", re.IGNORECASE
)
_LABEL_CMD  = re.compile(r"\\[a-zA-Z]+\{[^}]*\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*")


def _clean(text: str) -> str:
    text = _COMMENT_RE.sub("", text)         # remove % comments
    text = _LABEL_CMD.sub("", text)          # remove \cmd{...}
    text = re.sub(r"\s+", " ", text)         # collapse whitespace
    return text.strip()


class LatexParser:
    def __init__(self, source: str | Path):
        source = Path(source)
        if source.suffix == ".gz" or source.suffix == ".tar":
            self._tex = self._extract_tar(source)
        elif source.is_dir():
            self._tex = self._load_dir(source)
        else:
            self._tex = source.read_text(errors="ignore")


    def get_methodology_sections(self, max_chars: int = 8000) -> list[tuple[str, str]]:
        """Return list of (section_title, section_text) for method sections."""
        sections = self._split_sections(self._tex)
        results = []
        for title, body in sections:
            if _METHOD_KEYWORDS.search(title):
                cleaned = _clean(body)[:max_chars]
                if len(cleaned) > 100:
                    results.append((title, cleaned))
        return results

    def get_figure_captions(self) -> list[str]:
        """Return all \\caption{...} texts in the document."""
        caps = []
        for m in _CAPTION_RE.finditer(self._tex):
            c = _clean(m.group(1))
            if len(c) > 20:
                caps.append(c)
        return caps

    def get_methodology_text(self, max_chars: int = 8000) -> str:
        """Return merged methodology text (best single string for planner input)."""
        secs = self.get_methodology_sections(max_chars=max_chars)
        if not secs:
            # Fallback: return first 8000 chars of full doc
            return _clean(self._tex)[:max_chars]
        return "\n\n".join(body for _, body in secs)[:max_chars]


    def _split_sections(self, tex: str) -> list[tuple[str, str]]:
        parts = _SECTION_RE.split(tex)
        # parts: [pre, title1, body1, title2, body2, ...]
        sections = []
        for i in range(1, len(parts) - 1, 2):
            title = parts[i].strip()
            body  = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append((title, body))
        return sections

    def _load_dir(self, d: Path) -> str:
        texts = []
        for f in sorted(d.glob("**/*.tex")):
            try:
                texts.append(f.read_text(errors="ignore"))
            except Exception:
                pass
        return "\n".join(texts)

    def _extract_tar(self, path: Path) -> str:
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            with tarfile.open(path) as tar:
                tar.extractall(tmp)
            return self._load_dir(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
