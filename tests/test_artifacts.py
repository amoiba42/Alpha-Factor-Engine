"""Output layout checks: semantic names, tear-sheet folders, and no broken links in the docs.

The link and file checks need generated results; they are skipped before the first pipeline run.
"""
import re
from pathlib import Path

import pytest

from src.config import FIGURES, PROJECT_ROOT, SUMMARY_TABLES, figure_relpath, table_relpath

RESULTS = PROJECT_ROOT / "results"
LINK = re.compile(r"\]\(([^)\s]+)\)")


def test_figure_names_are_semantic_and_foldered():
    for key, rel in FIGURES.items():
        folder, name = rel.split("/")
        assert folder in ("tearsheet", "diagnostics")
        assert re.fullmatch(r"[a-z][a-z0-9_]*\.png", name), name
        assert not re.match(r"\d", name)
    assert len(set(FIGURES.values())) == len(FIGURES)


def test_table_routing():
    assert table_relpath("performance_full") == "summary/performance_full.csv"
    assert table_relpath("lookback_selection", "md") == "diagnostics/lookback_selection.md"
    assert all(table_relpath(n).startswith("summary/") for n in SUMMARY_TABLES)


needs_results = pytest.mark.skipif(not (RESULTS / "RESULTS.md").exists(), reason="pipeline not run yet")


@needs_results
def test_all_figures_exist_and_no_loose_files():
    for key in FIGURES:
        assert (RESULTS / "figures" / figure_relpath(key)).exists(), key
    for d in ("figures", "tables"):
        loose = [p.name for p in (RESULTS / d).iterdir() if p.is_file()]
        assert not loose, f"loose files in results/{d}: {loose}"
    for name in SUMMARY_TABLES:
        for ext in ("csv", "md"):
            assert (RESULTS / "tables" / table_relpath(name, ext)).exists(), name


@needs_results
@pytest.mark.parametrize("doc", [PROJECT_ROOT / "README.md", RESULTS / "RESULTS.md"])
def test_doc_links_resolve(doc: Path):
    links = [l for l in LINK.findall(doc.read_text()) if not l.startswith(("http", "#"))]
    assert links, f"no local links found in {doc.name}"
    missing = [l for l in links if not (doc.parent / l).exists()]
    assert not missing, f"broken links in {doc.name}: {missing}"
