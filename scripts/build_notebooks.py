"""Build the Jupyter notebooks from their percent-format sources and execute them.

Sources live in notebooks/source/NN_name.py. Cells are separated by lines that
start with "# %%"; a "# %% [markdown]" cell holds markdown written as comment
lines ("# " prefix stripped). The executed .ipynb files are written to
notebooks/.

    python scripts/build_notebooks.py            # build and run all
    python scripts/build_notebooks.py 03 05      # only notebooks whose name starts with these
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "source"
OUT = ROOT / "notebooks"


def parse(path: Path) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in text.splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb.cells = cells
    return nb


def main(prefixes: list[str]) -> None:
    sources = sorted(SRC.glob("*.py"))
    if prefixes:
        sources = [s for s in sources if any(s.name.startswith(p) for p in prefixes)]
    for src in sources:
        nb = parse(src)
        t = time.time()
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT)}})
        out = OUT / (src.stem + ".ipynb")
        nbformat.write(nb, out)
        errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
                  if o.get("output_type") == "error"]
        images = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
                     if "image/png" in o.get("data", {}))
        print(f"{out.name}: {len(nb.cells)} cells, {images} figures, {len(errors)} errors, {time.time() - t:.0f}s")
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
