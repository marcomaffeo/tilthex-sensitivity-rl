# Sphinx configuration for the Tilthex Sensitivity RL documentation.
# Build from this folder with `pixi run build` (output in _build/html/index.html).

import re

project = "Tilthex Sensitivity RL"
author = "Marco Maffeo"
copyright = "2026, Marco Maffeo"

extensions = [
    "myst_parser",        # Markdown pages (the README files of the repository)
    "autoapi.extension",  # API reference read from the source code, without importing it
    "sphinx.ext.inheritance_diagram",  # class inheritance trees (needs Graphviz)
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", ".pixi"]

# --- API reference (sphinx-autoapi) ------------------------------------------
# The code is parsed statically: JAX, ROS 2 and genomix do not need to be installed.
autoapi_type = "python"
autoapi_dirs = [
    "../sensitivity_rl/sensitivity_rl",   # training framework   -> package `sensitivity_rl`
    "../sensitivity_rl_telekyb/src",      # deployment (ROS 2)   -> package `src`
    "../data_analysis",                   # flight analysis      -> modules analyze_* / compare_* / export_*
]
autoapi_ignore = [
    "*tk3-tilthex-control*",  # interactive console script, not a module
    "*/.pixi/*",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
]
# Class docstring followed by the __init__ docstring, which holds the constructor arguments.
autoapi_python_class_content = "both"
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_member_order = "groupwise"

# --- Inheritance diagrams -----------------------------------------------------
# One diagram per module, drawn by Graphviz as png.
graphviz_output_format = "png"
inheritance_graph_attrs = {"rankdir": "LR", "size": '"10.0, 20.0"', "fontsize": 12}
inheritance_node_attrs = {"shape": "box", "fontsize": 12, "height": 0.3, "style": "rounded"}

# --- Markdown -----------------------------------------------------------------
myst_heading_anchors = 3
# Links in the README files point to folders of the repository, not to pages of this site.
suppress_warnings = [
    "myst.xref_missing", "myst.header", "autoapi.python_import_resolution",
    "ref.python",  # same attribute name in several environments
]

# --- HTML ----------------------------------------------------------------------
html_theme = "furo"
html_title = "Tilthex Sensitivity RL"


# --- Docstring to Table converter (Args, Attributes, Returns) -----------------
_SECTION_HEADERS = {
    "Args:": ("Parameters", ("Parameter", "Type", "Description"), 3),
    "Arguments:": ("Parameters", ("Parameter", "Type", "Description"), 3),
    "Parameters:": ("Parameters", ("Parameter", "Type", "Description"), 3),
    "Attributes:": ("Attributes", ("Attribute", "Type", "Description"), 3),
    "Returns:": ("Returns", ("Return Type", "Description"), 2),
    "Return:": ("Returns", ("Return Type", "Description"), 2),
}

_ARG_RE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$")
_RET_RE = re.compile(r"^\s*(?:([^:]+?)\s*:\s*)?(.*)$")


def _parse_section_items(block_lines: list[str], cols: int) -> list[tuple[str, ...]]:
    items: list[list[str]] = []
    base_indent: int | None = None

    for line in block_lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None or indent <= base_indent:
            base_indent = indent
            if cols == 3:
                m = _ARG_RE.match(line)
                if m:
                    name, typ, desc = m.group(1), m.group(2) or "—", m.group(3).strip()
                    items.append([f"``{name}``", f"``{typ}``" if typ != "—" else "—", desc])
                else:
                    items.append([line.strip(), "—", ""])
            else:
                m = _RET_RE.match(line)
                if m and m.group(1):
                    typ, desc = m.group(1).strip(), m.group(2).strip()
                    items.append([f"``{typ}``", desc])
                else:
                    items.append(["—", line.strip()])
        else:
            if items:
                items[-1][-1] = (items[-1][-1] + " " + line.strip()).strip()

    return [tuple(it) for it in items]


def _render_list_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    if not rows:
        return []
    widths = "20 25 55" if len(headers) == 3 else "30 70"
    out = [
        "",
        f"**{title}**",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {widths}",
        "",
    ]
    for idx, row in enumerate([headers, *rows]):
        for col_idx, cell in enumerate(row):
            prefix = "   * - " if col_idx == 0 else "     - "
            out.append(f"{prefix}{cell}")
    out.append("")
    return out


def _convert_docstring_to_tables(app, what, name, obj, options, lines: list[str]) -> None:
    new_lines: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()
        if stripped in _SECTION_HEADERS:
            title, headers, cols = _SECTION_HEADERS[stripped]
            header_indent = len(lines[i]) - len(lines[i].lstrip())
            i += 1
            block: list[str] = []
            while i < n:
                line = lines[i]
                s = line.strip()
                if s in _SECTION_HEADERS:
                    break
                if s and (len(line) - len(line.lstrip())) <= header_indent:
                    break
                block.append(line)
                i += 1
            rows = _parse_section_items(block, cols)
            new_lines.extend(_render_list_table(title, headers, rows))
        else:
            new_lines.append(lines[i])
            i += 1

    lines[:] = new_lines


def setup(app):
    app.connect("autodoc-process-docstring", _convert_docstring_to_tables)