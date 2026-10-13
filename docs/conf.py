# Sphinx configuration for the Tilthex Sensitivity RL documentation.
# Build from this folder with `pixi run build` (output in _build/html/index.html).

project = "Tilthex Sensitivity RL"
author = "Marco Maffeo"
copyright = "2026, Marco Maffeo"

extensions = [
    "myst_parser",        # Markdown pages (the README files of the repository)
    "autoapi.extension",  # API reference read from the source code, without importing it
    "sphinx.ext.napoleon",  # Google-style docstrings (Args:, Returns:)
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
# "Attributes:" sections as a field list: the attributes are already documented as class members.
napoleon_use_ivar = True
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_member_order = "groupwise"

# --- Inheritance diagrams -----------------------------------------------------
# One diagram per module, drawn by Graphviz as SVG.
graphviz_output_format = "svg"
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
