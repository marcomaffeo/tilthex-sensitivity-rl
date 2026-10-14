"""Render the parameters, return value, exceptions and attributes of every docstring as a table.

Napoleon turns the Google-style sections (``Args``, ``Returns``, ``Raises``,
``Attributes``) into reST field lists. This extension runs after it and
rewrites those fields as a list-table with three columns: name, type and
description.
"""

import re

_FIELD = re.compile(r"^:(param|type|returns?|rtype|raises?|ivar|vartype)(?:\s+([^:]+))?:\s*(.*)$")


def _collect(lines):
    """Split the docstring into the field lines and the rest.

    Returns the index of the first field line, the parsed fields in order and
    the remaining lines.
    """
    fields, rest, first = [], [], None
    i = 0
    while i < len(lines):
        m = _FIELD.match(lines[i])
        if not m:
            rest.append(lines[i])
            i += 1
            continue
        if first is None:
            first = len(rest)
        kind, name, text = m.group(1), (m.group(2) or "").strip(), [m.group(3).strip()]
        i += 1
        # continuation lines are indented
        while i < len(lines) and (lines[i].startswith(" ") and lines[i].strip()):
            text.append(lines[i].strip())
            i += 1
        fields.append((kind, name, " ".join(t for t in text if t)))
    return first, fields, rest


def _rows(fields):
    """Table rows (name, type, description) of the parameters, attributes, return value and exceptions."""
    params, ptypes, ivars, vtypes = {}, {}, {}, {}
    porder, iorder = [], []
    ret_desc, ret_type, raises = "", "", []
    for kind, name, text in fields:
        if kind == "param":
            # napoleon writes ":param *args:" for varargs; the type may be inline: ":param int x:"
            parts = name.split()
            if len(parts) > 1:
                ptypes[parts[-1]] = " ".join(parts[:-1])
                name = parts[-1]
            params[name] = text
            porder.append(name)
        elif kind == "type":
            ptypes[name] = text
        elif kind == "ivar":
            ivars[name] = text
            iorder.append(name)
        elif kind == "vartype":
            vtypes[name] = text
        elif kind in ("returns", "return"):
            ret_desc = text
        elif kind == "rtype":
            ret_type = text
        elif kind in ("raises", "raise"):
            raises.append((name, text))
    tables = []
    if porder:
        tables.append(("Parameters", ["Parameter", "Type", "Description"],
                       [(f"``{n}``", ptypes.get(n, ""), params[n]) for n in porder]))
    if ret_desc or ret_type:
        tables.append(("Returns", ["", "Type", "Description"], [("return", ret_type, ret_desc)]))
    if raises:
        tables.append(("Raises", ["", "Exception", "Condition"], [("raises", n, t) for n, t in raises]))
    if iorder:
        tables.append(("Attributes", ["Attribute", "Type", "Description"],
                       [(f"``{n}``", vtypes.get(n, ""), ivars[n]) for n in iorder]))
    return tables


def _table(caption, header, rows):
    """reST list-table with the given caption, header and rows."""
    out = [f".. list-table:: {caption}", "   :header-rows: 1", "   :widths: 22 28 50",
           "   :class: api-table", ""]
    for row in [header] + rows:
        out.append(f"   * - {row[0] or ' '}")
        for cell in row[1:]:
            out.append(f"     - {cell or ' '}")
    out.append("")
    return out


def process_docstring(app, what, name, obj, options, lines):
    """Replace the field lists written by napoleon with tables, in place."""
    first, fields, rest = _collect(lines)
    if not fields:
        return
    new = []
    for caption, header, rows in _rows(fields):
        new += _table(caption, header, rows)
    if first and rest[first - 1].strip():
        new.insert(0, "")
    lines[:] = rest[:first] + new + rest[first:]


def setup(app):
    """Register the docstring hook after napoleon (priority 500)."""
    app.connect("autodoc-process-docstring", process_docstring, priority=600)
    app.add_css_file("api_tables.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
