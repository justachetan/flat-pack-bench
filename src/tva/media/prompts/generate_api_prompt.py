from __future__ import annotations
import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import os
import os.path as osp
import inspect, textwrap, re
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from contextlib import contextmanager
from contextvars import ContextVar

# ---- Global default via ContextVar (safe for threads/async) -----------------
_API_DESC_DEFAULT_EXPORT: ContextVar[bool] = ContextVar(
    "_API_DESC_DEFAULT_EXPORT", default=False
)

# ---- Decorator --------------------------------------------------------------
def api_desc(
    description: str,
    *,
    export: bool | None = None,
    body: Optional[str] = None,
    include_code: bool = True,
    display_signature: Optional[str] = None,   # NEW: optional override
    parent_blurb: Optional[str] = None,
):
    """
    Attach an LLM-facing description to a class, method, or function.
    If `export` is None, falls back to the current default set by the context manager.

    New/Updated semantics:
      - body: optional text to appear as the implementation body (after the docstring).
              If None, we will try to extract the actual implementation body from source.
      - include_code: if False, do not include the implementation body (methods/functions only).
                      Docstrings still appear if present. Defaults to True.
      - display_signature: optional string to override how the signature is rendered
                           in the prompt (e.g., "(self, x: int) -> str").
    """
    def _wrap(obj):
        obj.__api_desc__ = textwrap.dedent(description).strip()
        obj.__api_body__ = None if body is None else textwrap.dedent(body).strip()
        obj.__api_include_code__ = bool(include_code)
        if display_signature is not None:
            obj.__api_signature_override__ = display_signature.strip()
        if parent_blurb is not None:
            obj.__api_parent_blurb__ = textwrap.dedent(parent_blurb).strip()
        should_export = _API_DESC_DEFAULT_EXPORT.get() if export is None else export
        if should_export:
            obj.__api_export__ = True
        else:
            obj.__api_export__ = False
        return obj
    return _wrap

# Attach a handy context manager to the decorator namespace
@contextmanager
def _default_export_cm(value: bool):
    token = _API_DESC_DEFAULT_EXPORT.set(value)
    try:
        yield
    finally:
        _API_DESC_DEFAULT_EXPORT.reset(token)

api_desc.default_export = _default_export_cm  # type: ignore[attr-defined]

# ---- Introspection helpers --------------------------------------------------
def _is_exported(o: Any) -> bool:
    return bool(getattr(o, "__api_export__", False))

def _include_code(o: Any) -> bool:
    return bool(getattr(o, "__api_include_code__", True))

def _extract_body_from_source(func: Any) -> Optional[str]:
    try:
        lines, _ = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return None
    if not lines:
        return None

    i, n = 0, len(lines)

    # --- Skip decorators, including multi-line decorator argument blocks
    while i < n and lines[i].lstrip().startswith("@"):
        i += 1
        # consume all continuation lines of THIS decorator until we hit
        # the start of another decorator or the 'def' line
        while i < n and not lines[i].lstrip().startswith(("@", "def ", "async def ")):
            i += 1

    # --- Advance to the 'def' (or 'async def') line
    while i < n and not lines[i].lstrip().startswith(("def ", "async def ")):
        i += 1
    if i < n:
        i += 1  # move to first body line

    body_lines = lines[i:]

    # Strip leading blank lines
    while body_lines and body_lines[0].strip() == "":
        body_lines = body_lines[1:]

    # Remove a leading triple-quoted docstring ("""...""" or '''...''')
    if body_lines:
        first = body_lines[0].lstrip()
        if first.startswith('"""') or first.startswith("'''"):
            quote = first[:3]
            # If docstring ends on the same line
            if first.count(quote) >= 2:
                body_lines = body_lines[1:]
            else:
                j = 1
                while j < len(body_lines):
                    if quote in body_lines[j]:
                        j += 1
                        break
                    j += 1
                body_lines = body_lines[j:]

    # Strip any leading blanks again
    while body_lines and body_lines[0].strip() == "":
        body_lines = body_lines[1:]

    if not body_lines:
        return None

    # Dedent common indentation of the body block, keep internal structure
    body = textwrap.dedent("".join(body_lines)).rstrip("\n")
    body = body.strip("\n")
    return body or None

def _get_desc(o: Any) -> str:
    d = getattr(o, "__api_desc__", None)
    if d:
        return d
    if o.__doc__:
        return textwrap.dedent(o.__doc__).strip()
    return ""

def _signature_for(obj: Any, is_class: bool) -> str:
    # If an explicit display signature override was provided, use it.
    override = getattr(obj, "__api_signature_override__", None)
    if override:
        # Expect the override to include surrounding parentheses, e.g. "(self) -> int"
        return override

    if is_class:
        init = getattr(obj, "__init__", None)
        if init and init is not object.__init__:
            try:
                return str(inspect.signature(init))
            except ValueError:
                return "()"
        return "()"
    try:
        return str(inspect.signature(obj))
    except ValueError:
        return "()"

def _public_own_methods(cls: type) -> List[Tuple[str, Callable]]:
    out: List[Tuple[str, Callable]] = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if member.__qualname__.split(".")[0] != cls.__name__:
            continue
        # OLD: if name.startswith("_"): continue
        # NEW: allow __init__; skip other private/dunder
        if name == "__init__":
            continue
        if name.startswith("_"):
            continue
        # Only include exported members; descriptions/docstrings alone should not force inclusion
        # if _is_exported(member) or getattr(member, "__api_desc__", None) or member.__doc__:
        if _is_exported(member):
            out.append((name, member))
    try:
        out.sort(key=lambda kv: kv[1].__code__.co_firstlineno)
    except Exception:
        out.sort(key=lambda kv: kv[0])
    return out

# ---- Rendering --------------------------------------------------------------
def _ensure_methods_section(base_desc: str, methods: List[Tuple[str, Callable]]) -> str:
    """Attach a runtime-generated Methods section that mirrors `methods`.

    Any existing Methods block is stripped before inserting the regenerated
    content so the docstring stays in sync with the filtered method list.
    """

    def _strip_existing_methods(desc: str) -> str:
        match = re.search(r"(^|\n)\s*Methods\n\s*-{2,}\n", desc)
        if not match:
            return desc.rstrip()
        return desc[: match.start()].rstrip()

    def _build_methods_block(descs: List[Tuple[str, Callable]]) -> str:
        def _format_blurb(text: str) -> List[str]:
            normalized = textwrap.dedent(text).strip()
            if not normalized:
                return []
            compact = " ".join(normalized.split())
            if not compact:
                return []
            wrapped = textwrap.wrap(
                compact,
                width=max(1, 100 - 8),
                break_long_words=False,
            )
            return ["        " + line for line in wrapped]

        entries: List[str] = []
        for mname, method in descs:
            parent_blurb = getattr(method, "__api_parent_blurb__", None)
            if not parent_blurb:
                continue
            blurb_lines = _format_blurb(parent_blurb)
            if not blurb_lines:
                continue
            msig = _signature_for(method, is_class=False)
            entries.append("    " + mname + msig)
            entries.extend(blurb_lines)
        if not entries:
            return ""
        header = ["Methods", "-------"]
        return "\n".join(header + entries)

    prefix = _strip_existing_methods(base_desc)
    methods_block = _build_methods_block(methods)

    if prefix and methods_block:
        return "\n\n".join([prefix, methods_block])
    if methods_block:
        return methods_block
    return prefix

def _render_class_block(
    cls: type,
    *,
    methods: Optional[List[Tuple[str, Callable]]] = None,
) -> str:
    cls_name = cls.__name__
    desc = _get_desc(cls)
    methods = _public_own_methods(cls) if methods is None else methods
    desc = _ensure_methods_section(desc, methods) if desc else ""

    # Indent the class docstring body so content aligns with method indentation
    if desc:
        indented_desc = textwrap.indent(desc, "    ")
        doc = f'    """\n{indented_desc}\n    """'
    else:
        doc = '    """\n    """'

    block = [
        f"class {cls_name}:",
        doc,
        "",
    ]
    
    for mname, m in methods:
        msig = _signature_for(m, is_class=False)
        mdesc = _get_desc(m)  # docstring text (if any)
        include_code = _include_code(m)
        explicit_body = getattr(m, "__api_body__", None)

        block.append(f"    def {mname}{msig}:")
        # (1) Always include docstring if present (export determines inclusion of the block)
        if mdesc:
            block.append(textwrap.indent(f'"""{mdesc}\n"""', "        "))

        # (2) Implementation body: respect include_code
        if include_code:
            # explicit override -> actual source body -> pass
            if explicit_body is not None:
                body = explicit_body
            else:
                body = _extract_body_from_source(m)

            if body and body.strip():
                block.append(textwrap.indent(body, "        "))
            else:
                block.append("        pass")
        else:
            # No code body; ensure syntactic validity if there was no docstring
            if not mdesc:
                block.append("        pass")

        block.append("")
    return "\n".join(block).rstrip()

def _render_function_block(fn: Callable) -> str:
    name = fn.__name__
    sig = _signature_for(fn, is_class=False)
    desc = _get_desc(fn)  # docstring (if any)
    include_code = _include_code(fn)
    explicit_body = getattr(fn, "__api_body__", None)

    lines = [f"def {name}{sig}:"]
    # Always include docstring if present (export controls inclusion of the block)
    if desc:
        lines.append(textwrap.indent(f'"""{desc}\n"""', "    "))

    if include_code:
        # explicit override -> actual source body -> pass
        if explicit_body is not None:
            body = explicit_body
        else:
            body = _extract_body_from_source(fn)

        if body and body.strip():
            lines.append(textwrap.indent(body, "    "))
        else:
            lines.append("    pass")
    else:
        # No code body; ensure syntactic validity if there was no docstring
        if not desc:
            lines.append("    pass")

    return "\n".join(lines)

# ---- Selection utilities ----------------------------------------------------
Patternish = Union[str, re.Pattern, Callable[[Any], bool]]

def _match_any(obj: Any, selectors: Iterable[Patternish]) -> bool:
    if not selectors:
        return False
    qn = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
    nm = getattr(obj, "__name__", "")
    for s in selectors:
        if callable(s):
            try:
                if s(obj):
                    return True
            except Exception:
                continue
        elif isinstance(s, re.Pattern):
            if s.search(qn) or s.search(nm):
                return True
        elif isinstance(s, str):
            if s == qn or s == nm or qn.startswith(s) or nm.startswith(s):
                return True
    return False

def _collect_examples_for_classes(classes: List[type]) -> str:
    """Call an optional `_examples()` on each class and join results with a blank line."""
    chunks: List[str] = []
    for cls in classes:
        ex = getattr(cls, "_examples", None)
        if callable(ex):
            try:
                text = ex()
            except Exception:
                text = None
            if isinstance(text, str) and text.strip():
                chunks.append(textwrap.dedent(text).strip())
    return "\n>>>\n>>>\n".join(chunks) if len(chunks) > 0 else ""


# ---- Prompt builder ---------------------------------------------------------
ModuleT = object  # just for readability

def generate_api_prompt(
    module: Optional[ModuleT] = None,                      # backward-compat
    *,
    modules: Optional[Iterable[ModuleT]] = None,           # NEW: multi-module
    include_imports: Iterable[str] = ("import math",),
    class_order: Optional[List[str]] = None,
    include: Optional[Iterable[Patternish]] = None,
    exclude: Optional[Iterable[Patternish]] = None,
    include_functions: bool = True,
    include_classes: bool = True,
    example_files: Optional[Iterable[str]] = None,
) -> str:
    """
    Build the code-like API prompt from one or more modules.

    You can call either:
        generate_api_prompt(module=my_module)
    or:
        generate_api_prompt(modules=[mod_a, mod_b, ...])

    Filtering:
      - include: keep only items matching any selector
      - exclude: drop items matching any selector
      Selectors may be strings (exact/prefix on __qualname__/__name__),
      compiled regexes, or predicates (Callable[[obj], bool]).
    """
    # ---- normalize modules input
    if modules is None:
        if module is None:
            raise ValueError("Pass either `module=` or `modules=`")
        modules = [module]
    else:
        # ignore `module=` if `modules=` is provided
        if not isinstance(modules, Iterable) or isinstance(modules, (str, bytes)):
            modules = [modules]  # type: ignore[assignment]
        else:
            modules = list(modules)

    # ---- header imports
    lines: List[str] = []
    seen_imports = set()
    for imp in include_imports:
        if imp not in seen_imports:
            lines.append(imp)
            seen_imports.add(imp)
    lines.append("")

    # ---- collect exported symbols across all modules
    include_selectors = tuple(include) if include is not None else ()
    exclude_selectors = tuple(exclude) if exclude is not None else ()

    class_entries: List[Tuple[type, List[Tuple[str, Callable]]]] = []
    funcs: List[Callable] = []
    for mod in modules:
        for _, obj in inspect.getmembers(mod):
            if include_classes and inspect.isclass(obj) and _is_exported(obj):
                class_entries.append((obj, _public_own_methods(obj)))
            elif include_functions and inspect.isfunction(obj) and _is_exported(obj):
                funcs.append(obj)
    # ---- apply include/exclude
    if include_selectors or exclude_selectors:
        filtered_entries: List[Tuple[type, List[Tuple[str, Callable]]]] = []
        for cls, methods in class_entries:
            if exclude_selectors and _match_any(cls, exclude_selectors):
                continue

            methods_after_exclude = list(methods)
            if exclude_selectors:
                methods_after_exclude = [
                    (name, method)
                    for name, method in methods_after_exclude
                    if not _match_any(method, exclude_selectors)
                ]

            if include_selectors:
                if _match_any(cls, include_selectors):
                    filtered_methods = methods_after_exclude
                else:
                    filtered_methods = [
                        (name, method)
                        for name, method in methods_after_exclude
                        if _match_any(method, include_selectors)
                    ]
                    if not filtered_methods:
                        continue
            else:
                filtered_methods = methods_after_exclude

            filtered_entries.append((cls, filtered_methods))
        class_entries = filtered_entries

    if include_selectors:
        funcs = [f for f in funcs if _match_any(f, include_selectors)]
    if exclude_selectors:
        funcs = [f for f in funcs if not _match_any(f, exclude_selectors)]

    # ---- ordering
    if class_order:
        def _ckey(c):
            cls, _ = c
            try:
                return (0, class_order.index(cls.__name__))
            except ValueError:
                return (1, cls.__name__)
        class_entries.sort(key=_ckey)
    else:
        try:
            class_entries.sort(key=lambda c: c[0].__code__.co_firstlineno)  # type: ignore
        except Exception:
            class_entries.sort(key=lambda c: c[0].__name__)
    try: funcs.sort(key=lambda f: f.__code__.co_firstlineno)  # type: ignore
    except Exception: funcs.sort(key=lambda f: f.__name__)

    # ---- render
    for cls, methods in class_entries:
        lines.append("")
        lines.append(_render_class_block(cls, methods=methods))
        lines.append("")

    for fn in funcs:
        lines.append("")
        lines.append(_render_function_block(fn))
        lines.append("")

    # Build the final string
    # TODO: hacky fix now because I want to selectively include examples for
    # two different versions of the API prompt without rewriting everything
    # classes_to_include_examples = classes
    # if exclude_examples_for_classes:
    #     classes_to_include_examples = [
    #         c for c in classes if c.__name__ not in exclude_examples_for_classes
    #     ]
    # examples_text = _collect_examples_for_classes(classes_to_include_examples)
    examples_text = ""
    if example_files:
        example_chunks: List[str] = []
        for ef in example_files:
            if osp.isfile(ef):
                with open(ef, "r") as f:
                    text = f.readlines()
                text = [line.strip() for line in text if not line.strip().startswith("#")]
                text = "\n".join(text)
                if text.strip():
                    example_chunks.append(textwrap.dedent(text).strip())
        examples_text = "\n>>>\n>>>\n".join(example_chunks) if len(example_chunks) > 0 else ""
    if not any("{examples}" in s for s in lines) and examples_text:
        lines += [
            "\n",
            "# Examples of how to use the API",
            "--------------------------------",
            "\n"
            "{examples}",
        ]

    # Build the final string
    prompt_text = "\n".join(lines).rstrip()
    
    # Replace the {examples} placeholder with concatenated class examples (if any)
    if examples_text:
        prompt_text = prompt_text.replace("{examples}", examples_text)
    # If no examples available, leave the placeholder as-is

    return prompt_text

if __name__ == "__main__":
    
    from src.tva.media import (
        image_patch, video_segment
    )
    # import ipdb; ipdb.set_trace()
    with open(osp.join(
        osp.dirname(osp.abspath(__file__)), "api_prompt.v1.txt"
    ), "w") as f:
        f.write(generate_api_prompt(
            modules=[image_patch, video_segment],
            include_imports=(
                "import pandas as pd",
                "import numpy as np", 
                "from PIL import Image", 
                "from typing import Dict, Union, List, Optional"
            ),
            class_order=["ImagePatch", "VideoSegment"],
            include_classes=True,
            include_functions=False,
            exclude=[
                "ImagePatch.track_and_find_all_connected_pairs",
                "ImagePatch.check_part_connectivity",
            ],
            example_files=[osp.join(
                osp.dirname(osp.abspath(__file__)), "api_examples.v1.txt"
            )],
        ))

    with open(osp.join(
        osp.dirname(osp.abspath(__file__)), "api_prompt.v2.txt"
    ), "w") as f:
        f.write(generate_api_prompt(
            modules=[image_patch, video_segment],
            include_imports=(
                "import pandas as pd",
                "import numpy as np", 
                "from PIL import Image", 
                "from typing import Dict, Union, List, Optional"
            ),
            class_order=["ImagePatch", "VideoSegment"],
            include_classes=True,
            include_functions=False,
            exclude=[
                "VideoSegment.track_object_segments_in_video",
                "ImagePatch.check_part_connectivity",
            ],
            example_files=[osp.join(
                osp.dirname(osp.abspath(__file__)), "api_examples.v2.txt"
            )],
        ))
