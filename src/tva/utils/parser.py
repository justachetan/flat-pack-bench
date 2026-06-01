from typing import Callable, Dict, Optional
import ast
import inspect
from functools import wraps

from contextlib import contextmanager

# ---- Context manager to suppress overrides ----
@contextmanager
def no_overrides(owner):
    """
    Temporarily disable @override_from_method_params for this instance or class.
    Re-entrant and exception-safe.
    Usage: with no_overrides(self): self.fn_a(...)
    """
    attr = "_no_method_param_overrides"
    count = getattr(owner, attr, 0)
    setattr(owner, attr, count + 1)
    try:
        yield
    finally:
        new_count = getattr(owner, attr, 1) - 1
        if new_count <= 0:
            # Clean up to avoid leaving junk state around
            try:
                delattr(owner, attr)
            except Exception:
                setattr(owner, attr, 0)
        else:
            setattr(owner, attr, new_count)

# ---- The decorator ----
def override_from_method_params(func):
    """
    If the instance/class defines:
        method_params = {'method_name': {'arg': value, ...}}
    then those values override call-time args—unless no_overrides(...) is active.
    """
    sig = inspect.signature(func)
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    name = func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args:
            return func(*args, **kwargs)

        owner = args[0]  # self or cls

        # 1) Respect the suppression flag (works for instance or class)
        if getattr(owner, "_no_method_param_overrides", 0) > 0:
            return func(*args, **kwargs)

        # 2) Pull overrides (instance first, then class)
        overrides = {}
        owner_cls = owner if isinstance(owner, type) else type(owner)
        for obj in (owner, owner_cls):
            mp = getattr(obj, "method_params", None)
            if isinstance(mp, dict) and name in mp and isinstance(mp[name], dict):
                overrides = mp[name]
                break

        if not overrides:
            return func(*args, **kwargs)

        # 3) Bind and overwrite by name
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        for k, v in overrides.items():
            if k in sig.parameters:
                bound.arguments[k] = v
            else:
                if has_var_kw:
                    kw_name = next(p.name for p in sig.parameters.values()
                                   if p.kind == p.VAR_KEYWORD)
                    bound.arguments.setdefault(kw_name, {})
                    bound.arguments[kw_name][k] = v
                else:
                    raise TypeError(
                        f"Override specifies unknown parameter '{k}' for {name} "
                        "and the method has no **kwargs."
                    )

        return func(*bound.args, **bound.kwargs)

    return wrapper

# @contextmanager
# def no_overrides(self):
#     setattr(self, "_no_method_param_overrides", True)
#     try:
#         yield
#     finally:
#         setattr(self, "_no_method_param_overrides", False)

# def override_from_method_params(func: Callable) -> Callable:
#     """
#     If the owning object (or its class) defines `method_params` like:
#         {
#           'method_name': {'arg1': val, 'arg2': val, ...},
#           ...
#         }
#     then those entries override the method's call-time args/kwargs.
#     """
#     sig = inspect.signature(func)
#     has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
#     name = func.__name__

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         if not args:
#             # plain function, nothing to do
#             return func(*args, **kwargs)

#         owner = args[0]  # usually `self` (or `cls` for classmethods)
#         # Look on instance first, then class
#         overrides = {}
#         for obj in (owner, type(owner)):
#             mp = getattr(obj, "method_params", None)
#             if isinstance(mp, dict) and name in mp and isinstance(mp[name], dict):
#                 overrides = mp[name]
#                 break

#         if not overrides:
#             return func(*args, **kwargs)

#         # Bind given args so we can safely overwrite by name
#         bound = sig.bind_partial(*args, **kwargs)
#         bound.apply_defaults()

#         for k, v in overrides.items():
#             if k in sig.parameters:
#                 bound.arguments[k] = v
#             else:
#                 # Not a declared parameter name; only allow if **kwargs exists
#                 if has_var_kw:
#                     # put into kwargs bucket; create if missing
#                     # (bind() materializes **kwargs as its declared name)
#                     kw_name = next(p.name for p in sig.parameters.values()
#                                    if p.kind == p.VAR_KEYWORD)
#                     bound.arguments.setdefault(kw_name, {})
#                     bound.arguments[kw_name][k] = v
#                 else:
#                     raise TypeError(
#                         f"Override specifies unknown parameter '{k}' for {name} "
#                         "and the method has no **kwargs."
#                     )

#         return func(*bound.args, **bound.kwargs)

#     return wrapper

class _ConstructorRewriter(ast.NodeTransformer):
    """
    Replace constructor calls to target classes with provided callable names (e.g., partial constructors).
    Examples:
      - ImagePatch(...)           -> image_patch_ctor(...)
      - media.ImagePatch(...)     -> image_patch_ctor(...)
      - pkg.media.ImagePatch(...) -> image_patch_ctor(...)
      - from x import ImagePatch as IP; IP(...) -> image_patch_ctor(...)
    """

    def __init__(self, target_map: Dict[str, str]):
        """
        Args:
            target_map: maps class name -> replacement callable name
                e.g. {"ImagePatch": "image_patch_ctor", "VideoSegment": "video_segment_ctor"}
        """
        super().__init__()
        self.target_map = dict(target_map)
        # Names in current file that are aliases for a target class (e.g., IP -> ImagePatch)
        self.alias_to_target: Dict[str, str] = {}

    # --- Import/alias collection (best-effort) --------------------------------
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        # from pkg import ImagePatch as IP
        for alias in node.names:
            original = alias.name
            asname = alias.asname or alias.name
            if original in self.target_map:
                self.alias_to_target[asname] = original
        return self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        # For "import pkg as m", we don't know until Attribute access;
        # we handle Attribute.attr == target class in visit_Call.
        return self.generic_visit(node)

    # --- Core rewriting logic -------------------------------------------------
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)  # first, visit args/kwargs

        # Case 1: direct name call:  ImagePatch(...)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            # Is this name an alias for a target? (e.g., IP for ImagePatch)
            if name in self.alias_to_target:
                target = self.alias_to_target[name]
                repl = self.target_map.get(target)
                if repl:
                    node.func = ast.Name(id=repl, ctx=ast.Load())
                    return node
            # Or is it directly the target?
            if name in self.target_map:
                node.func = ast.Name(id=self.target_map[name], ctx=ast.Load())
                return node

        # Case 2: attribute call: media.ImagePatch(...), pkg.media.ImagePatch(...)
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in self.target_map:
                # Replace the whole callee with the replacement name
                node.func = ast.Name(id=self.target_map[attr], ctx=ast.Load())
                return node

        return node


def rewrite_constructors_to_partials(
    code_str: str,
    *,
    replacements: Dict[str, str] = None,
) -> str:
    """
    Rewrite constructor calls in `code_str` to use replacement callables.
    Args:
        code_str: Python source to transform.
        replacements: mapping { "ImagePatch": "image_patch_ctor",
                               "VideoSegment": "video_segment_ctor" }
    Returns:
        Transformed Python source (string).
    """
    if not replacements:
        return code_str

    tree = ast.parse(code_str)
    tree = _ConstructorRewriter(replacements).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)