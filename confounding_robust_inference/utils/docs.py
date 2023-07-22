import types
from typing import Any, Callable


def find_methods(cls: Any) -> list[Callable]:
    cls_attrs = [getattr(cls, a) for a in dir(cls)]
    cls_methods = [a for a in cls_attrs if isinstance(a, types.FunctionType)]
    parent_methods = []
    for p in cls.__bases__:
        p_attrs = [getattr(p, a) for a in dir(p)]
        p_methods = [a for a in p_attrs if isinstance(a, types.FunctionType)]
        parent_methods.extend(p_methods)
    return list(set(cls_methods) - set(parent_methods))


def find_method_names(cls: Any) -> list[str]:
    return [m.__name__ for m in find_methods(cls)]


def find_parent_classes_with_method_docstring(cls: Any, method_name: str) -> list[Any]:
    found_parents = []
    for p in cls.__bases__:
        if hasattr(p, method_name):
            parent_method = getattr(p, method_name)
            if hasattr(parent_method, "__doc__"):
                found_parents.append(p)
    return found_parents


def _populate_docstrings(cls: Any) -> None:
    """Populate missing docstring of the methods if it is defined in the parent class."""
    # print(f"Found methods for {cls.__name__}: {find_method_names(cls)}")
    for method_name in find_method_names(cls):
        if method_name.startswith("_"):  # skip private methods
            continue
        method = getattr(cls, method_name)
        if method.__doc__:  # no need to update docstring
            continue
        # print(f"Method {cls.__name__}.{method_name} is considered...")
        parent_cls = find_parent_classes_with_method_docstring(cls, method_name)
        if parent_cls:
            assert len(parent_cls) == 1
            c = parent_cls[0].__name__
            method.__doc__ = f"See :func:`{c}.{method_name}`"
            # print(f"Set {cls.__name__}.{method_name}.__doc__ as \"\"\"{method.__doc__}\"\"\".")
