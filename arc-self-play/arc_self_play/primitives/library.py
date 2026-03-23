"""Primitive library: a named registry of grid transformation operations."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import operations

Grid = List[List[int]]


class _Primitive:
    """A named, callable grid transformation."""

    __slots__ = ("name", "category", "fn")

    def __init__(self, name: str, category: str, fn: Callable):
        self.name = name
        self.category = category
        self.fn = fn

    def execute(self, grid: Grid, *args: Any, **kwargs: Any) -> Grid:
        return self.fn(grid, *args, **kwargs)


class PrimitiveLibrary:
    """Central registry of grid transformation primitives.

    >>> lib = PrimitiveLibrary()
    >>> len(lib)
    9
    >>> lib.get("rotate_90").execute([[1, 2], [3, 4]])
    [[3, 1], [4, 2]]
    """

    def __init__(self) -> None:
        self._primitives: Dict[str, _Primitive] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        geometric = [
            ("rotate_90",           operations.rotate_90),
            ("rotate_180",          operations.rotate_180),
            ("rotate_270",          operations.rotate_270),
            ("reflect_horizontal",  operations.reflect_horizontal),
            ("reflect_vertical",    operations.reflect_vertical),
            ("reflect_diagonal",    operations.reflect_diagonal),
        ]
        spatial = [
            ("auto_crop", operations.auto_crop),
            ("tile",      operations.tile),
        ]
        color = [
            ("recolor", operations.recolor),
        ]

        for name, fn in geometric:
            self._register(name, "geometric", fn)
        for name, fn in spatial:
            self._register(name, "spatial", fn)
        for name, fn in color:
            self._register(name, "color", fn)

    def _register(self, name: str, category: str, fn: Callable) -> None:
        self._primitives[name] = _Primitive(name, category, fn)

    def get(self, name: str) -> Optional[_Primitive]:
        return self._primitives.get(name)

    def get_all_names(self) -> List[str]:
        return list(self._primitives.keys())

    def get_by_category(self, category: str) -> List[_Primitive]:
        return [p for p in self._primitives.values() if p.category == category]

    def __len__(self) -> int:
        return len(self._primitives)

    def __contains__(self, name: str) -> bool:
        return name in self._primitives
