"""File fragments are byte identities independent of load addresses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fragment:
    """A named, immutable byte fragment.

    The fragment id remains stable if the same bytes are later mapped into
    more than one address space or referenced by future data-reference
    analysis.
    """

    fragment_id: str
    name: str
    data: bytes
    source: str = ""

    def __post_init__(self) -> None:
        if not self.fragment_id:
            raise ValueError("fragment_id must not be empty")
        if not isinstance(self.data, bytes | bytearray | memoryview):
            raise TypeError("fragment data must be bytes-like")
        object.__setattr__(self, "data", bytes(self.data))

    def __len__(self) -> int:
        return len(self.data)
