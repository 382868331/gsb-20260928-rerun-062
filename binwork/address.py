"""Address spaces map virtual addresses onto immutable file fragments."""

from __future__ import annotations

from dataclasses import dataclass

from .fragments import Fragment


@dataclass(frozen=True, slots=True)
class SourceLocation:
    address: int
    fragment_id: str
    fragment_offset: int


@dataclass(frozen=True, slots=True)
class FragmentMapping:
    mapping_id: str
    fragment: Fragment
    load_address: int
    fragment_offset: int = 0
    size: int | None = None

    def __post_init__(self) -> None:
        if self.load_address < 0:
            raise ValueError("load_address must be non-negative")
        if self.fragment_offset < 0 or self.fragment_offset > len(self.fragment):
            raise ValueError("fragment_offset is outside the fragment")
        size = len(self.fragment) - self.fragment_offset if self.size is None else self.size
        if size < 0 or self.fragment_offset + size > len(self.fragment):
            raise ValueError("mapping range is outside the fragment")
        object.__setattr__(self, "size", size)

    @property
    def end_address(self) -> int:
        return self.load_address + self.size  # type: ignore[operator]

    def contains(self, address: int) -> bool:
        return self.load_address <= address < self.end_address


@dataclass(frozen=True, slots=True)
class ReadResult:
    data: bytes
    locations: tuple[SourceLocation, ...]
    missing_from: int | None = None

    @property
    def complete(self) -> bool:
        return self.missing_from is None


class AddressSpace:
    """A collection of fragment mappings.

    Mappings are not required to be contiguous. Earlier mappings have
    precedence when mapped ranges overlap, which makes ambiguous provenance
    explicit through the returned source locations rather than hiding it.
    """

    def __init__(self, mappings: list[FragmentMapping] | None = None) -> None:
        self._mappings: list[FragmentMapping] = list(mappings or [])

    def add_mapping(self, mapping: FragmentMapping) -> None:
        self._mappings.append(mapping)

    @property
    def mappings(self) -> tuple[FragmentMapping, ...]:
        return tuple(self._mappings)

    def mapping_at(self, address: int) -> FragmentMapping | None:
        for mapping in self._mappings:
            if mapping.contains(address):
                return mapping
        return None

    def location_at(self, address: int) -> SourceLocation | None:
        mapping = self.mapping_at(address)
        if mapping is None:
            return None
        return SourceLocation(
            address=address,
            fragment_id=mapping.fragment.fragment_id,
            fragment_offset=address - mapping.load_address + mapping.fragment_offset,
        )

    def read(self, address: int, size: int) -> ReadResult:
        if address < 0 or size < 0:
            raise ValueError("address and size must be non-negative")
        out = bytearray()
        locations: list[SourceLocation] = []
        missing_from: int | None = None
        for current in range(address, address + size):
            location = self.location_at(current)
            if location is None:
                missing_from = current
                break
            mapping = self.mapping_at(current)
            assert mapping is not None
            out.append(mapping.fragment.data[location.fragment_offset])
            locations.append(location)
        return ReadResult(bytes(out), tuple(locations), missing_from)
