"""CSV helpers shared by the service adapters.

CSV exports differ in delimiter, column naming, and whether they carry a byte-order
mark, so those concerns are handled once here rather than in each adapter.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path

from sine.ingestion.records import RawRecord

_DELIMITERS = (",", ";", "\t", "|")


def sniff_delimiter(text: str) -> str:
    """Guess the delimiter from the first non-empty line."""

    line = next((line for line in text.splitlines() if line.strip()), "")
    return max(_DELIMITERS, key=line.count)


def read_rows(path: Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a CSV export into rows with stripped keys and values.

    ``utf-8-sig`` transparently drops a byte-order mark, which several services add
    and which otherwise corrupts the first column name.
    """

    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(
        text.splitlines(), delimiter=delimiter or sniff_delimiter(text)
    )
    rows: list[dict[str, str]] = []
    for raw in reader:
        row: dict[str, str] = {}
        for key, value in raw.items():
            if key is None:
                continue
            # Services suffix duplicate names with "#name"; keep the first.
            name = key.strip().lstrip("\ufeff").split("#", 1)[0].strip()
            if not name:
                continue
            cleaned = (value or "").strip()
            if name not in row:
                row[name] = cleaned
            # Exports disagree on header casing, so also index case-folded. A
            # source that reads "Song Name" by its exact name is unaffected.
            folded = name.casefold()
            if folded not in row:
                row[folded] = cleaned
        rows.append(row)
    return rows


def column(
    row: Mapping[str, str], names: tuple[str, ...], *aliases: tuple[str, ...]
) -> str | None:
    """Return the first non-empty value among candidate column names.

    Column names vary between export generations of the same service, so adapters
    pass every spelling they have seen and let the data decide.
    """

    for name in (*names, *aliases):
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return None


def raw_records(
    rows: list[Mapping[str, object]], *, source: str, path: Path, start_line: int = 2
) -> Iterator[RawRecord]:
    """Wrap positional rows as :class:`RawRecord`, tracking a 1-based line number."""

    for offset, row in enumerate(rows):
        yield RawRecord(
            source=source,
            index=offset,
            line_number=start_line + offset,
            fields=dict(row),
            source_path=path,
        )
