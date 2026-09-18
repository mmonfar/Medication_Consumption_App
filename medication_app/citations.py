"""Citation and provenance tracking.

Every methodological choice in this project traces to a source. This module is
where that trail lives, so a reviewer can ask "why is Meropenem scored on
RMSSE?" and get a publication, a section, a retrieval date, and the code that
implements it -- rather than a plausible-sounding paragraph.

SQLite rather than a Markdown file, because the questions asked of this are
relational: which claims rest on one source, which code has no citation, what
was retrieved from where and when. ``REFERENCES.md`` is generated from the
database for reading; the database stays the source of truth. Regenerate with:

    python -m medication_app.cite export

Retrievals are recorded separately from sources: a source is a publication, a
retrieval is one act of consulting it (which vault, which query, which day),
so repeated consultation of the same work does not overwrite its history.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "references" / "citations.db"
MARKDOWN_PATH = Path(__file__).resolve().parent.parent / "REFERENCES.md"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    key            TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    authors        TEXT NOT NULL,
    year           TEXT,
    title          TEXT NOT NULL,
    container      TEXT,
    edition        TEXT,
    publisher      TEXT,
    volume         TEXT,
    pages          TEXT,
    doi            TEXT,
    url            TEXT,
    licence        TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS retrievals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key     TEXT NOT NULL REFERENCES sources(key) ON DELETE CASCADE,
    retrieved_on   TEXT NOT NULL,
    via            TEXT NOT NULL,
    locator        TEXT,
    query          TEXT,
    UNIQUE(source_key, retrieved_on, via, locator)
);

CREATE TABLE IF NOT EXISTS claims (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    claim          TEXT NOT NULL,
    source_key     TEXT NOT NULL REFERENCES sources(key) ON DELETE CASCADE,
    section        TEXT,
    implemented_in TEXT,
    verified       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(claim, source_key, section)
);

CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_key);
CREATE INDEX IF NOT EXISTS idx_retrievals_source ON retrievals(source_key);
"""


@dataclass(frozen=True)
class Source:
    key: str
    type: str
    authors: str
    title: str
    year: str = ""
    container: str = ""
    edition: str = ""
    publisher: str = ""
    volume: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    licence: str = ""
    notes: str = ""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def add_source(connection: sqlite3.Connection, source: Source) -> None:
    connection.execute(
        """
        INSERT INTO sources (key, type, authors, year, title, container, edition,
                             publisher, volume, pages, doi, url, licence, notes)
        VALUES (:key, :type, :authors, :year, :title, :container, :edition,
                :publisher, :volume, :pages, :doi, :url, :licence, :notes)
        ON CONFLICT(key) DO UPDATE SET
            type=excluded.type, authors=excluded.authors, year=excluded.year,
            title=excluded.title, container=excluded.container,
            edition=excluded.edition, publisher=excluded.publisher,
            volume=excluded.volume, pages=excluded.pages, doi=excluded.doi,
            url=excluded.url, licence=excluded.licence, notes=excluded.notes
        """,
        source.__dict__,
    )


def add_retrieval(
    connection: sqlite3.Connection,
    source_key: str,
    via: str,
    locator: str = "",
    query: str = "",
    retrieved_on: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO retrievals (source_key, retrieved_on, via, locator, query)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_key, retrieved_on or date.today().isoformat(), via, locator, query),
    )


def add_claim(
    connection: sqlite3.Connection,
    claim: str,
    source_key: str,
    section: str = "",
    implemented_in: str = "",
    verified: bool = False,
) -> None:
    connection.execute(
        """
        INSERT INTO claims (claim, source_key, section, implemented_in, verified)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(claim, source_key, section) DO UPDATE SET
            implemented_in=excluded.implemented_in, verified=excluded.verified
        """,
        (claim, source_key, section, implemented_in, int(verified)),
    )


def format_reference(row: sqlite3.Row) -> str:
    """A plain author-date reference line."""
    parts = [f"{row['authors']}"]
    if row["year"]:
        parts.append(f"({row['year']})")
    parts.append(f"*{row['title']}*." if row["type"] == "book" else f"{row['title']}.")
    for field in ("container", "edition", "volume", "pages", "publisher"):
        if row[field]:
            value = row[field]
            parts.append(f"*{value}*," if field == "container" else f"{value}.")
    if row["doi"]:
        parts.append(f"https://doi.org/{row['doi']}")
    elif row["url"]:
        parts.append(row["url"])
    return " ".join(parts)


def export_markdown(connection: sqlite3.Connection, path: Path = MARKDOWN_PATH) -> Path:
    """Render the database to a human-readable bibliography."""
    sources = connection.execute("SELECT * FROM sources ORDER BY authors, year").fetchall()

    lines = [
        "# References",
        "",
        "Generated from `references/citations.db` by `python -m medication_app.cite export`.",
        "Do not edit by hand -- edit the database and regenerate.",
        "",
        f"{len(sources)} sources.",
        "",
    ]

    for source in sources:
        lines.append(f"## {source['key']}")
        lines.append("")
        lines.append(format_reference(source))
        if source["licence"]:
            lines.append("")
            lines.append(f"*Licence:* {source['licence']}")
        if source["notes"]:
            lines.append("")
            lines.append(source["notes"])

        retrievals = connection.execute(
            "SELECT * FROM retrievals WHERE source_key = ? ORDER BY retrieved_on",
            (source["key"],),
        ).fetchall()
        if retrievals:
            lines += ["", "**Retrieved**", ""]
            for retrieval in retrievals:
                locator = f" — {retrieval['locator']}" if retrieval["locator"] else ""
                query = f' (query: "{retrieval["query"]}")' if retrieval["query"] else ""
                lines.append(f"- {retrieval['retrieved_on']} via {retrieval['via']}{locator}{query}")

        claims = connection.execute(
            "SELECT * FROM claims WHERE source_key = ? ORDER BY id", (source["key"],)
        ).fetchall()
        if claims:
            lines += ["", "**Used for**", ""]
            for claim in claims:
                section = f" [{claim['section']}]" if claim["section"] else ""
                code = f" → `{claim['implemented_in']}`" if claim["implemented_in"] else ""
                mark = "verified against this project's data" if claim["verified"] else ""
                suffix = f" _({mark})_" if mark else ""
                lines.append(f"- {claim['claim']}{section}{code}{suffix}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def uncited_claims(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Claims whose source is missing -- a provenance gap."""
    return connection.execute(
        """
        SELECT c.* FROM claims c
        LEFT JOIN sources s ON s.key = c.source_key
        WHERE s.key IS NULL
        """
    ).fetchall()


def stats(connection: sqlite3.Connection) -> dict[str, int]:
    def count(table: str) -> int:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    return {
        "sources": count("sources"),
        "retrievals": count("retrievals"),
        "claims": count("claims"),
        "verified_claims": int(
            connection.execute("SELECT COUNT(*) FROM claims WHERE verified = 1").fetchone()[0]
        ),
    }
