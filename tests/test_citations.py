"""Tests for the citation database and its provenance guarantees."""

from __future__ import annotations

import sqlite3

import pytest

from medication_app.cite import CLAIMS, RETRIEVALS, SOURCES, seed
from medication_app.citations import (
    Source,
    add_claim,
    add_retrieval,
    add_source,
    connect,
    export_markdown,
    stats,
    uncited_claims,
)


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "citations.db")
    yield connection
    connection.close()


def test_source_roundtrip(db):
    add_source(db, Source(key="x", type="article", authors="A. Author", title="A title", year="2020"))
    row = db.execute("SELECT * FROM sources WHERE key = 'x'").fetchone()
    assert row["authors"] == "A. Author"
    assert row["year"] == "2020"


def test_source_insert_is_idempotent(db):
    for title in ("First", "Second"):
        add_source(db, Source(key="x", type="article", authors="A", title=title))
    rows = db.execute("SELECT title FROM sources WHERE key = 'x'").fetchall()
    assert len(rows) == 1
    assert rows[0]["title"] == "Second"


def test_claim_requires_an_existing_source(db):
    """A claim cannot cite a source that is not in the database."""
    with pytest.raises(sqlite3.IntegrityError):
        add_claim(db, "unfounded claim", "does-not-exist")


def test_no_orphaned_claims(db):
    add_source(db, Source(key="x", type="article", authors="A", title="T"))
    add_claim(db, "a claim", "x", section="§1")
    assert uncited_claims(db) == []


def test_retrieval_is_deduplicated(db):
    add_source(db, Source(key="x", type="article", authors="A", title="T"))
    for _ in range(3):
        add_retrieval(db, "x", via="vault", locator="§1", retrieved_on="2026-09-18")
    assert stats(db)["retrievals"] == 1


def test_repeated_consultation_is_recorded_separately(db):
    add_source(db, Source(key="x", type="article", authors="A", title="T"))
    add_retrieval(db, "x", via="vault", locator="§1", retrieved_on="2026-09-18")
    add_retrieval(db, "x", via="vault", locator="§2", retrieved_on="2026-09-18")
    assert stats(db)["retrievals"] == 2


def test_export_markdown_lists_every_source(db, tmp_path):
    add_source(db, Source(key="alpha", type="book", authors="A", title="T1"))
    add_source(db, Source(key="beta", type="article", authors="B", title="T2"))
    add_claim(db, "claim", "alpha", section="§3", implemented_in="mod.py")
    path = export_markdown(db, tmp_path / "REFERENCES.md")
    text = path.read_text(encoding="utf-8")
    assert "## alpha" in text and "## beta" in text
    assert "§3" in text and "mod.py" in text


# --------------------------------------------------------------------------- #
# The seeded research trail itself
# --------------------------------------------------------------------------- #


def test_seed_data_is_internally_consistent():
    """Every retrieval and claim must point at a source that exists."""
    keys = {source.key for source in SOURCES}
    assert {key for key, *_ in RETRIEVALS} <= keys
    assert {claim[1] for claim in CLAIMS} <= keys


def test_source_keys_are_unique():
    keys = [source.key for source in SOURCES]
    assert len(keys) == len(set(keys))


def test_every_source_has_a_recorded_retrieval():
    """A source in the database must record how it was obtained."""
    retrieved = {key for key, *_ in RETRIEVALS}
    assert {source.key for source in SOURCES} <= retrieved


def test_claims_that_drive_code_name_where_they_are_implemented():
    for claim, _source, _section, implemented_in, _verified in CLAIMS:
        assert implemented_in, f"claim has no implementation reference: {claim[:50]}"
