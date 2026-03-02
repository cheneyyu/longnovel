"""SQLite + JSON-backed world/character database utilities."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import JSON_DB_PATH, SQLITE_DB_PATH, ensure_project_dirs


@dataclass(slots=True)
class WorldMapEntry:
    original_term: str
    xianxia_term: str


@dataclass(slots=True)
class CharacterEntry:
    original_name: str
    xianxia_name: str
    sect: str
    cultivation_level: str
    status: str = "Alive"


class DatabaseManager:
    """Manages local SQLite storage and JSON snapshots for quick inspection."""

    def __init__(self, db_path: Path = SQLITE_DB_PATH, json_path: Path = JSON_DB_PATH):
        ensure_project_dirs()
        self.db_path = Path(db_path)
        self.json_path = Path(json_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS WorldMap (
                    original_term TEXT PRIMARY KEY,
                    xianxia_term TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS Characters (
                    original_name TEXT PRIMARY KEY,
                    xianxia_name TEXT NOT NULL,
                    sect TEXT NOT NULL,
                    cultivation_level TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('Alive', 'Dead'))
                );
                """
            )

    def seed_mock_data(self) -> None:
        """Insert initial Monte Cristo -> Xianxia mock data."""
        world_map_rows: list[WorldMapEntry] = [
            WorldMapEntry("Prison", "Demon Abyss"),
            WorldMapEntry("Chateau d'If", "Heaven-Sealed Black Reef"),
            WorldMapEntry("Marseilles", "Azure Tide City"),
            WorldMapEntry("Count", "Venerable Lord"),
            WorldMapEntry("Duel", "Dao Challenge"),
            WorldMapEntry("Court", "Imperial Tribunal Hall"),
            WorldMapEntry("Treasure", "Spirit Vein Legacy"),
            WorldMapEntry("Ship", "Cloud-Sailing Spirit Ark"),
        ]
        character_rows: list[CharacterEntry] = [
            CharacterEntry("Edmond Dantes", "Mo Tiansheng", "No Sect (Rogue Cultivator)", "Mortal Realm", "Alive"),
            CharacterEntry("Mercedes", "Mei Lian", "Lotus Heart Pavilion", "Foundation Establishment", "Alive"),
            CharacterEntry("Fernand Mondego", "Fan Mie", "Ironblood War Hall", "Core Formation", "Alive"),
            CharacterEntry("Danglars", "Dan Ge", "Golden Abacus Clan", "Foundation Establishment", "Alive"),
            CharacterEntry("Villefort", "Wei Fa", "Lawkeeper Palace", "Core Formation", "Alive"),
            CharacterEntry("Abbe Faria", "Master Fa Li", "Hidden Scripture Peak", "Nascent Soul", "Alive"),
            CharacterEntry("Caderousse", "Ka De", "Dust Market Brotherhood", "Qi Condensation", "Alive"),
            CharacterEntry("Haydee", "Hai Die", "Moon Pearl Isle", "Core Formation", "Alive"),
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO WorldMap (original_term, xianxia_term)
                VALUES (?, ?)
                ON CONFLICT(original_term) DO UPDATE
                SET xianxia_term = excluded.xianxia_term
                """,
                [(row.original_term, row.xianxia_term) for row in world_map_rows],
            )
            conn.executemany(
                """
                INSERT INTO Characters (original_name, xianxia_name, sect, cultivation_level, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(original_name) DO UPDATE
                SET
                    xianxia_name = excluded.xianxia_name,
                    sect = excluded.sect,
                    cultivation_level = excluded.cultivation_level,
                    status = excluded.status
                """,
                [
                    (
                        row.original_name,
                        row.xianxia_name,
                        row.sect,
                        row.cultivation_level,
                        row.status,
                    )
                    for row in character_rows
                ],
            )

    def export_to_json(self) -> dict[str, list[dict[str, str]]]:
        payload = {
            "WorldMap": self.fetch_world_map(),
            "Characters": self.fetch_characters(),
        }
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def fetch_world_map(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT original_term, xianxia_term FROM WorldMap ORDER BY original_term"
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_characters(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT original_name, xianxia_name, sect, cultivation_level, status
                FROM Characters
                ORDER BY original_name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_characters_by_original_names(self, names: Iterable[str]) -> list[dict[str, str]]:
        unique_names = sorted({name.strip() for name in names if name and name.strip()})
        if not unique_names:
            return []

        placeholders = ", ".join("?" for _ in unique_names)
        query = (
            "SELECT original_name, xianxia_name, sect, cultivation_level, status "
            f"FROM Characters WHERE original_name IN ({placeholders}) ORDER BY original_name"
        )
        with self._connect() as conn:
            rows = conn.execute(query, unique_names).fetchall()
        return [dict(row) for row in rows]


def bootstrap_database() -> DatabaseManager:
    """Initialize schema + seed data + JSON snapshot."""
    db = DatabaseManager()
    db.initialize_schema()
    db.seed_mock_data()
    db.export_to_json()
    return db


if __name__ == "__main__":
    manager = bootstrap_database()
    print(f"Database initialized at: {manager.db_path}")
    print(f"JSON snapshot written to: {manager.json_path}")
