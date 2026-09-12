"""
Gestion des sessions de prise de notes : un fichier Markdown par session,
avec un titre, la date, et les segments de transcription horodatés.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

NOTES_DIR = Path.home() / "LectureNotes"


def ensure_notes_dir() -> Path:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return NOTES_DIR


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "session"


class Session:
    """Représente une session de cours en cours d'enregistrement."""

    def __init__(self, title: str):
        ensure_notes_dir()
        self.title = title
        self.started_at = datetime.datetime.now()
        filename = f"{self.started_at.strftime('%Y-%m-%d_%H-%M')}_{_slugify(title)}.md"
        self.path = NOTES_DIR / filename

        with self.path.open("w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"_Session du {self.started_at.strftime('%d/%m/%Y à %H:%M')}_\n\n")

    def append_segment(self, text: str, segment_started_at: float):
        """Ajoute un segment transcrit, horodaté par rapport au début de session."""
        elapsed = segment_started_at - self.started_at.timestamp()
        minutes, seconds = divmod(max(0, int(elapsed)), 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"

        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {text}\n\n")

    def close(self):
        with self.path.open("a", encoding="utf-8") as f:
            ended_at = datetime.datetime.now()
            duration = ended_at - self.started_at
            minutes = int(duration.total_seconds() // 60)
            f.write(f"\n---\n_Session terminée — durée : {minutes} min_\n")


class SessionSummary:
    """Métadonnées légères d'une session passée, pour affichage en liste."""

    def __init__(self, path: Path):
        self.path = path
        self.title = path.stem
        self.date_str = ""
        self._load_header()

    def _load_header(self):
        try:
            with self.path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
        except OSError:
            return
        if first_line.startswith("# "):
            self.title = first_line[2:].strip()
        if second_line.startswith("_") and second_line.endswith("_"):
            self.date_str = second_line.strip("_")


def list_sessions() -> list[SessionSummary]:
    """Liste les sessions passées, triées de la plus récente à la plus ancienne."""
    ensure_notes_dir()
    files = sorted(NOTES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [SessionSummary(p) for p in files]


def read_session_content(path: Path) -> str:
    return path.read_text(encoding="utf-8")