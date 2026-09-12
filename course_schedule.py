"""
Emploi du temps de l'utilisateur, stocké dans un fichier YAML facilement
éditable à la main. Permet de pré-remplir automatiquement le titre (et
des métadonnées libres comme la salle) du cours en fonction du jour et
de l'heure actuels.

Format du fichier (voir DEFAULT_SCHEDULE_CONTENT ci-dessous) :

lundi:
  - debut: "08:00"
    fin: "10:00"
    titre: "Histoire - Révolution industrielle"
    salle: "Amphi A"

Les clés autres que "debut", "fin" et "titre" sont libres (salle,
professeur, etc.) et affichées telles quelles dans l'interface.
"""

from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path.home() / "Echo"
SCHEDULE_PATH = CONFIG_DIR / "emploi_du_temps.yaml"

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_JOURS = JOURS  # alias interne

_HEADER_COMMENT = (
    "# Emploi du temps géré depuis l'application Echo (bouton \"EMPLOI DU TEMPS\").\n"
    "# Ce fichier reste modifiable à la main si besoin (format HH:MM pour les horaires).\n"
)

DEFAULT_SCHEDULE_CONTENT = _HEADER_COMMENT + "\n" + "\n".join(f"{jour}: []" for jour in JOURS) + "\n"


def ensure_schedule_file() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SCHEDULE_PATH.exists():
        SCHEDULE_PATH.write_text(DEFAULT_SCHEDULE_CONTENT, encoding="utf-8")
    return SCHEDULE_PATH


def load_schedule() -> dict:
    ensure_schedule_file()
    try:
        with SCHEDULE_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}
    return data


def save_schedule(data: dict) -> None:
    """Réécrit le fichier d'emploi du temps avec les données données (utilisé
    par l'éditeur intégré à l'application)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    SCHEDULE_PATH.write_text(_HEADER_COMMENT + "\n" + body, encoding="utf-8")


def _parse_time(value: str) -> datetime.time:
    hours, minutes = value.split(":")
    return datetime.time(int(hours), int(minutes))


def get_current_entry(now: datetime.datetime | None = None) -> dict | None:
    """
    Retourne le créneau correspondant au jour/heure donnés (ou maintenant
    par défaut), ou None si aucun cours n'est prévu à ce moment.
    """
    now = now or datetime.datetime.now()
    schedule = load_schedule()
    jour = _JOURS[now.weekday()]
    creneaux = schedule.get(jour) or []

    for creneau in creneaux:
        try:
            debut = _parse_time(str(creneau["debut"]))
            fin = _parse_time(str(creneau["fin"]))
        except (KeyError, ValueError):
            continue
        if debut <= now.time() <= fin:
            return creneau
    return None


def open_schedule_file_in_editor():
    """Ouvre le fichier dans l'éditeur de texte par défaut du système."""
    path = ensure_schedule_file()
    system = platform.system()
    if system == "Windows":
        import os

        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)