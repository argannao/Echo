"""
Génération de notes de cours structurées à partir de la transcription
brute, et export en PDF — entièrement en local via un modèle GGUF chargé
directement dans le processus Python (llama-cpp-python), sans application
tierce (contrairement à Ollama).

Le modèle est un simple fichier .gguf téléchargé automatiquement (voir
ai_setup.py) dans ~/Echo/models/ au premier lancement.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_DIR = Path.home() / "Echo" / "models"

_SYSTEM_PROMPT = """Tu es un assistant qui reformate une transcription brute et parfois décousue d'un cours magistral en notes de cours claires et structurées, en français.

Règle absolue, la plus importante de toutes : tu n'as STRICTEMENT PAS le droit d'ajouter la moindre information, exemple, explication, définition ou phrase de liaison qui ne soit pas déjà présente dans la transcription fournie. Ton rôle est UNIQUEMENT de réorganiser et clarifier ce qui a été dit, jamais de compléter, expliquer, ou enrichir avec tes propres connaissances. Si la transcription est courte, décousue, ou ne contient pas assez de matière pour un "vrai cours", contente-toi de la reformuler fidèlement en gardant exactement le même niveau de contenu — n'invente surtout pas de contexte, de définitions ou d'exemples pour combler les manques.

Règles de formatage strictes (respecte-les à la lettre, n'utilise aucun autre symbole de mise en forme) :
- Une seule ligne commençant par "# " pour le titre général du cours
- Des lignes commençant par "## " pour chaque grande partie/thème réellement abordé dans la transcription (n'invente pas de sections qui ne correspondent à rien de dit)
- Des lignes commençant par "- " pour les points clés effectivement mentionnés
- Des paragraphes de texte normal (sans préfixe) uniquement pour reformuler ce qui a été dit, jamais pour ajouter du contenu
- Ignore les répétitions, hésitations et erreurs de transcription évidentes, mais ne remplace jamais une idée par une autre
"""

_USER_REMINDER = "\n\n(Rappel : n'ajoute rien qui ne soit pas déjà dans le texte ci-dessus.)"

_llm_instance = None  # chargé une seule fois, réutilisé ensuite (le chargement prend quelques secondes)


def get_model_path() -> Path | None:
    """Retourne le chemin du modèle s'il est déjà téléchargé, sinon None."""
    path = MODEL_DIR / MODEL_FILENAME
    return path if path.exists() else None


def ensure_model_downloaded(on_progress=None) -> Path:
    """
    Télécharge le modèle depuis Hugging Face si besoin (fichier unique,
    ~2 Go). Réutilise huggingface_hub, déjà présent comme dépendance de
    faster-whisper.
    """
    existing = get_model_path()
    if existing is not None:
        return existing

    from huggingface_hub import hf_hub_download

    if on_progress:
        on_progress("téléchargement en cours (peut prendre plusieurs minutes)...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(MODEL_DIR),
    )
    return Path(downloaded_path)


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from llama_cpp import Llama

        model_path = get_model_path()
        if model_path is None:
            raise RuntimeError(
                "Modèle IA local introuvable — lance la vérification IA dans l'app "
                "(puce en haut de la fenêtre) pour le télécharger."
            )
        _llm_instance = Llama(
            model_path=str(model_path),
            n_ctx=8192,
            n_threads=os.cpu_count() or 4,
            chat_format="llama-3",
            verbose=False,
        )
    return _llm_instance


def generate_structured_notes(raw_transcript: str) -> str:
    """
    Envoie la transcription brute au modèle local et retourne des notes
    structurées en français, en Markdown (sous-ensemble simple : #, ##, -).
    """
    try:
        llm = _get_llm()
    except ImportError as exc:
        raise RuntimeError(
            "Le paquet Python 'llama-cpp-python' n'est pas installé (uv add llama-cpp-python)."
        ) from exc

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_transcript + _USER_REMINDER},
            ],
            max_tokens=3000,
            temperature=0.1,
        )
    except Exception as exc:
        raise RuntimeError(f"Erreur du modèle IA local : {exc}") from exc

    return result["choices"][0]["message"]["content"]


def generate_notes_document(raw_transcript: str, target_language: str | None = None) -> str:
    """
    Génère les notes structurées en français, et si une langue cible est
    fournie, ajoute une traduction complète en dessous (séparée par "---"),
    via le module de traduction dédié (NLLB — voir translation.py).
    """
    french_notes = generate_structured_notes(raw_transcript)

    if not target_language or target_language == "Aucune":
        return french_notes

    from translation import translate_markdown

    translated = translate_markdown(french_notes, target_language)
    return french_notes + "\n\n---\n\n" + translated


def open_containing_folder(path: Path) -> None:
    """
    Ouvre l'explorateur de fichiers du système sur le dossier contenant ce
    fichier, en le sélectionnant si possible (Windows/macOS ; sur Linux,
    ouvre simplement le dossier).
    """
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(["explorer", f"/select,{path}"])
        elif system == "Darwin":
            subprocess.run(["open", "-R", str(path)])
        else:
            subprocess.run(["xdg-open", str(path.parent)])
    except Exception:
        pass  # ouvrir le dossier est un confort, jamais bloquant


FONT_DIR = Path.home() / "Echo" / "fonts"
UNICODE_FONT_FILENAME = "NotoSansMyanmar-Regular.ttf"
UNICODE_FONT_URL = (
    "https://cdn.jsdelivr.net/npm/@fontsource/noto-sans-myanmar/files/"
    "noto-sans-myanmar-myanmar-400-normal.ttf"
)


def _ensure_unicode_font() -> Path | None:
    """
    Télécharge (une seule fois) une police couvrant à la fois le latin et
    l'écriture birmane (Noto Sans Myanmar), nécessaire pour que le PDF
    affiche correctement une traduction en birman plutôt que des "?".
    Retourne None si le téléchargement échoue (le PDF retombe alors sur la
    police de base, en Latin uniquement).
    """
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    font_path = FONT_DIR / UNICODE_FONT_FILENAME
    if font_path.exists():
        return font_path
    try:
        import urllib.request

        urllib.request.urlretrieve(UNICODE_FONT_URL, font_path)
        return font_path
    except Exception:
        return None


def _safe_text(text: str) -> str:
    """Les polices de base de fpdf2 ne supportent que le Latin-1 : on
    remplace les caractères non représentables plutôt que de planter."""
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_from_markdown(markdown_text: str, output_path: Path) -> None:
    """
    Convertit un texte structuré (sous-ensemble simple de Markdown : #, ##,
    -, paragraphes, "---" comme saut de page) en PDF proprement mis en page.

    Si le texte contient des caractères hors Latin-1 (ex: écriture birmane),
    utilise automatiquement une police Unicode téléchargée à la volée.
    """
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()

    needs_unicode = any(ord(ch) > 0xFF for ch in markdown_text)
    font_family = "Helvetica"
    use_unicode = False

    if needs_unicode:
        font_path = _ensure_unicode_font()
        if font_path is not None:
            pdf.add_font("NotoUnicode", "", str(font_path))
            pdf.add_font("NotoUnicode", "B", str(font_path))  # même fichier, pas de vraie graisse
            font_family = "NotoUnicode"
            use_unicode = True

    def render(text: str) -> str:
        return text if use_unicode else _safe_text(text)

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            pdf.ln(3)
            continue

        if line.strip() == "---":
            pdf.add_page()
            continue

        if line.startswith("# "):
            pdf.set_font(font_family, "B", 20)
            pdf.multi_cell(0, 10, render(line[2:].strip()))
            pdf.ln(4)
        elif line.startswith("## "):
            pdf.set_font(font_family, "B", 14)
            pdf.ln(2)
            pdf.multi_cell(0, 8, render(line[3:].strip()))
            pdf.ln(2)
        elif line.startswith("- "):
            pdf.set_font(font_family, "", 11)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 6, render(f"-  {line[2:].strip()}"))
        else:
            pdf.set_font(font_family, "", 11)
            pdf.multi_cell(0, 6, render(line.strip()))

    pdf.output(str(output_path))