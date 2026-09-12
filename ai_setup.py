"""
Détection et configuration automatique de l'IA locale.

Contrairement à Ollama, il n'y a pas d'application externe à surveiller :
seulement le paquet Python `llama-cpp-python` et le fichier modèle (.gguf),
téléchargé automatiquement au besoin.
"""

from __future__ import annotations

from notes_generator import ensure_model_downloaded, get_model_path


def check_status() -> str:
    """
    Retourne l'un de :
      "missing_package" -> llama-cpp-python n'est pas installé
      "missing_model"    -> le paquet est là mais le modèle n'est pas téléchargé
      "ready"            -> tout est prêt
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return "missing_package"

    if get_model_path() is None:
        return "missing_model"
    return "ready"


def download_model(on_progress=None) -> None:
    ensure_model_downloaded(on_progress=on_progress)