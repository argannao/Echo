"""
Couche d'abstraction pour la transcription.

But : isoler le moteur Whisper derrière une interface commune pour pouvoir
remplacer facilement le backend de dev (faster-whisper, cross-platform,
utilisé sous Windows) par le backend final (whisper.cpp + Core ML, qui ne
tourne que sur Apple Silicon) sans toucher au reste du code.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class TranscriptionBackend(ABC):
    """Interface commune : un backend prend de l'audio, rend du texte."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        audio : signal audio mono en float32, valeurs entre -1.0 et 1.0
        sample_rate : fréquence d'échantillonnage en Hz (16000 attendu par Whisper)
        Retourne le texte transcrit (peut être une chaîne vide si silence/pas de parole).
        """
        raise NotImplementedError


class FasterWhisperBackend(TranscriptionBackend):
    """
    Backend de développement, cross-platform (Windows/Mac/Linux).
    Utilise CTranslate2 en CPU (ou CUDA si dispo) — pas d'accélération
    Metal/Core ML sur Mac, mais suffisant pour développer et tester la
    logique de capture/segmentation/UI sous Windows.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError("faster-whisper attend un sample_rate de 16000 Hz")

        segments, _info = self.model.transcribe(
            audio,
            language="fr",
            vad_filter=True,  # filtre silence interne en plus de notre propre VAD
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class WhisperCppBackend(TranscriptionBackend):
    """
    Backend final visé pour macOS Apple Silicon (MacBook Air M2).

    Appelle le binaire whisper.cpp compilé avec support Core ML (encodeur
    converti en modèle .mlmodelc) via subprocess. À implémenter et tester
    ce week-end sur le Mac — non fonctionnel sous Windows (le binaire
    Core ML n'existe que sur macOS).

    Piste d'implémentation :
      - compiler whisper.cpp avec WHISPER_COREML=1
      - convertir le modèle avec le script generate-coreml-model.sh fourni
      - appeler le binaire `main` (ou `whisper-cli`) en subprocess avec
        l'audio écrit temporairement en .wav 16kHz mono, parser la sortie
    """

    def __init__(self, binary_path: str, model_path: str):
        self.binary_path = binary_path
        self.model_path = model_path

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError(
            "Backend whisper.cpp à implémenter et tester sur macOS (ce week-end)."
        )


def get_default_backend() -> TranscriptionBackend:
    """Backend utilisé par défaut pendant la phase de dev sous Windows."""
    return FasterWhisperBackend(model_size="small", device="cpu", compute_type="int8")
