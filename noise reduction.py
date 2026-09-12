"""
Réduction de bruit ambiant, appliquée à chaque segment audio juste avant
la transcription — filtre les bruits de fond stationnaires (ventilo,
bourdonnement, souffle...) sans toucher à la voix.

Utilise `noisereduce` (spectral gating), plus léger et plus simple à
déployer qu'un modèle de suppression de bruit par deep learning (type
Krisp/RNNoise), tout en étant efficace sur le cas d'usage principal :
un cours enregistré dans une salle avec un bruit de fond à peu près
stable.
"""

from __future__ import annotations

import numpy as np


def denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Retourne une version débruitée du signal. En cas de souci (dépendance
    absente, erreur de traitement...), retourne l'audio d'origine plutôt
    que de faire planter la transcription — le débruitage est un bonus,
    jamais un point de blocage.
    """
    try:
        import noisereduce as nr
    except ImportError:
        return audio

    try:
        return nr.reduce_noise(y=audio, sr=sample_rate, stationary=False)
    except Exception:
        return audio