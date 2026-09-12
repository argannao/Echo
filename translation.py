"""
Traduction des notes générées, via NLLB-200 (Meta) — un modèle dédié à la
traduction, entraîné sur ~200 langues dont beaucoup à faibles ressources
(birman inclus), bien plus fiable pour ça qu'un petit LLM généraliste.

Entièrement local (téléchargé une fois depuis Hugging Face, comme le
modèle Whisper), aucune donnée envoyée en ligne après le premier
téléchargement.
"""

from __future__ import annotations

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANG_CODE = "fra_Latn"

# Langues proposées dans l'UI -> code NLLB correspondant. Facile à étendre :
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANGUAGE_CODES = {
    "Anglais": "eng_Latn",
    "Birman": "mya_Mymr",
    "Espagnol": "spa_Latn",
    "Allemand": "deu_Latn",
    "Thaï": "tha_Thai",
    "Vietnamien": "vie_Latn",
    "Chinois (simplifié)": "zho_Hans",
}

_model = None
_tokenizer = None


def _load_translator():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _model, _tokenizer


def _translate_text(text: str, target_code: str) -> str:
    model, tokenizer = _load_translator()
    tokenizer.src_lang = SOURCE_LANG_CODE

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
        max_length=512,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def translate_markdown(markdown_text: str, target_language: str) -> str:
    """
    Traduit un texte structuré (sous-ensemble simple de Markdown : #, ##,
    -) ligne par ligne, en préservant les préfixes de structure.
    """
    if target_language not in LANGUAGE_CODES:
        raise ValueError(f"Langue non supportée : {target_language}")
    target_code = LANGUAGE_CODES[target_language]

    try:
        translated_lines = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if not line:
                translated_lines.append("")
                continue

            prefix, content = "", line
            if line.startswith("# "):
                prefix, content = "# ", line[2:]
            elif line.startswith("## "):
                prefix, content = "## ", line[3:]
            elif line.startswith("- "):
                prefix, content = "- ", line[2:]

            translated_lines.append(prefix + _translate_text(content, target_code))

        return "\n".join(translated_lines)
    except ImportError as exc:
        raise RuntimeError(
            "Les paquets 'transformers'/'torch' ne sont pas installés "
            "(uv add transformers torch sentencepiece)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Erreur de traduction : {exc}") from exc