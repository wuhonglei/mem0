"""
BM25 lemmatization for consistent keyword matching.

Bilingual approach: splits text into English and Chinese segments using regex,
then applies en_core_web_sm for English lemmatization and zh_core_web_sm for
Chinese word segmentation. Results are joined back together.

- English: verb forms, comparatives, plurals are lemmatized
- Chinese: segmented into words by spaCy's Chinese tokenizer
- Mixed text (e.g. "喜欢 Pydantic AI 的 context management 功能") is handled correctly
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# CJK Unified Ideographs range (covers Chinese, Japanese Kanji, Korean Hanja)
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')
# Split text into CJK runs and non-CJK runs
_SEGMENT_RE = re.compile(r'([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+)')


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(_CJK_RE.search(text))


def _lemmatize_en(text: str) -> str:
    """Lemmatize a pure-English text segment."""
    from mem0.utils.spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)


def _segment_zh(text: str) -> str:
    """Segment Chinese text into words, filtering stop words and punctuation."""
    from mem0.utils.spacy_models import get_nlp_zh

    nlp = get_nlp_zh()
    if nlp is None:
        # Fallback: return text with spaces between characters
        return " ".join(text)

    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        word = token.text.strip()
        if word:
            tokens.append(word)

    return " ".join(tokens)


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize/segment text for BM25 matching.

    For mixed Chinese+English text, splits into segments and processes each:
    - English segments: lemmatized via en_core_web_sm
    - Chinese segments: word-segmented via zh_core_web_sm
    Results are joined with spaces.

    Falls back to original text if spaCy is unavailable.
    """
    # Pure CJK text — no English to lemmatize
    if not _has_cjk(text):
        # Pure English path (original behavior)
        return _lemmatize_en(text)

    # Mixed or pure Chinese text — split into CJK and non-CJK segments
    segments = _SEGMENT_RE.split(text)
    result_parts = []

    for seg in segments:
        if not seg.strip():
            continue

        if _has_cjk(seg):
            # Chinese segment — word segmentation
            result_parts.append(_segment_zh(seg))
        else:
            # English segment — lemmatize
            result_parts.append(_lemmatize_en(seg))

    result = " ".join(result_parts)
    # Collapse multiple spaces from punctuation removal
    return re.sub(r'\s+', ' ', result).strip()
