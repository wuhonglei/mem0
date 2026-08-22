"""
Shared spaCy model loader.

Consolidates spaCy model loading into a single module so that
entity_extraction and lemmatization share one instance instead of
each loading their own copy from disk.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_nlp_full = None
_nlp_lemma = None
_nlp_zh = None
_load_failed_full = False
_load_failed_lemma = False
_load_failed_zh = False
_lock = threading.Lock()


def _ensure_model_available():
    """Download en_core_web_sm if spaCy is installed but model is missing."""
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        )

    if not spacy.util.is_package("en_core_web_sm"):
        logger.info("Downloading spaCy model en_core_web_sm...")
        try:
            from spacy.cli import download

            download("en_core_web_sm")
            logger.info("spaCy model en_core_web_sm downloaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download spaCy model en_core_web_sm: {e}. "
                "Please install manually: python -m spacy download en_core_web_sm"
            ) from e


def _ensure_zh_model_available():
    """Download zh_core_web_sm if spaCy is installed but model is missing."""
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        )

    if not spacy.util.is_package("zh_core_web_sm"):
        logger.info("Downloading spaCy model zh_core_web_sm...")
        try:
            from spacy.cli import download

            download("zh_core_web_sm")
            logger.info("spaCy model zh_core_web_sm downloaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download spaCy model zh_core_web_sm: {e}. "
                "Please install manually: python -m spacy download zh_core_web_sm"
            ) from e


def get_nlp_full():
    """Return spaCy model with all pipelines (NER, tagger, etc.) for entity extraction."""
    global _nlp_full, _load_failed_full
    if _load_failed_full:
        return None
    if _nlp_full is not None:
        return _nlp_full
    with _lock:
        if _nlp_full is not None:
            return _nlp_full
        if _load_failed_full:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_full = spacy.load("en_core_web_sm")
            logger.info("spaCy full model loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy full model: {e}")
            _load_failed_full = True
            return None
    return _nlp_full


def get_nlp_lemma():
    """Return spaCy model with only lemmatizer for BM25 text processing."""
    global _nlp_lemma, _load_failed_lemma
    if _load_failed_lemma:
        return None
    if _nlp_lemma is not None:
        return _nlp_lemma
    with _lock:
        if _nlp_lemma is not None:
            return _nlp_lemma
        if _load_failed_lemma:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_lemma = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            logger.info("spaCy lemma model loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy lemma model: {e}")
            _load_failed_lemma = True
            return None
    return _nlp_lemma


def get_nlp_zh():
    """Return spaCy zh_core_web_sm model for Chinese text segmentation."""
    global _nlp_zh, _load_failed_zh
    if _load_failed_zh:
        return None
    if _nlp_zh is not None:
        return _nlp_zh
    with _lock:
        if _nlp_zh is not None:
            return _nlp_zh
        if _load_failed_zh:
            return None
        try:
            _ensure_zh_model_available()
            import spacy

            _nlp_zh = spacy.load("zh_core_web_sm", disable=["ner", "parser"])
            logger.info("spaCy zh model loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy zh model: {e}")
            _load_failed_zh = True
            return None
    return _nlp_zh


_nlp_zh_full = None
_load_failed_zh_full = False


def get_nlp_zh_full():
    """Return spaCy zh_core_web_sm model with all pipelines for entity extraction."""
    global _nlp_zh_full, _load_failed_zh_full
    if _load_failed_zh_full:
        return None
    if _nlp_zh_full is not None:
        return _nlp_zh_full
    with _lock:
        if _nlp_zh_full is not None:
            return _nlp_zh_full
        if _load_failed_zh_full:
            return None
        try:
            _ensure_zh_model_available()
            import spacy

            _nlp_zh_full = spacy.load("zh_core_web_sm")
            logger.info("spaCy zh full model loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy zh full model: {e}")
            _load_failed_zh_full = True
            return None
    return _nlp_zh_full


def preload_all() -> None:
    """Load all spaCy models eagerly so the first user request is not slow.

    Safe to call multiple times — already-loaded models are returned instantly.
    Failures are logged at warning level and never raise.
    """
    for name, loader in [
        ("en_core_web_sm (full)", get_nlp_full),
        ("en_core_web_sm (lemma)", get_nlp_lemma),
        ("zh_core_web_sm (tokenize)", get_nlp_zh),
        ("zh_core_web_sm (full)", get_nlp_zh_full),
    ]:
        try:
            loader()
        except Exception as e:
            logger.warning(f"Preload failed for {name}: {e}")
