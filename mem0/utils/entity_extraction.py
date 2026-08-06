"""
Entity extraction from text using spaCy NLP.

Extracts three types of entities from a spaCy-processed document:
- **Proper nouns**: Capitalized multi-word sequences (person names, places, brands)
- **Quoted text**: Text in single or double quotes (titles, specific terms)
- **Noun compounds**: Multi-word noun phrases with specific modifiers (e.g., "machine learning")

Public API:
    ``extract_entities(text)`` accepts a string and owns spaCy model loading.
    ``extract_entities_batch(texts)`` uses ``nlp.pipe`` for batched extraction.

Returns:
    List of ``(entity_type, entity_text)`` tuples where entity_type is one of
    PROPER, QUOTED, TOPIC, or IDENTIFIER. Returns ``[]`` if spaCy is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class _EntityCandidate:
    entity_type: str
    text: str
    source: str
    start: int
    end: int
    confidence: float
    priority: int


# Words that are too generic to be useful as entity heads
_GENERIC_HEADS = {
    "thing",
    "stuff",
    "way",
    "time",
    "experience",
    "situation",
    "case",
    "fact",
    "matter",
    "issue",
    "idea",
    "thought",
    "feeling",
    "place",
    "area",
    "part",
    "kind",
    "type",
    "sort",
    "lot",
    "bit",
    "day",
    "year",
    "week",
    "month",
    "moment",
    "instance",
    "example",
    "technique",
    "method",
    "approach",
    "process",
    "step",
    "tool",
    "result",
    "outcome",
    "goal",
    "task",
    "item",
    "topic",
    "scale",
    "size",
    "level",
    "degree",
    "amount",
    "number",
    "style",
    "look",
    "color",
    "colour",
    "shape",
    "form",
    "piece",
    "section",
    "side",
    "end",
    "edge",
    "surface",
    "point",
}

# Entity labels emitted by spaCy that are usually safe to treat as named
# entities. Numeric and temporal labels are intentionally excluded.
_ACCEPTED_NER_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "PRODUCT",
    "WORK_OF_ART",
    "EVENT",
    "NORP",
    "LAW",
    "LANGUAGE",
}

_REJECTED_NER_LABELS = {
    "DATE",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "MONEY",
    "PERCENT",
}

# Generic role words and title-cased English words that should not become
# single-token named entities just because spaCy tagged them as PROPN.
_GENERIC_SINGLE_ENTITY_TERMS = {
    "user",
    "assistant",
    "agent",
    "customer",
    "client",
    "person",
    "people",
    "human",
    "memory",
    "message",
    "conversation",
    "chat",
    "session",
    "system",
    "top",
}

# Modifiers that describe circumstance, not content
_CIRCUMSTANTIAL_MODS = {
    "solo",
    "individual",
    "team",
    "group",
    "joint",
    "collaborative",
    "first",
    "last",
    "next",
    "previous",
    "final",
    "initial",
    "main",
    "side",
    "top",
}

# Adjectives too vague to make a compound entity specific
_NON_SPECIFIC_ADJ = {
    "many",
    "few",
    "several",
    "some",
    "any",
    "all",
    "most",
    "more",
    "less",
    "much",
    "little",
    "enough",
    "various",
    "numerous",
    "multiple",
    "countless",
    "great",
    "good",
    "bad",
    "nice",
    "terrible",
    "awful",
    "awesome",
    "amazing",
    "wonderful",
    "horrible",
    "excellent",
    "poor",
    "best",
    "worst",
    "fine",
    "okay",
    "new",
    "old",
    "recent",
    "past",
    "future",
    "current",
    "previous",
    "next",
    "last",
    "first",
    "latest",
    "early",
    "late",
    "former",
    "modern",
    "ancient",
    "big",
    "small",
    "large",
    "tiny",
    "huge",
    "enormous",
    "long",
    "short",
    "tall",
    "high",
    "low",
    "wide",
    "narrow",
    "thick",
    "thin",
    "deep",
    "shallow",
    "similar",
    "different",
    "same",
    "other",
    "another",
    "such",
    "certain",
    "important",
    "main",
    "major",
    "minor",
    "key",
    "primary",
    "real",
    "actual",
    "true",
    "whole",
    "entire",
    "full",
    "complete",
    "total",
    "basic",
    "simple",
    "interesting",
    "boring",
    "exciting",
    "special",
    "particular",
    "general",
    "common",
    "unique",
    "rare",
    "typical",
    "usual",
    "normal",
    "regular",
    "possible",
    "likely",
    "potential",
    "available",
    "necessary",
    "only",
    "solo",
    "individual",
    "team",
    "group",
    "joint",
    "collaborative",
    "final",
    "initial",
    "side",
}

# Generic tail words to strip from compound entities
_GENERIC_ENDINGS = {
    "work",
    "works",
    "job",
    "jobs",
    "task",
    "tasks",
    "stuff",
    "things",
    "thing",
    "info",
    "information",
    "details",
    "data",
    "content",
    "material",
    "materials",
    "activities",
    "activity",
    "efforts",
    "effort",
    "options",
    "option",
    "choices",
    "choice",
    "results",
    "result",
    "output",
    "outputs",
    "products",
    "product",
    "items",
    "item",
}

# Capitalized single words that are too generic to be proper nouns
_GENERIC_CAPS = {
    "works",
    "items",
    "things",
    "stuff",
    "resources",
    "options",
    "tips",
    "ideas",
    "steps",
    "ways",
    "methods",
    "tools",
    "features",
    "benefits",
    "examples",
    "details",
    "notes",
    "instructions",
    "guidelines",
    "recommendations",
    "suggestions",
    "overview",
    "summary",
    "conclusion",
    "introduction",
    "pros",
    "cons",
    "advantages",
    "disadvantages",
}

# Markdown/formatting markers to skip during extraction
_FORMATTING_MARKERS = {"*", "-", "+", "\u2022", "\u2013", "\u2014", "#", "##", "###", "**", "__"}


def _is_sentence_start(tokens: list, idx: int) -> bool:
    """Check if a token is at the start of a sentence or after formatting."""
    if idx == 0:
        return True
    tok = tokens[idx]
    if tok.is_sent_start:
        return True
    prev = tokens[idx - 1].text
    return prev in ".!?:" or prev in _FORMATTING_MARKERS or "\n" in prev


def _strip_generic_ending(toks: list) -> list:
    """Remove generic trailing words from compound token sequences."""
    if len(toks) <= 1:
        return toks
    last = toks[-1].lemma_.lower() if hasattr(toks[-1], "lemma_") else toks[-1].lower()
    return toks[:-1] if last in _GENERIC_ENDINGS and len(toks) > 2 else toks


def _lemmatize_compound(toks: list) -> str:
    """Join compound tokens, lemmatizing nouns."""
    return " ".join(t.lemma_ if t.pos_ == "NOUN" else t.text for t in toks)


def _has_artifacts(txt: str) -> bool:
    """Check for formatting artifacts that indicate non-entity text."""
    return any(
        [
            "**" in txt or "__" in txt or ":*" in txt,
            re.search(r"\s\*\s|\s\*$|^\*\s", txt),
            "  " in txt or "\n" in txt or "\t" in txt,
            len(txt) > 100,
            txt.startswith(("\u2022", "-", "+", "\u2013", "\u2014")),
        ]
    )


def _clean_text(txt: str) -> str:
    txt = re.sub(r"^\*+\s*|\s*\*+$", "", txt.strip())
    txt = re.sub(r"\s*:+$", "", txt)
    txt = re.sub(r"^\d+\s*\.\s*", "", txt)
    return " ".join(txt.split())


def _norm_text(txt: str) -> str:
    return " ".join(txt.lower().split())


def _looks_like_technical_identifier(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+", text))


def _has_internal_cap_or_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text) or any(ch.isupper() for ch in text[1:])


def _looks_like_metric_count_token(tok) -> bool:
    return tok.pos_ == "NUM" and bool(re.fullmatch(r"\d[\d,]*(?:\.\d+)?", tok.text))


def _is_metric_list_context(tokens: list, idx: int) -> bool:
    prev_text = tokens[idx - 1].text if idx > 0 else ""
    next_text = tokens[idx + 1].text if idx + 1 < len(tokens) else ""
    return prev_text in {":", ",", ";"} or next_text in {",", ";"}


def _strip_trailing_metric_counts(span_tokens: list, all_tokens: list) -> list:
    while len(span_tokens) > 1 and _looks_like_metric_count_token(span_tokens[-1]):
        tok = span_tokens[-1]
        if "," not in tok.text and not _is_metric_list_context(all_tokens, tok.i):
            break
        span_tokens = span_tokens[:-1]
    return span_tokens


def _is_list_item_name_token(tokens: list, idx: int) -> bool:
    tok = tokens[idx]
    if not tok.text or tok.text in _FORMATTING_MARKERS or not tok.text[0].isupper():
        return False
    if not any(ch.isalpha() for ch in tok.text) or _is_bad_single_name_token(tok):
        return False
    next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else None
    if not next_tok or not _looks_like_metric_count_token(next_tok):
        return False
    return _is_metric_list_context(tokens, idx) or _is_metric_list_context(tokens, idx + 1)


def _is_name_like_token(tok, tokens: list | None = None, idx: int | None = None) -> bool:
    if not tok.text or tok.text in _FORMATTING_MARKERS:
        return False
    if not tok.text[0].isupper():
        return False
    if not any(ch.isalpha() for ch in tok.text):
        return False
    if _is_bad_single_name_token(tok):
        return False
    if tok.pos_ == "PROPN" or tok.tag_ in {"NNP", "NNPS"}:
        return True
    if tokens is not None and idx is not None and _is_list_item_name_token(tokens, idx):
        return True
    if _has_internal_cap_or_digit(tok.text):
        return True
    return (
        tokens is not None
        and idx is not None
        and tok.pos_ == "NOUN"
        and tok.dep_ not in {"compound", "amod"}
        and not _is_sentence_start(tokens, idx)
    )


def _is_bad_single_name_token(tok) -> bool:
    lower = tok.text.lower()
    return lower in _GENERIC_SINGLE_ENTITY_TERMS or lower in _GENERIC_CAPS or tok.is_stop


def _add_candidate(
    candidates: list[_EntityCandidate],
    entity_type: str,
    text: str,
    source: str,
    start: int,
    end: int,
    confidence: float,
    priority: int,
) -> None:
    cleaned = _clean_text(text)
    if not cleaned or _has_artifacts(cleaned):
        return
    # For CJK text, 2 characters is often meaningful (e.g., person names)
    # For non-CJK text, require at least 3 characters
    min_len = 2 if _has_cjk(cleaned) else 3
    if len(cleaned) < min_len:
        return
    candidates.append(
        _EntityCandidate(
            entity_type=entity_type,
            text=cleaned,
            source=source,
            start=start,
            end=end,
            confidence=confidence,
            priority=priority,
        )
    )


def _add_ner_candidates(doc, candidates: list[_EntityCandidate]) -> None:
    tokens = list(doc)
    for ent in doc.ents:
        if ent.label_ in _REJECTED_NER_LABELS or ent.label_ not in _ACCEPTED_NER_LABELS:
            continue
        ent_tokens = _strip_trailing_metric_counts(list(ent), tokens)
        if not ent_tokens:
            continue
        if any(tok.pos_ == "CCONJ" and tok.text.lower() == "and" for tok in ent_tokens):
            continue
        if len(ent_tokens) == 1 and _is_bad_single_name_token(ent_tokens[0]):
            continue
        if (
            len(ent_tokens) == 1
            and ent_tokens[0].dep_ in {"compound", "amod"}
            and ent_tokens[0].head.pos_ in {"NOUN", "PROPN"}
        ):
            continue
        _add_candidate(
            candidates,
            "PROPER",
            "".join(tok.text_with_ws for tok in ent_tokens).strip(),
            "spacy_ner",
            ent_tokens[0].i,
            ent_tokens[-1].i + 1,
            0.95,
            0,
        )


def _add_technical_identifier_candidates(tokens: list, candidates: list[_EntityCandidate]) -> None:
    for tok in tokens:
        if _looks_like_technical_identifier(tok.text):
            _add_candidate(
                candidates,
                "IDENTIFIER",
                tok.text,
                "technical_identifier",
                tok.i,
                tok.i + 1,
                0.9,
                1,
            )


def _add_proper_name_candidates(tokens: list, candidates: list[_EntityCandidate]) -> None:
    allowed_inner_connectors = {"of", "the", "for", "at", "in"}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not _is_name_like_token(tok, tokens, i):
            i += 1
            continue

        span_tokens = [tok]
        j = i + 1
        while j < len(tokens):
            current = tokens[j]
            if _is_name_like_token(current, tokens, j):
                span_tokens.append(current)
                j += 1
                continue
            if (
                current.text.lower() in allowed_inner_connectors
                and j + 1 < len(tokens)
                and _is_name_like_token(tokens[j + 1], tokens, j + 1)
            ):
                span_tokens.extend([current, tokens[j + 1]])
                j += 2
                continue
            break

        name_tokens = [
            t
            for t in span_tokens
            if _is_name_like_token(t, tokens, t.i) or (0 <= t.i < len(tokens) and _is_list_item_name_token(tokens, t.i))
        ]
        if len(name_tokens) > 1 or not _is_bad_single_name_token(name_tokens[0]):
            text = "".join(t.text_with_ws for t in span_tokens).strip()
            _add_candidate(candidates, "PROPER", text, "proper_name_span", i, j, 0.8, 2)
        i = max(j, i + 1)


def _add_quoted_candidates(text: str, candidates: list[_EntityCandidate]) -> None:
    for m in re.finditer(r'"([^"]+)"', text):
        if len(m.group(1).strip()) > 2:
            _add_candidate(candidates, "QUOTED", m.group(1).strip(), "quoted", -1, -1, 0.75, 3)
    for m in re.finditer(r"(?:^|[\s\(\[{,;])'([^']+)'(?=[\s\.,;:!?\)\]]|$)", text):
        if len(m.group(1).strip()) > 2:
            _add_candidate(candidates, "QUOTED", m.group(1).strip(), "quoted", -1, -1, 0.75, 3)


def _add_topic_phrase_candidates(doc, candidates: list[_EntityCandidate]) -> None:
    for chunk in doc.noun_chunks:
        chunk_tokens = list(chunk)
        split_indices: list[int] = []
        poss_splits: list[int] = []
        for idx, tok in enumerate(chunk_tokens):
            if tok.dep_ == "case" and tok.text in {"'s", "\u2019s", "'"}:
                split_indices.append(idx)
                poss_splits.append(idx)
            elif tok.pos_ == "PUNCT" and tok.text in {"'", '"', "\u2018", "\u2019", "\u201c", "\u201d"}:
                split_indices.append(idx)

        if split_indices:
            groups: list[list] = []
            prev = 0
            for split_idx in split_indices:
                if split_idx > prev:
                    groups.append(chunk_tokens[prev:split_idx])
                if split_idx in poss_splits:
                    next_split = next((s for s in split_indices if s > split_idx), None)
                    owned = chunk_tokens[split_idx + 1 : next_split if next_split else len(chunk_tokens)]
                    if owned:
                        first_content = next((t for t in owned if t.pos_ not in {"PUNCT", "PART"}), None)
                        if not (first_content and first_content.text and first_content.text[0].isupper()):
                            prev = next_split if next_split else len(chunk_tokens)
                            continue
                prev = split_idx + 1
            if prev < len(chunk_tokens):
                groups.append(chunk_tokens[prev:])
        else:
            groups = [chunk_tokens]

        for group in groups:
            if not group:
                continue
            head = next((t for t in reversed(group) if t.pos_ in {"NOUN", "PROPN"}), None)
            if not head:
                continue
            head_generic = head.lemma_.lower() in _GENERIC_HEADS
            content = [
                t
                for t in group
                if t.pos_ not in {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"}
                and (t.pos_ == "ADJ" or not t.is_stop)
            ]
            if not content:
                continue

            compound_toks = [t for t in content if t.dep_ == "compound"]
            adj_toks = [t for t in content if t.pos_ == "ADJ" or t.dep_ == "amod"]
            has_spec_adj = any(t.lemma_.lower() not in _NON_SPECIFIC_ADJ for t in adj_toks)
            if head_generic and not has_spec_adj and not compound_toks:
                continue

            if compound_toks:
                is_circ = any(t.lemma_.lower() in _CIRCUMSTANTIAL_MODS for t in compound_toks)
                if is_circ:
                    val = head.text
                    if len(val) > 2:
                        _add_candidate(
                            candidates,
                            "TOPIC",
                            val,
                            "topic_phrase",
                            head.i,
                            head.i + 1,
                            0.45,
                            4,
                        )
                else:
                    filtered = _strip_generic_ending(
                        [t for t in content if not (t.pos_ == "ADJ" and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                    )
                    if filtered:
                        phrase = " ".join(t.text for t in filtered)
                        if len(phrase) > 3 and " " in phrase:
                            _add_candidate(
                                candidates,
                                "TOPIC",
                                phrase,
                                "topic_phrase",
                                filtered[0].i,
                                filtered[-1].i + 1,
                                0.45,
                                4,
                            )
            elif len(content) > 1 and has_spec_adj:
                filtered = _strip_generic_ending(
                    [
                        t
                        for t in content
                        if not ((t.pos_ == "ADJ" or t.dep_ == "amod") and t.lemma_.lower() in _NON_SPECIFIC_ADJ)
                    ]
                )
                if filtered:
                    phrase = " ".join(t.text for t in filtered)
                    if len(phrase) > 3 and " " in phrase:
                        _add_candidate(
                            candidates,
                            "TOPIC",
                            phrase,
                            "topic_phrase",
                            filtered[0].i,
                            filtered[-1].i + 1,
                            0.45,
                            4,
                        )


def _spans_overlap(a: _EntityCandidate, b: _EntityCandidate) -> bool:
    if a.start < 0 or b.start < 0:
        return False
    return a.start < b.end and b.start < a.end


def _resolve_candidates(candidates: list[_EntityCandidate]) -> list[tuple[str, str]]:
    deduped_by_text: dict[str, _EntityCandidate] = {}
    for candidate in candidates:
        key = _norm_text(candidate.text)
        current = deduped_by_text.get(key)
        if current is None or (candidate.priority, -candidate.confidence) < (current.priority, -current.confidence):
            deduped_by_text[key] = candidate

    ordered = sorted(
        deduped_by_text.values(),
        key=lambda c: (c.priority, -c.confidence, -(c.end - c.start), c.start),
    )
    accepted: list[_EntityCandidate] = []
    for candidate in ordered:
        if any(
            _spans_overlap(candidate, existing)
            and not (
                candidate.entity_type == "TOPIC"
                and len(candidate.text) > len(existing.text)  # multi-word/multi-char TOPIC overrides single PROPER
                and existing.entity_type == "PROPER"
            )
            for existing in accepted
        ):
            continue
        accepted.append(candidate)

    accepted.sort(key=lambda c: (c.start if c.start >= 0 else 10**9, c.end, c.priority))
    return [(candidate.entity_type, candidate.text) for candidate in accepted]


def _extract_entities_from_doc(doc) -> list[tuple[str, str]]:
    """Extract typed entity candidates from a spaCy Doc.

    Args:
        doc: A spaCy ``Doc`` object (from ``nlp(text)``).

    Returns:
        Deduplicated list of ``(entity_type, entity_text)`` tuples.
        Entity types include PROPER, QUOTED, TOPIC, and IDENTIFIER.
    """
    tokens = list(doc)
    candidates: list[_EntityCandidate] = []
    _add_ner_candidates(doc, candidates)
    _add_technical_identifier_candidates(tokens, candidates)
    _add_proper_name_candidates(tokens, candidates)
    _add_quoted_candidates(doc.text, candidates)
    _add_topic_phrase_candidates(doc, candidates)
    return _resolve_candidates(candidates)


# ─── CJK / Chinese entity extraction ─────────────────────────────────────────

_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')
_SEGMENT_RE = re.compile(r'([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+)')

# Chinese stop words and generic words to exclude from entities
_ZH_GENERIC_WORDS = {
    "用户", "我们", "他们", "自己", "这个", "那个", "什么", "怎么",
    "可以", "需要", "应该", "已经", "还是", "因为", "所以", "但是",
    "如果", "虽然", "或者", "以及", "关于", "对于", "一些", "一个",
    "比较", "非常", "可能", "已经", "正在", "之后", "之前", "时候",
    "东西", "事情", "方面", "地方", "情况", "问题", "方式", "方法",
}


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _extract_zh_entities(doc) -> list[tuple[str, str]]:
    """Extract entities from a Chinese spaCy Doc.

    Uses NER labels and noun phrase patterns specific to Chinese.
    """
    candidates: list[_EntityCandidate] = []

    # 1. NER entities (PERSON, GPE, ORG, etc.)
    for ent in doc.ents:
        if ent.label_ in _REJECTED_NER_LABELS:
            continue
        text = ent.text.strip()
        if len(text) < 2 or text in _ZH_GENERIC_WORDS:
            continue
        _add_candidate(candidates, "PROPER", text, "zh_ner", ent.start, ent.end, 0.9, 0)

    # 2. Quoted text
    _add_quoted_candidates(doc.text, candidates)

    # 3. Chinese noun phrases via dependency parsing and linear adjacency
    #    Since noun_chunks is not available for zh, we build phrases manually
    tokens = list(doc)
    used_spans: set[tuple[int, int]] = set()

    for i, tok in enumerate(tokens):
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue
        if tok.text in _ZH_GENERIC_WORDS:
            continue

        # Build compound noun phrase: collect modifiers
        phrase_tokens = [tok]

        # Look left for dependency-based modifiers
        for child in tok.children:
            if child.i < tok.i and child.dep_ in (
                "amod", "compound", "nmod", "nmod:assmod",
                "nmod:prep", "compound:nn", "advmod:rcomp",
            ):
                if child.text not in _ZH_GENERIC_WORDS:
                    phrase_tokens.append(child)

        # Linear adjacency: check if preceding token is a VERB used as modifier
        # e.g., "咖喱" (VERB) + "牛肉" (NOUN) where spaCy parses them separately
        if i > 0:
            prev = tokens[i - 1]
            if prev.pos_ == "VERB" and prev.text not in _ZH_GENERIC_WORDS and len(prev.text) >= 2:
                # Check if this VERB acts as a modifier (advmod:rcomp or similar)
                if prev.dep_ in ("advmod:rcomp", "compound") or prev.head == tok or prev.head == tok.head:
                    if prev not in phrase_tokens:
                        phrase_tokens.append(prev)

        # Sort by position
        phrase_tokens.sort(key=lambda t: t.i)

        # Check if this is a meaningful phrase
        phrase_text = "".join(t.text for t in phrase_tokens)

        # Skip if too short or just a generic word
        if len(phrase_text) < 2:
            continue

        # Skip if all tokens are generic
        if all(t.text in _ZH_GENERIC_WORDS for t in phrase_tokens):
            continue

        start = phrase_tokens[0].i
        end = phrase_tokens[-1].i + 1
        span_key = (start, end)

        if span_key not in used_spans:
            used_spans.add(span_key)
            # Multi-word phrases get TOPIC, single proper nouns get PROPER
            if len(phrase_tokens) > 1:
                _add_candidate(candidates, "TOPIC", phrase_text, "zh_noun_phrase", start, end, 0.6, 4)
            elif tok.pos_ == "PROPN":
                _add_candidate(candidates, "PROPER", phrase_text, "zh_proper_noun", start, end, 0.8, 2)
            elif tok.pos_ == "NOUN" and len(phrase_text) >= 2:
                _add_candidate(candidates, "TOPIC", phrase_text, "zh_noun", start, end, 0.5, 4)

    return _resolve_candidates(candidates)


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract typed entity candidates from text.

    Supports mixed Chinese-English text by splitting into CJK and non-CJK
    segments, processing each with the appropriate spaCy model.
    """
    if not text or not text.strip():
        return []

    # Pure English path
    if not _has_cjk(text):
        from mem0.utils.spacy_models import get_nlp_full

        nlp = get_nlp_full()
        if nlp is None:
            return []
        return _extract_entities_from_doc(nlp(text))

    # Pure Chinese or mixed text
    segments = _SEGMENT_RE.split(text)
    all_entities: list[tuple[str, str]] = []

    from mem0.utils.spacy_models import get_nlp_full, get_nlp_zh_full

    nlp_en = get_nlp_full()
    nlp_zh = get_nlp_zh_full()

    for seg in segments:
        if not seg.strip():
            continue

        if _has_cjk(seg):
            # Chinese segment
            if nlp_zh is not None:
                doc = nlp_zh(seg)
                all_entities.extend(_extract_zh_entities(doc))
        else:
            # English segment
            if nlp_en is not None:
                doc = nlp_en(seg)
                all_entities.extend(_extract_entities_from_doc(doc))

    # Deduplicate across segments
    seen: dict[str, tuple[str, str]] = {}
    for entity_type, entity_text in all_entities:
        key = entity_text.strip().lower()
        if key not in seen:
            seen[key] = (entity_type, entity_text)

    return list(seen.values())


def extract_entities_batch(texts: list[str], batch_size: int = 32) -> list[list[tuple[str, str]]]:
    """Extract typed entity candidates from multiple texts.

    Handles mixed Chinese-English by routing each text through the
    appropriate model (en_core_web_sm or zh_core_web_sm).
    """
    if not texts:
        return []

    from mem0.utils.spacy_models import get_nlp_full, get_nlp_zh_full

    nlp_en = get_nlp_full()
    nlp_zh = get_nlp_zh_full()

    # Separate texts by language
    results: list[list[tuple[str, str]]] = [[] for _ in texts]
    en_indices: list[int] = []
    zh_indices: list[int] = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        if _has_cjk(text):
            zh_indices.append(i)
        else:
            en_indices.append(i)

    # Process English texts in batch
    if en_indices and nlp_en is not None:
        en_texts = [texts[i] for i in en_indices]
        for idx, doc in zip(en_indices, nlp_en.pipe(en_texts, batch_size=batch_size)):
            results[idx] = _extract_entities_from_doc(doc)

    # Process Chinese/mixed texts individually (need segment splitting)
    if zh_indices:
        for idx in zh_indices:
            results[idx] = extract_entities(texts[idx])

    return results
