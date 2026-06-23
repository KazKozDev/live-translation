"""Text cleanup, segmentation, and prompt helpers for live translation."""

import re

LANG_NAMES = {
    "auto": "the target language",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "zh": "Chinese",
}

LANG_MENU = [
    ("auto", "Auto"),
    ("en", "English"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("de", "German"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("uk", "Ukrainian"),
    ("zh", "Chinese"),
]

WAITING_ORIGINAL = "Waiting for speech…"
WAITING_TRANSLATION = {
    "en": "Waiting for translation…",
    "ru": "Жду перевод…",
    "es": "Esperando traducción…",
    "de": "Warte auf Übersetzung…",
    "fr": "En attente de traduction…",
    "it": "In attesa di traduzione…",
    "pt": "Aguardando tradução…",
    "uk": "Чекаю на переклад…",
    "zh": "等待翻译…",
}


def language_name(code):
    return LANG_NAMES.get((code or "").lower(), code or "the target language")


def language_label(code):
    for item_code, label in LANG_MENU:
        if item_code == code:
            return label
    return code or "Auto"


def prompt_language(code, auto_label="auto-detected speech"):
    code = (code or "").lower()
    if code == "auto":
        return auto_label
    name = language_name(code)
    return f"{name} ({code})" if code else auto_label


def live_translation_messages(src_code, tgt_code, text, history=None):
    target = language_name(tgt_code)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a low-latency live speech translation engine. "
                f"Translate the new transcript into {target}. "
                "Return only the translation. Do not explain. Do not summarize. "
                "Do not add commentary. Preserve names, numbers, places, dates, "
                "product names, and technical terms. Fix obvious speech-recognition "
                "errors only when the intended meaning is clear. If the transcript is "
                "incomplete, translate only what is present. Earlier turns are prior "
                "context for consistent terminology and pronouns — translate only the "
                "newest transcript, never repeat earlier ones."
            ),
        },
    ]
    # Prior committed source -> translation pairs, replayed as conversation turns so the
    # model keeps terminology, names and pronouns consistent across blocks.
    for prev_src, prev_tgt in history or []:
        prev_src = re.sub(r"\s+", " ", str(prev_src or "")).strip()
        prev_tgt = re.sub(r"\s+", " ", str(prev_tgt or "")).strip()
        if not prev_src or not prev_tgt:
            continue
        messages.append({"role": "user", "content": f"New transcript:\n{prev_src}"})
        messages.append({"role": "assistant", "content": prev_tgt})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Source language: {prompt_language(src_code)}\n"
                f"Target language: {prompt_language(tgt_code, 'the target language')}\n\n"
                f"New transcript:\n{text}"
            ),
        }
    )
    return messages


HALLUCINATION_PATTERNS = [
    r"¡?\s*gracias\s*!?",
    r"muchas\s+gracias",
    r"subt[ií]tulos?(?:\s+(?:realizados?|por)[^.\n]*)?",
    r"amara\.org",
    r"thanks?\s+for\s+watching",
    r"please\s+subscribe",
    r"subtitles?\s+by[^.\n]*",
    r"спасибо\s+за\s+просмотр",
    r"продолжение\s+следует",
]
_HALLUCINATION_RE = re.compile(
    r"(?<![\w¡])(?:" + "|".join(HALLUCINATION_PATTERNS) + r")(?![\w])",
    re.IGNORECASE,
)


def _repeated_ngram_start(words, min_size=3, max_size=10, min_repeats=3):
    if len(words) < min_size * min_repeats:
        return None
    upper_size = min(max_size, len(words) // min_repeats)
    for size in range(upper_size, min_size - 1, -1):
        positions = {}
        for idx in range(len(words) - size + 1):
            gram = tuple(words[idx : idx + size])
            found = positions.setdefault(gram, [])
            if found and idx < found[-1] + size:
                continue
            found.append(idx)
            if len(found) >= min_repeats:
                span_start = found[0]
                span_end = found[-1] + size
                repeated_coverage = len(found) * size / max(1, span_end - span_start)
                if repeated_coverage >= 0.6:
                    return span_start
    return None


def _strip_repeated_hallucination_loop(text):
    matches = _word_matches(text)
    words = [m.group(0).casefold().strip("'’-") for m in matches]
    if len(words) < 12:
        return text
    lexical_diversity = len(set(words)) / len(words)
    if len(words) >= 18 and lexical_diversity <= 0.35:
        return ""
    loop_start = _repeated_ngram_start(words)
    if loop_start is None:
        return text
    if loop_start < 8 or loop_start / len(words) <= 0.35:
        return ""
    return text[: matches[loop_start].start()].rstrip(" ,.;:!?…-—")


def strip_hallucinations(text):
    """Remove standalone Whisper boilerplate hallucinations."""
    text = _strip_repeated_hallucination_loop(text)
    if not text:
        return ""
    cleaned = _HALLUCINATION_RE.sub(" ", text)
    cleaned = re.sub(r"\s+([,.;:!?…])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _strip_repeated_hallucination_loop(cleaned)
    return cleaned.strip(" ,.;:!-—")


def strip_llm_noise(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?is)^\s*thinking process:.*?(final answer:|answer:)", "", text)
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip().strip('"')


def sentence_case_text(text):
    chars = list(text)
    should_capitalize = True
    for idx, char in enumerate(chars):
        if not char.isalpha():
            if char in ".!?":
                should_capitalize = True
            continue
        if should_capitalize:
            chars[idx] = char.upper()
            should_capitalize = False
        else:
            should_capitalize = False
    return "".join(chars)


def _word_matches(text):
    return list(re.finditer(r"[\w'’-]+", text, flags=re.UNICODE))


def _normalized_words(text):
    return [m.group(0).casefold().strip("'’-") for m in _word_matches(text)]


def _common_prefix_len(a, b):
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def _same_token(a, b):
    """Whether two normalized words are the same word, allowing a re-transcribed ending
    from a chunk-boundary cut (e.g. 'энерго'/'энергию', 'невероятной'/'невероятный',
    'соотношения'/'соотношение'). Requires a strong shared stem so distinct words
    ('стол'/'стул', 'красная'/'красивая') are not collapsed."""
    if a == b:
        return True
    shorter = min(len(a), len(b))
    if shorter < 4:
        return False
    return _common_prefix_len(a, b) >= max(4, int(shorter * 0.6) + 1)


def _drop_prefix_words(text, word_count):
    end = 0
    for idx, match in enumerate(_word_matches(text), 1):
        if idx == word_count:
            end = match.end()
            break
    return text[end:].lstrip(" ,.;:!?…-—")


def _sentence_end_matches(text):
    return list(re.finditer(r"(?<!\.)[.!?](?!\.)(?:[\"')\]]+)?(?=\s|$)|…(?:[\"')\]]+)?(?=\s|$)", text))


def merge_overlap_text(previous_tail, incoming, min_overlap=12, max_overlap=120):
    incoming = re.sub(r"\s+", " ", incoming).strip()
    previous_tail = re.sub(r"\s+", " ", previous_tail).strip()
    if not previous_tail or not incoming:
        return incoming

    prev_words = _normalized_words(previous_tail)
    incoming_words = _normalized_words(incoming)
    if incoming_words and " ".join(incoming_words) in " ".join(prev_words):
        return ""

    # Largest word overlap where the leading words match exactly and the boundary (last)
    # word may be a re-transcribed stem variant — the common chunk-seam duplication.
    max_words = min(len(prev_words), len(incoming_words), 28)
    for size in range(max_words, 0, -1):
        prev_slice = prev_words[-size:]
        inc_slice = incoming_words[:size]
        if prev_slice[:-1] != inc_slice[:-1]:
            continue
        if not _same_token(prev_slice[-1], inc_slice[-1]):
            continue
        # A lone boundary word with no preceding anchor is only safe to drop when it's a
        # solid word (>=4 chars); otherwise short coincidental repeats would be eaten.
        if size == 1 and min(len(prev_slice[0]), len(inc_slice[0])) < 4:
            continue
        return _drop_prefix_words(incoming, size)

    prev_lower = previous_tail.lower()
    incoming_lower = incoming.lower()
    max_len = min(len(prev_lower), len(incoming_lower), max_overlap)
    for size in range(max_len, min_overlap - 1, -1):
        if prev_lower[-size:] == incoming_lower[:size]:
            return incoming[size:].lstrip()
    return incoming


def merge_partial_buffer(buffer, incoming):
    buffer = re.sub(r"\s+", " ", buffer).strip()
    incoming = re.sub(r"\s+", " ", incoming).strip()
    if not buffer:
        return incoming
    if not incoming:
        return buffer

    buffer_words = _normalized_words(buffer)
    incoming_words = _normalized_words(incoming)
    if len(incoming_words) >= 8:
        window = " ".join(incoming_words[: min(10, len(incoming_words))])
        joined_buffer = " ".join(buffer_words)
        if window and window in joined_buffer:
            prefix = joined_buffer.split(window, 1)[0].split()
            keep_words = len(prefix)
            matches = _word_matches(buffer)
            cut = matches[keep_words].start() if keep_words < len(matches) else len(buffer)
            return f"{buffer[:cut].strip()} {incoming}".strip()

    merged = merge_overlap_text(buffer[-500:], incoming, min_overlap=8, max_overlap=160)
    if not merged:
        return buffer
    return f"{buffer} {merged}".strip()


def split_complete_sentences(buffer, min_sentence_chars):
    text = re.sub(r"\s+", " ", buffer).strip()
    if not text:
        return [], ""

    matches = _sentence_end_matches(text)
    if not matches:
        return [], text

    cut = matches[-1].end()
    complete = text[:cut].strip()
    rest = text[cut:].strip()
    sentences = []
    start = 0
    for match in _sentence_end_matches(complete):
        sentence = complete[start : match.end()].strip()
        start = match.end()
        if len(sentence) >= min_sentence_chars:
            sentences.append(sentence)
        elif sentences:
            sentences[-1] = f"{sentences[-1]} {sentence}".strip()
        elif sentence:
            sentences.append(sentence)
    return sentences, rest


def take_sentences_for_translation(buffer, min_sentence_chars, max_sentences, force=False):
    sentences, rest = split_complete_sentences(buffer, min_sentence_chars)
    if force and not sentences and buffer.strip():
        return [buffer.strip()], ""
    if not sentences:
        return [], rest
    selected = sentences[:max_sentences]
    remaining_sentences = sentences[max_sentences:]
    remaining = " ".join(remaining_sentences + ([rest] if rest else [])).strip()
    return selected, remaining


def take_blocks_for_translation(buffer, min_chars, max_chars, max_sentences, force=False):
    sentences, rest = split_complete_sentences(buffer, min_sentence_chars=1)
    if force and not sentences:
        return [], buffer.strip()

    blocks = []
    current = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        should_flush = (
            current
            and (
                current_len >= min_chars
                or len(current) >= max_sentences
                or current_len + sentence_len + 1 > max_chars
            )
        )
        if should_flush:
            blocks.append(" ".join(current).strip())
            current = []
            current_len = 0
        current.append(sentence)
        current_len += sentence_len + 1

    if current and (current_len >= min_chars or force):
        blocks.append(" ".join(current).strip())
        current = []

    remaining = " ".join(current + ([rest] if rest else [])).strip()
    return blocks, remaining


def take_confirmed_blocks_for_translation(buffer, min_chars, max_chars, max_sentences):
    """Emit complete sentence blocks, but hold back the newest terminal sentence."""
    sentences, rest = split_complete_sentences(buffer, min_sentence_chars=1)
    if not sentences:
        return [], rest
    held_back = ""
    if not rest:
        held_back = sentences.pop()
    if not sentences:
        return [], held_back
    blocks, remaining = take_blocks_for_translation(
        " ".join(sentences),
        min_chars=min_chars,
        max_chars=max_chars,
        max_sentences=max_sentences,
        force=True,
    )
    remaining = " ".join(part for part in (remaining, held_back, rest) if part).strip()
    return blocks, remaining


_SOFT_PUNCT = re.compile(r"[,;:—–](?=\s)")


def split_soft_boundaries(text, min_chars, max_chars):
    """Split a punctuation-less run into readable blocks without cutting words."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    target = min(max_chars, max(min_chars * 2, 200))
    blocks = []
    while len(text) > target:
        cut = None
        for match in _SOFT_PUNCT.finditer(text[: target + 1]):
            if match.end() >= min_chars:
                cut = match.end()
        if cut is None:
            space = text.rfind(" ", min_chars, target + 1)
            cut = space if space >= min_chars else target
        blocks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        blocks.append(text)
    return blocks


def take_endpoint_blocks(buffer, min_chars, max_chars, min_words):
    blocks, remaining = take_blocks_for_translation(
        buffer,
        min_chars=min_chars,
        max_chars=max_chars,
        max_sentences=100,
        force=True,
    )
    if blocks:
        return blocks, remaining

    text = re.sub(r"\s+", " ", buffer).strip()
    if len(text) >= min_chars or len(_normalized_words(text)) >= min_words:
        return split_soft_boundaries(text, min_chars, max_chars), ""
    return [], text


def punctuated_context(text, max_unpunctuated_tail=80):
    """Recent Whisper prompt context, disabled during punctuation drift."""
    text = re.sub(r"\s+", " ", text).strip()[-240:]
    if not text:
        return None
    matches = _sentence_end_matches(text)
    if not matches:
        return None
    tail = text[matches[-1].end():].strip()
    if len(tail) > max_unpunctuated_tail:
        return None
    return text


def last_word_end_seconds(result):
    last_end = None
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            end = word.get("end")
            if end is not None:
                last_end = float(end)
    return last_end


def absolute_words(result, offset_seconds):
    """Flatten a Whisper result into (start, end, text) tuples with timestamps shifted to
    the global timeline by offset_seconds (the chunk's start time)."""
    words = []
    base = float(offset_seconds or 0.0)
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            text = (word.get("word") or "").strip()
            if not text:
                continue
            start = base + float(word.get("start", 0.0))
            end = base + float(word.get("end", start))
            words.append((start, end, text))
    return words


def dedup_words_by_time(words, committed_until):
    """Drop words already covered by a previous overlapping chunk, by time rather than
    spelling. `words` is (start, end, text) tuples on the global timeline. A word is kept
    when its midpoint falls after `committed_until` (the end time of the last accepted
    word); duplicates re-transcribed in the overlap region sit before it and are dropped.
    Because it compares time, it removes overlaps that text matching misses — inflection
    changes ('непреодолимая'/'преодолимое') and words split across the seam.

    Returns (kept_text, new_committed_until). The boundary advances to the latest word end
    seen, so the next chunk dedups against full coverage."""
    kept = []
    max_end = committed_until
    for start, end, text in words:
        text = (text or "").strip()
        if not text:
            continue
        midpoint = (float(start) + float(end)) / 2.0
        if committed_until is None or midpoint > committed_until:
            kept.append(text)
        if max_end is None or end > max_end:
            max_end = float(end)
    return " ".join(kept), max_end
