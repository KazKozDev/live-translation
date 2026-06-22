from live_translation.text_pipeline import (
    last_word_end_seconds,
    merge_overlap_text,
    merge_partial_buffer,
    punctuated_context,
    sentence_case_text,
    split_complete_sentences,
    strip_hallucinations,
    strip_llm_noise,
    take_blocks_for_translation,
    take_confirmed_blocks_for_translation,
    take_endpoint_blocks,
)


def test_strip_hallucinations_removes_standalone_boilerplate():
    assert strip_hallucinations("Gracias. Thanks for watching!") == ""
    assert strip_hallucinations("The speaker said gracias for the answer.") == (
        "The speaker said for the answer"
    )


def test_strip_hallucinations_removes_repeated_russian_loop():
    text = (
        "ведущий закончил подробное объяснение бюджета и перешел к следующей теме, "
        "красный сигнал снова горит, "
        "красный сигнал снова горит, красный сигнал снова горит"
    )
    assert strip_hallucinations(text) == (
        "ведущий закончил подробное объяснение бюджета и перешел к следующей теме"
    )


def test_strip_hallucinations_removes_repeated_english_loop():
    text = (
        "blue window keeps moving, blue window keeps moving, blue window keeps moving"
    )
    assert strip_hallucinations(text) == ""


def test_strip_hallucinations_drops_short_prefix_before_loop():
    text = (
        "quick preface, silver frame turns left, silver frame turns left, "
        "silver frame turns left"
    )
    assert strip_hallucinations(text) == ""


def test_strip_hallucinations_keeps_non_repeated_channel_sentence():
    text = "The analyst discussed how the channel grew after a public subscription campaign."
    assert strip_hallucinations(text) == text.rstrip(".")


def test_strip_llm_noise_removes_reasoning_and_code_fences():
    assert strip_llm_noise("<think>hidden</think>```text\nHallo\n```") == "Hallo"


def test_sentence_case_text_capitalizes_sentence_starts():
    assert sentence_case_text("hello world. second thought? yes.") == (
        "Hello world. Second thought? Yes."
    )


def test_merge_overlap_text_removes_repeated_prefix_words():
    previous = "the prime minister spoke about the new budget today"
    incoming = "new budget today and promised another vote"
    assert merge_overlap_text(previous, incoming) == "and promised another vote"


def test_merge_partial_buffer_replaces_repeated_window():
    buffer = "one two three four five six seven eight nine ten eleven"
    incoming = "six seven eight nine ten eleven twelve thirteen fourteen"
    assert merge_partial_buffer(buffer, incoming) == (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    )


def test_split_complete_sentences_keeps_incomplete_tail():
    sentences, rest = split_complete_sentences("One. Two! Three still going", 1)
    assert sentences == ["One.", "Two!"]
    assert rest == "Three still going"


def test_take_blocks_for_translation_respects_sentence_limit():
    blocks, remaining = take_blocks_for_translation(
        "One sentence. Two sentence. Three sentence.",
        min_chars=100,
        max_chars=100,
        max_sentences=2,
        force=True,
    )
    assert blocks == ["One sentence. Two sentence.", "Three sentence."]
    assert remaining == ""


def test_take_blocks_force_emits_latest_complete_sentence():
    blocks, remaining = take_blocks_for_translation(
        "One complete sentence.",
        min_chars=100,
        max_chars=200,
        max_sentences=5,
        force=True,
    )
    assert blocks == ["One complete sentence."]
    assert remaining == ""


def test_take_blocks_force_keeps_unpunctuated_tail():
    blocks, remaining = take_blocks_for_translation(
        "This is not finished yet",
        min_chars=1,
        max_chars=200,
        max_sentences=5,
        force=True,
    )
    assert blocks == []
    assert remaining == "This is not finished yet"


def test_take_confirmed_blocks_holds_newest_terminal_sentence():
    blocks, remaining = take_confirmed_blocks_for_translation(
        "Earlier complete sentence. Newest likely chunk punctuation.",
        min_chars=1,
        max_chars=200,
        max_sentences=5,
    )
    assert blocks == ["Earlier complete sentence."]
    assert remaining == "Newest likely chunk punctuation."


def test_endpoint_blocks_emit_punctuationless_utterance():
    text = " ".join(f"word{i}" for i in range(20))
    blocks, remaining = take_endpoint_blocks(text, min_chars=40, max_chars=80, min_words=8)
    assert blocks
    assert remaining == ""
    assert "word0" in blocks[0]


def test_punctuated_context_drops_long_unpunctuated_tail():
    assert punctuated_context("A stable sentence. " + "tail " * 30) is None
    assert punctuated_context("A stable sentence. short tail") == "A stable sentence. short tail"


def test_last_word_end_seconds_reads_nested_whisper_result():
    result = {
        "segments": [
            {"words": [{"end": 1.25}, {"end": 2.5}]},
            {"words": [{"start": 3.0}, {"end": 4}]},
        ]
    }
    assert last_word_end_seconds(result) == 4.0
