from live_translation.sessions import (
    SessionStore,
    export_subtitles,
    export_txt,
    new_session,
    phrase_from_pair,
    write_session_exports,
)


def test_session_store_persists_phrases_and_duration(tmp_path):
    store = SessionStore(tmp_path)
    session = new_session(source_language="en", target_language="ru")
    session["phrases"].append(
        phrase_from_pair(
            source="hello",
            translated="привет",
            start=1.25,
            end=2.75,
            language="en",
            confidence=0.91,
        )
    )

    saved = store.save(session)
    loaded = store.load(saved["id"])

    assert loaded["duration_seconds"] == 2.75
    assert loaded["phrases"] == [
        {
            "confidence": 0.91,
            "end": 2.75,
            "language": "en",
            "source": "hello",
            "start": 1.25,
            "translated": "привет",
        }
    ]


def test_subtitle_exports_original_and_translation_timestamps():
    session = new_session()
    session["phrases"] = [
        phrase_from_pair(source="hello", translated="привет", start=0, end=1.234),
        phrase_from_pair(source="world", translated="мир", start=61.5, end=62.0),
    ]

    assert export_subtitles(session, field="source", fmt="srt") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,234\n"
        "hello\n\n"
        "2\n"
        "00:01:01,500 --> 00:01:02,000\n"
        "world\n"
    )
    assert export_subtitles(session, field="translated", fmt="vtt").startswith(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.234\nпривет"
    )


def test_write_session_exports_creates_text_and_subtitle_sidecars(tmp_path):
    session = new_session()
    session["phrases"] = [
        phrase_from_pair(source="one", translated="один", start=0.0, end=1.0),
    ]

    written = write_session_exports(session, tmp_path / "meeting", formats=("txt", "srt", "vtt"))
    names = sorted(path.name for path in written)

    assert names == [
        "meeting.original.srt",
        "meeting.original.vtt",
        "meeting.translation.srt",
        "meeting.translation.vtt",
        "meeting.txt",
    ]
    assert "TRANSCRIPTION" in export_txt(session)
