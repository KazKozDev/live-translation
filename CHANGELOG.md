# Changelog

## Unreleased

- **All Whisper languages in the menus**: the From/To pickers now list every language Whisper can transcribe (Japanese, Korean, Arabic, Hindi, …) instead of nine.
- **Setup preflight**: `setup.sh` checks for Apple silicon, macOS 14+ and Python 3.12+ up front and explains what is missing, instead of failing inside pip.

## 1.1.0

Faster, smoother transcription and translation — and cleaner text at chunk boundaries.

### Speed
- **Whisper reads audio in memory** instead of writing a temp WAV every pass, skipping the file write + ffmpeg decode + resample on each transcription.
- **Shorter decode fallback** (`temperature=(0.0, 0.2)`) on the chunked path caps worst-case re-decoding on hard speech.
- **Single resample per pass**, shared between the Silero VAD gate and Whisper in streaming mode.
- **No re-fed overlap on clean cuts**: when VAD cuts on a silence boundary, the next chunk starts fresh — less audio per chunk and no boundary duplication.

### Translation
- **Live streaming translation**: the translation now appears word-by-word as it's generated instead of all at once when the block finishes.
- **Cross-block context**: recent source→translation pairs are fed back to the translator so terminology, names and pronouns stay consistent across sentences.
- **Ellipsis phrases merge forward**: a phrase ending in `…` is glued to the following sentence before translating, so trailing-off thoughts are translated as a whole instead of as a dangling fragment.

### Transcription quality (all languages)
- **Time-based seam dedup** using Whisper word timestamps removes overlap duplicates regardless of spelling — catching inflection changes and words split across the seam that text matching misses.
- **Smarter text dedup** also collapses short and inflected repeats at chunk seams (e.g. repeated short words, or the same word re-transcribed with a different ending).

### Housekeeping
- The app's boot log is trimmed to the last 2000 lines on each launch, so it can no longer grow without bound.

Full commit history: https://github.com/KazKozDev/live-translation/commits/main
