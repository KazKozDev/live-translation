# Live Translation

A macOS app that captures any audio your Mac plays — YouTube, Zoom, Twitch, a foreign livestream — transcribes it, translates it, and shows both the original and the translation on a floating glass overlay, live. Real-time transcription + translation for the whole machine, fully offline.

<p align="center">
  <img src="docs/screenshot.png" alt="Live Translation overlay — original on the left, translation on the right" width="700">
</p>

## TLDR

```
system audio ──► whisper (speech→text) ──► local LLM (translate) ──► glass overlay
   (BlackHole)     (mlx: small/med/turbo/large)   (Ollama Gemma 4)       (pyobjc)
```

MLX Whisper for transcription, Ollama Gemma 4 for translation, Silero VAD to drop silence. The UI is a translucent always-on-top window drawn with pyobjc. No API key, no account, no cloud — the audio never leaves your Mac.

## Who's this for

- You watch foreign-language media and want live captions + translation instead of waiting for subtitles.
- You're learning a language and want the original and translation side by side on real speech.
- You sit in calls/meetings in a language you only half-speak.
- You need captions and translation for accessibility on audio that has none.
- You want it fully offline and private.

## How it's different

Most "whisper + something" projects are either cloud-based (your audio leaves the machine), transcription-only (no translation, often file-based), OBS/streamer plugins (built for broadcasting), meeting bots (one specific call via an account), or word-by-word MT (choppy output). This combines all of: any app's system audio + a real LLM doing sentence-level translation + a floating overlay + fully local + a deliberate don't-lose-words design.

## Quickstart

Requires a Mac with Apple silicon. The fast path:

```bash
./setup.sh   # python deps + BlackHole + (optional) ollama + downloads the models
```

If you'd rather do it by hand:

### 1. Install deps

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
brew bundle --file=Brewfile   # BlackHole + ffmpeg + ollama
```

If installs break in the future, use `requirements.lock.txt` for the exact pinned environment.

### 2. Route system audio through BlackHole

macOS won't let an app grab the system output directly, so we use [BlackHole](https://github.com/ExistentialAudio/BlackHole) (a free virtual audio driver, 2ch variant) to loopback output → input. To still hear audio yourself, create a Multi-Output Device:

1. Open **Audio MIDI Setup** (`/Applications/Utilities`)
2. Click **+** → **Create Multi-Output Device**
3. Tick both your normal output and *BlackHole 2ch*
4. Set it as your system output (System Settings → Sound → Output)

### 3. Models

Downloaded on first run, or up front by `setup.sh`:

- **Whisper medium + turbo** (MLX, from Hugging Face). `small`/`large` download on the fly if selected.
- **Gemma 4 via Ollama** — pick `gemma4:26b-mlx`, `gemma4:e4b-mlx`, or `gemma4:12b-mlx` in settings.
- **Silero VAD** ships in the `silero-vad` package — nothing to download.

To pre-fetch:

```bash
./.venv/bin/python -c "from huggingface_hub import snapshot_download; \
[snapshot_download(r) for r in ('mlx-community/whisper-medium-mlx', 'mlx-community/whisper-large-v3-turbo')]"

for model in gemma4:26b-mlx gemma4:e4b-mlx gemma4:12b-mlx; do ollama pull "$model"; done
```

### 4. Run

Double-click `LiveTranslate.app`, or from the terminal:

```bash
./.venv/bin/python live_translate_overlay.py --target ru
```

The defaults match the app launcher (lossless chunker, `--whisper turbo`, `--ollama-model gemma4:26b-mlx`). Override anything: `--source es --whisper large --streaming`, etc. Then pick source/target in the overlay's top bar, play audio in any app, and text appears.

### 5. (Optional) install to /Applications

```bash
./install-app.sh            # or: ./install-app.sh ~/Apps
```

The app is ad-hoc signed (no Apple Developer ID), so on first launch right-click → Open once to get past Gatekeeper. Allow the one-time microphone prompt (that's BlackHole's loopback).

## How it works

Chopping audio into fixed 5 s chunks and transcribing each independently is bad — Whisper adds a period per chunk, cuts words at boundaries, and repeats itself. The interesting work is avoiding that:

- **Lossless segmentation (default).** Audio is cut at natural pauses (via VAD), with overlap between chunks. An overlap-dedup merges seams, a sentence assembler regroups on real punctuation, and a spurious-period stripper removes the period Whisper adds on a breath pause. Under normal load the pipeline catches up rather than dropping speech; only under severe overload does it skip stale audio to jump back to live.

- **Streaming mode (off by default).** A LocalAgreement-2 mode re-transcribes a rolling buffer every ~1.5 s and commits a word once two passes agree (`--show-partial` shows the unconfirmed tail). Reads smoothly but is expensive and lossy under load — hence behind a flag.

- **Punctuation-drift guards.** On fast speech Whisper sometimes stops emitting punctuation, and since recent text feeds back as `initial_prompt`, it self-reinforces. Guards: drop the prompt feedback when recent context has no sentence punctuation, and split punctuation-less blocks at clause/word boundaries.

- **VAD before Whisper.** Silero VAD filters non-speech so Whisper doesn't hallucinate ("thanks for watching") on silence or music.

- **Hallucination blocklist** for the ones that slip through ("gracias", "subtitles by amara.org", etc.).

- **LLM translation, not word-by-word MT.** The LLM gets whole sentences plus recent context, producing coherent output. It runs in the same thread that loaded the model (MLX uses a thread-local GPU stream).

## The models

**Transcription: Whisper on MLX** — `small`, `medium`, `turbo`, or `large`. `turbo` is the default (faster large-v3 variant, good live headroom); use `large` when accuracy matters more than latency. MLX runs on the GPU via Metal with unified memory, beating faster-whisper/whisper.cpp on Apple silicon.

**Translation: Gemma 4 via Ollama** — switch between three sizes in settings. For best quality: `--whisper turbo` + `--ollama-model gemma4:26b-mlx`.

## Knobs

```
--source / --target      languages (source=auto detects per-utterance)
--whisper turbo          MLX Whisper size: small, medium, turbo, large
--ollama-model MODEL     Gemma 4 Ollama model
--silence-rms 0.006      louder = stricter silence gate
--vad-min-speech-ms 250  min real speech before whisper sees it
--audio-queue 120        raw audio backlog before old audio is skipped
--show-partial           show the unfinalized live draft
--streaming              LocalAgreement streaming instead of the lossless chunker
--update-seconds 1.5     how often to re-transcribe the rolling buffer
```

`--help` for the full list (~30 flags, most rarely needed).

## Caveats

- macOS + Apple silicon only (pyobjc/Cocoa overlay, MLX inference).
- You must route audio through BlackHole yourself.
- Whisper isn't free; drop to `--whisper small` if it can't keep up, `--whisper large` for accuracy. Streaming is heavier.
- Auto language detection wobbles on mixed-language audio — pin `--source` if you know it.
- It still hallucinates sometimes. We mitigate, not cure.

## Layout

```
live_translate_overlay.py   CLI + orchestration + Cocoa overlay/audio workers
live_translation/           tested text pipeline + translation backends
LiveTranslate.app           double-clickable bundle
install-app.sh              install the .app to /Applications
setup.sh / Brewfile         install everything
requirements*.txt           deps (direct / locked / dev)
tests/                      unit tests for segmentation/cleanup
```

## Developer checks

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m ruff check .
./.venv/bin/python -m pyright
./.venv/bin/python -m pytest
```

## License

[MIT](LICENSE) — take it and do whatever. PRs welcome, no promises about keeping this tidy.

## Thanks

Thanks to [Alex Ziskind](https://github.com/alexziskind1) for the [benchmark](https://www.youtube.com/watch?v=PxUSE2KwyUQ) that pointed me at the MLX Whisper path.
