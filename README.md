# Live Translation

A macOS app that captures any audio playing on your Mac — YouTube, Zoom, Twitch, livestreams, or any other app — transcribes it, translates it, and displays the original and translation side by side in a floating glass overlay.

Everything runs locally on Apple silicon. No API keys, accounts, or cloud services are required, and your audio never leaves the Mac.

<p align="center">
  <img src="docs/screenshot.png" alt="Live Translation overlay — original on the left, translation on the right" width="700">
</p>

## TL;DR

```text
system audio ──► Whisper ──► local LLM ──► glass overlay
   BlackHole       MLX        Ollama         PyObjC
```

- MLX Whisper for transcription
- Gemma 4 through Ollama for translation
- Silero VAD for speech detection
- PyObjC for the always-on-top overlay

## Who is this for?

- People watching foreign-language media without subtitles
- Language learners who want original speech and translation side by side
- Users attending calls or meetings in another language
- Anyone who needs private, fully offline live captions and translation

## How it is different

Most similar projects are cloud-based, transcription-only, file-based, tied to a meeting platform, or designed for OBS.

Live Translation combines:

- system-wide audio capture;
- real-time transcription;
- sentence-level LLM translation;
- a floating bilingual overlay;
- local processing;
- segmentation designed to avoid losing or repeating words.

## Quickstart

Requires macOS with Apple silicon.

```bash
git clone https://github.com/KazKozDev/live-translation.git
cd live-translation
./setup.sh
```

Or install manually:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
brew bundle --file=Brewfile
```

Use `requirements.lock.txt` if you need the exact pinned environment.

### Route system audio through BlackHole

macOS does not allow ordinary apps to capture system output directly, so Live Translation uses BlackHole 2ch.

To keep hearing audio while capturing it:

1. Open Audio MIDI Setup
2. Click **+** → **Create Multi-Output Device**
3. Enable your normal output and *BlackHole 2ch*
4. Select the Multi-Output Device in System Settings → Sound → Output

The first launch may request microphone permission. This is required to read BlackHole audio.

### Models

#### Whisper

Supported MLX Whisper models:

- `small` — lowest resource usage
- `medium` — balanced
- `turbo` — recommended for live use
- `large` — highest accuracy

Pre-download the default models:

```bash
./.venv/bin/python -c "from huggingface_hub import snapshot_download; \
[snapshot_download(r) for r in (
  'mlx-community/whisper-medium-mlx',
  'mlx-community/whisper-large-v3-turbo'
)]"
```

#### Gemma 4

Supported Ollama models:

```text
gemma4:26b-mlx
gemma4:e4b-mlx
gemma4:12b-mlx
```

Download them with:

```bash
for model in gemma4:26b-mlx gemma4:e4b-mlx gemma4:12b-mlx; do
    ollama pull "$model"
done
```

Recommended preset: **Whisper Turbo + Gemma 4 26B**

### Run

Launch `LiveTranslate.app`, or run:

```bash
./.venv/bin/python live_translate_overlay.py --target ru
```

Example with fixed languages and a different Whisper model:

```bash
./.venv/bin/python live_translate_overlay.py \
    --source es \
    --target en \
    --whisper large \
    --ollama-model gemma4:26b-mlx
```

### Install the app

```bash
./install-app.sh
```

Or choose another destination:

```bash
./install-app.sh ~/Apps
```

The app is ad-hoc signed. On first launch, right-click it and select **Open**.

## How it works

### Lossless segmentation

Fixed five-second chunks often make Whisper cut words, repeat phrases, or add false punctuation.

The default pipeline instead:

1. detects speech and pauses with Silero VAD;
2. creates overlapping speech segments;
3. removes duplicated text at segment boundaries;
4. fixes punctuation introduced by artificial pauses;
5. rebuilds complete sentences;
6. sends those sentences to the translation model.

Under normal load, the queue catches up instead of dropping speech. During severe overload, stale audio may be skipped to return to live output.

### Streaming mode

Enable it with `--streaming`. Streaming re-transcribes a rolling buffer and commits words after two consecutive passes agree. Use `--show-partial` to see the unconfirmed tail and `--update-seconds 1.5` to control re-transcription frequency.

It produces smoother partial captions but uses more compute and may lose content under load, so it is disabled by default.

### Hallucination reduction

Silero VAD removes silence and non-speech before transcription. The pipeline also filters common false outputs such as "thanks for watching", "subtitles by amara.org", and similar phrases. This reduces hallucinations but does not eliminate them completely.

### Sentence-level translation

The LLM receives complete sentences plus recent context, producing more coherent translations than word-by-word machine translation.

## Main options

```text
--source LANGUAGE
--target LANGUAGE
--whisper MODEL
--ollama-model MODEL
--silence-rms VALUE
--vad-min-speech-ms VALUE
--audio-queue SIZE
--show-partial
--streaming
--update-seconds VALUE
```

View all options:

```bash
./.venv/bin/python live_translate_overlay.py --help
```

## Limitations

- macOS and Apple silicon only
- BlackHole setup is required
- larger models increase latency and memory usage
- automatic language detection can be unstable with mixed-language audio
- streaming mode is more compute-demanding
- Whisper can still hallucinate occasionally

## Project structure

```text
live_translate_overlay.py   CLI, audio pipeline, and Cocoa overlay
live_translation/           segmentation, cleanup, and translation
LiveTranslate.app           macOS app bundle
install-app.sh              app installer
setup.sh / Brewfile         setup files
requirements*.txt           dependencies
tests/                      unit tests
```

## Development

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m ruff check .
./.venv/bin/python -m pyright
./.venv/bin/python -m pytest
```

## License

[MIT](LICENSE) — use, modify, and redistribute freely.

## Acknowledgements

Thanks to [Alex Ziskind](https://github.com/alexziskind1) for the [benchmark](https://www.youtube.com/watch?v=PxUSE2KwyUQ) that pointed to the MLX Whisper path.
