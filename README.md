# Live Translation

A macOS app that listens to **any sound your Mac is playing — from any app** — transcribes it, translates it, and paints both the original and the translation onto a floating glass overlay, live, as people talk.

<p align="center">
  <img src="docs/screenshot.png" alt="Live Translation overlay — original on the left, translation on the right" width="700">
</p>

## TLDR

```
system audio ──► whisper (speech→text) ──► local LLM (translate) ──► glass overlay
   (BlackHole)     (mlx, small/medium/turbo/large) (Ollama Gemma 4)     (pyobjc)
```

Under the hood: [MLX](https://github.com/ml-explore/mlx) Whisper for transcription, Ollama Gemma 4 for translation, Silero VAD to throw away silence. The UI is a translucent always-on-top window drawn with pyobjc.

## Who's this for

- You watch **foreign-language media** — news, YouTube, documentaries, films, Twitch — and want live captions and translation.
- You're **learning a language** and want the original and a translation side by side on real native speech.
- You sit in **calls / meetings / webinars** in a language you only half-speak.
- You want **captions for accessibility** on audio that has none.
- You need **fully offline, private** transcription — audio never leaves your Mac, no API key or account required.

## What makes this different

Most "whisper + something" projects are either cloud-backed, transcription-only, OBS plugins for streaming to others, meeting bots tied to a specific call, or word-by-word MT that produces choppy output.

This combines: **any app's audio** (system-wide, not file/URL/one-call) + **a real LLM doing sentence-level translation** + **a floating glass overlay** you read on top of anything + **fully local** + a deliberate **don't-lose-words** design.

## Quickstart

You need a Mac with Apple silicon. The fast path:

```bash
./setup.sh        # python deps + BlackHole + (optional) ollama + downloads the models
```

That does everything below. If you'd rather understand the pieces, here they are.

### 1. Install the Python + system deps

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
brew bundle --file=Brewfile      # installs BlackHole + ffmpeg + ollama
```

If dependency resolution changes and the install breaks, use `requirements.lock.txt` for the exact pinned environment.

### 2. Capturing system audio — why you need BlackHole

macOS won't let an app grab "whatever is coming out of the speakers," so we need [**BlackHole**](https://github.com/ExistentialAudio/BlackHole) — a free virtual audio driver that loopbacks output → input. The `2ch` variant is what we use.

To keep hearing audio yourself, create a **Multi-Output Device** in Audio MIDI Setup that sends to both your normal output and BlackHole 2ch, then set it as your system output (System Settings → Sound → Output). Now you hear everything normally and the app hears it too.

### 3. The models

- **Whisper medium + turbo** (transcription) — MLX, pulled automatically from Hugging Face on first run. (`small` and `large` download on the fly if selected in settings.)
- **Gemma 4 via Ollama** (translation) — choose `gemma4:26b-mlx`, `gemma4:e4b-mlx`, or `gemma4:12b-mlx` from in-app settings.
- **Silero VAD** ships inside the `silero-vad` pip package.

To pre-download instead of waiting on first launch:

```bash
./.venv/bin/python -c "from huggingface_hub import snapshot_download; \
[snapshot_download(r) for r in ('mlx-community/whisper-medium-mlx', 'mlx-community/whisper-large-v3-turbo')]"

for model in gemma4:26b-mlx gemma4:e4b-mlx gemma4:12b-mlx; do ollama pull "$model"; done
```

### 4. Run

**a) Double-click `LiveTranslate.app`** — launch from Finder (see step 5 to move it to /Applications).

**b) From the terminal:**

```bash
./.venv/bin/python live_translate_overlay.py --target ru
```

The defaults match the app launcher (`--whisper turbo`, `--ollama-model gemma4:26b-mlx`). To tweak:

```bash
./.venv/bin/python live_translate_overlay.py --source es --whisper large --target ru
```

See [Knobs](#knobs) or `--help` for the full list.

### 5. (Optional) put the app in /Applications

`LiveTranslate.app` works two ways:

- **portable** (default) — keep the `.app` where it is and double-click it. The launcher finds the project relative to itself, so copying the whole folder to another location or Mac just works.
- **installed** — to have the app in `/Applications` (Launchpad, Spotlight, Dock):

  ```bash
  ./install-app.sh            # copies to /Applications, or: ./install-app.sh ~/Apps
  ```

  This records the project's location so the detached app can find your venv + models. If you later move the project, run `./install-app.sh` again from its new spot.

**First launch on your Mac:** the app is ad-hoc signed (no Apple Developer ID), so Gatekeeper may block the first double-click. Right-click the app → Open and confirm once — macOS remembers afterward. You'll also get a one-time microphone prompt for BlackHole's loopback audio.

## The bar at the top

The window is two columns — original on the left, translation on the right — and a thin toolbar:

- **Source / Target** — the language you're listening to and the one you want. Leave Source on *Auto* and Whisper figures it out per sentence (pin it if the audio mixes languages).
- **Compact** — hide the original, show only the translation in one wide column.
- **Pin** — keep the window floating above everything (on by default).
- **A- / A+** — font size.
- **Color** — text color (white, graphite, warm/cool tints).
- **lag: N** — audio chunks waiting. Zero-ish = keeping up live; if it climbs and stays up, try `--whisper small`.
- 🗑 **(trash)** — wipe both columns, the history, and all model state behind them.
- **Save** — dump the whole session as TXT, PDF, or timestamped subtitles (SRT / VTT).
- **History** — every session is auto-saved to `~/Library/Application Support/LiveTranslate`; reopen or rename past ones from here.
- **⚙ (settings)** — switch the Whisper size and Gemma model live, without restarting.

The window is draggable and resizable; controls on the right tuck away if you make it narrow.

## How it works

The naive approach — fixed 5 s chunks, each transcribed independently — is bad. Whisper sticks a period at the end of every chunk, cuts words at boundaries, and repeats itself. Most of the interesting work is avoiding that.

- **Lossless chunker (default).** Audio is cut at natural pauses (VAD), not fixed sizes, so words don't get sliced. Each chunk is transcribed exactly once with overlap; overlap-dedup merges seams, a sentence assembler regroups on real punctuation, and a spurious-period stripper removes the period Whisper adds when a speaker only paused for breath. Under normal load the pipeline catches up; under severe overload it skips stale raw audio and jumps to live rather than stalling forever.

- **Streaming mode (`--streaming`), off by default.** A rolling buffer is re-transcribed every ~1.5 s; a word is committed once two consecutive passes agree (LocalAgreement-2). Text flows and self-corrects instead of appearing in blocks, but re-transcribing the same audio repeatedly is expensive — under load it either drops audio or trims its buffer. Use it paired with a faster model.

- **Surviving Whisper's punctuation drift.** On fast, pause-less speech Whisper sometimes stops emitting punctuation; because recently finalized text is fed back as `initial_prompt`, a punctuation-less run reinforces itself. Two guards: (1) prompt feedback is dropped when recent context has no sentence punctuation; (2) when a block arrives with no punctuation, it's split at clause boundaries or word boundaries near a target length — never one wall, never mid-word.

- **VAD before Whisper.** Whisper invents speech out of silence and music — it was trained on YouTube subtitles, so on non-speech it confidently emits "thanks for watching." Silero VAD runs first and blocks chunks without enough real speech. This eliminated most hallucinations and fixed stalls caused by long pauses between videos.

- **A hallucination blocklist** for the ones that slip through ("gracias", "subtitles by amara.org", etc.), stripped before translation.

- **Translation by a real LLM.** The LLM gets whole sentences plus recently-translated context, producing coherent output instead of word salad. Translation runs in the same worker thread that loaded the model — MLX uses a thread-local GPU stream, so loading on one thread and generating on another silently explodes on long prompts. Lazy-load on first use fixed it.

## The models

**Transcription: Whisper on MLX.** Pick `small`, `medium`, `turbo`, or `large` in settings. `turbo` is the default (faster large-v3 variant, gives the live pipeline more headroom while staying much stronger than small models); use `large` when accuracy matters more than latency. MLX is the fastest way to run Whisper on a Mac — Apple's own array framework, running on the GPU through Metal with unified memory, beats CPU-bound alternatives for this workload.

**Translation** uses **Gemma 4 via Ollama** — switch between the three sizes in settings.

Best overall quality: `--whisper turbo` with `--ollama-model gemma4:26b-mlx`.

## Knobs

```
--source / --target        languages (or pick in the UI). source=auto detects per-utterance
--whisper turbo            MLX Whisper size: small, medium, turbo, or large
--ollama-model MODEL       Gemma 4 Ollama model
--silence-rms 0.006        louder = stricter silence gate
--vad-min-speech-ms 250    min real speech per window before whisper sees it
--audio-queue 120          raw audio backlog before old audio is skipped to recover
--show-partial             also show the unfinalized live draft (off by default)
# lossless chunker is the default; opt into streaming mode (smoother, but lossy under load):
--streaming                LocalAgreement streaming instead of the lossless chunker
--update-seconds 1.5       how often to re-transcribe the rolling buffer (lower = snappier, heavier)
```

`./.venv/bin/python live_translate_overlay.py --help` for the full list (~30 flags).

## Robustness

- **Audio thread silently dying.** Switch YouTube videos, CoreAudio re-inits the route, the input callback goes quiet forever. A watchdog now notices the silence and reopens the stream.
- **Live-vs-complete tradeoff.** Dropping the oldest audio to stay live is the wrong call when you care about every word. The pipeline runs late and catches up; it skips stale audio only as a last-resort recovery.
- **The Clear button only clearing the screen.** "Clear" needs to wipe the rolling audio buffer, both worker threads' state, the pending queues, and the chunk mid-flight inside Whisper at that moment. All of that now resets from one generation counter.
- **Workers dying on a single bad frame.** One unguarded exception in the audio loop silences the pipeline with nothing in the log. Everything is wrapped and logs `[chunk]`/`[audio]`/`[stream]`/`[ui]` so the next stall names itself.
- **Shipping as a double-clickable app.** macOS LaunchServices won't run a `.app` whose executable is a shell script. The bundle's executable is a tiny compiled C launcher that execs the real script, the whole thing is ad-hoc codesigned, and only then does double-click work.

## Honest caveats

- macOS + Apple silicon only. The overlay is pyobjc/Cocoa, the inference is MLX.
- You have to route audio through BlackHole — see setup above.
- Whisper is not free. Drop to `--whisper small` if the default turbo can't keep up, or use `--whisper large` when accuracy matters more than headroom. Streaming mode is heavier still.
- Auto language detection wobbles on mixed-language audio. Pin `--source` if you know it.
- It still hallucinates sometimes. We mitigate, we don't cure.
- The Cocoa overlay and audio workers are intentionally compact; the text pipeline and translation backends are split out and covered by tests.

## Layout

```
live_translate_overlay.py   CLI + orchestration + Cocoa overlay/audio workers
live_translation/           tested text pipeline + translation backends
LiveTranslate.app           double-clickable bundle (launches the script via your venv)
install-app.sh              put the .app in /Applications, detached from the project folder
setup.sh / Brewfile         install everything
requirements.txt            direct deps for normal install
requirements.lock.txt       full pinned environment for exact reproduction
requirements-dev.txt        lint/type/test tooling
tests/                      fast unit tests for segmentation/cleanup behavior
```

## Developer checks

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m ruff check .
./.venv/bin/python -m pyright
./.venv/bin/python -m pytest
```

## License

[MIT](LICENSE) — personal hack, take it and do whatever. PRs welcome.

## Thanks

Thanks to [Alex Ziskind](https://github.com/alexziskind1) for the [benchmark video on the fastest high-quality way to transcribe on a Mac](https://www.youtube.com/watch?v=PxUSE2KwyUQ). I used that result to choose the MLX Whisper path, which made this pipeline practical.
