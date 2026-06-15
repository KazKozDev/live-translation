# live translation

a little macOS app that listens to **any sound your mac is playing — from any app** — transcribes it, translates it, and paints both the original and the translation onto a floating glass overlay, live, as people talk.

<p align="center">
  <img src="docs/screenshot.png" alt="Live Translation overlay — original on the left, translation on the right" width="700">
</p>

that's the whole idea: it doesn't care *where* the audio comes from. a youtube video in your browser, a zoom call, vlc, a podcast, a twitch stream, a foreign news livestream, a game — if your mac can play it through its speakers, this can caption it and translate it into a language you actually read. real-time subtitles for the entire machine, running **entirely on your machine**. nothing gets uploaded anywhere.

i built it to watch foreign news (spanish/catalan, in my case) without waiting for anyone to add subtitles. it grew a bunch of knobs from there.

it's a hack. it works surprisingly well. it also occasionally hallucinates the word "gracias" out of thin air. read on.

## TLDR

```
system audio ──► whisper (speech→text) ──► local LLM (translate) ──► glass overlay
   (BlackHole)        (mlx, large-v3)         (mlx, Qwen3.5-9B)         (pyobjc)
```

everything is local: [MLX](https://github.com/ml-explore/mlx) Whisper for transcription, a local LLM (MLX Qwen by default) for translation, Silero VAD to throw away silence. the UI is a translucent always-on-top window drawn with pyobjc.

## who's this for

honestly i built it for me, but if any of these is you, it'll probably help:

- you watch **foreign-language media** — news, youtube, documentaries, films, twitch — and you're tired of waiting for (or doing without) subtitles.
- you're **learning a language** and want to see the original *and* a translation side by side, live, on real native speech instead of textbook audio.
- you sit in **calls / meetings / webinars** in a language you only half-speak.
- you want **captions for accessibility** on audio that has none.
- you just want to know **what that video is actually saying** without copy-pasting anything anywhere.

and a hard requirement for some people: it's **fully offline and private**. the audio never leaves your mac. no API key, no account, no cloud. that matters if you're translating a confidential call or you just don't want your media diet logged somewhere.

## how it's different from the other repos

there are a lot of "whisper + something" projects on github. i looked. most fall into one of these, and this one sits in the gap between them:

- **cloud captioners / translators** (anything calling the OpenAI/Deepgram/Google APIs) — fast and accurate, but your audio leaves the machine and you pay per minute. this is 100% local.
- **transcription-only tools** (MacWhisper, Buzz, whisper.cpp GUIs) — great at speech→text, but they don't translate, and many want a *file*, not live system audio.
- **OBS / streamer plugins** (LocalVocal) — live and local, but built for broadcasting to *your* viewers, configured inside OBS, not a thing you pop open to read a video yourself.
- **meeting bots** — join a specific zoom/meet call via an account/bot; they don't caption arbitrary system audio from any app.
- **word-by-word MT** — translate token streams with a small MT model; cheaper, but you get choppy word-salad instead of sentences.

what this combines, which i didn't find in one place: **any app's audio** (system-wide, not file/URL/one-call) + **a real LLM doing sentence-level translation** (not word-by-word MT) + **a floating glass overlay** you read on top of anything + **fully local** + a deliberate **don't-lose-words** design. it's not the fastest or the prettiest; it's the one that translates the *whole machine*, locally, into readable sentences.

(it's also a single python file you can actually read end to end, if that's your thing.)

## quickstart

you need a mac with apple silicon. the fast path:

```bash
./setup.sh        # python deps + BlackHole + (optional) ollama + downloads the models
```

that does everything below. if you'd rather understand the pieces, here they are.

### 1. install the python + system deps

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
brew bundle --file=Brewfile      # installs BlackHole + ffmpeg + ollama
```

### 2. capturing system audio — why you need BlackHole

macOS, on purpose, won't let an app just grab "whatever is coming out of the speakers." a microphone, sure. the system output, no. so to feed your mac's audio into this app we need a **virtual audio device** that loopbacks output → input. that's [**BlackHole**](https://github.com/ExistentialAudio/BlackHole) — a free, open-source virtual audio driver. the `2ch` variant is what we use.

the catch: if you send audio *into* BlackHole, it stops coming out of your speakers, so you'd go deaf. the fix is a **Multi-Output Device** that sends audio to your speakers/headphones *and* BlackHole at the same time:

1. open **Audio MIDI Setup** (in /Applications/Utilities)
2. click **+** → **Create Multi-Output Device**
3. tick both your normal output (e.g. *MacBook Speakers*) **and** *BlackHole 2ch*
4. set that Multi-Output Device as your system output (System Settings → Sound → Output)

now you hear everything normally, and the app hears it too. (if you don't care about hearing it yourself, you can skip the Multi-Output and just set output straight to BlackHole.)

### 3. the models

three models get downloaded the first time (or up front by `setup.sh`):

- **Whisper large-v3** (transcription) and **Qwen3.5-9B** (translation) — both MLX, pulled automatically from Hugging Face on first run. no API keys, no accounts.
- **Silero VAD** ships inside the `silero-vad` pip package — nothing to download.

to grab them ahead of time instead of on first launch:

```bash
./.venv/bin/python -c "from huggingface_hub import snapshot_download; \
[snapshot_download(r) for r in ('mlx-community/whisper-large-v3-mlx','mlx-community/Qwen3.5-9B-MLX-4bit')]"
```

(optional) if you want the translategemma backend instead of Qwen: `ollama pull translategemma:12b`.

### 4. run

two ways — pick whichever you like, they open the exact same overlay.

**a) double-click `LiveTranslate.app`** — the no-terminal way. just launch it from Finder (see step 5 to move it to /Applications).

**b) from the terminal** — if you'd rather not use the app at all:

```bash
./.venv/bin/python live_translate_overlay.py --target ru
```

(call the venv's python directly — don't run `./live_translate_overlay.py`, its shebang is hard-coded.) this is also where you tweak anything: pass `--source es`, `--whisper medium`, etc. to make it behave *identically to the app's launcher*, add the flags the bundle uses by default:

```bash
./.venv/bin/python live_translate_overlay.py \
    --legacy-chunking --mlx-llm mlx-community/Qwen3.5-9B-MLX-4bit --target ru
```

see [knobs](#knobs) below or `--help` for the full list.

either way: pick source/target languages in the overlay's top bar, play a video in any app, text shows up.

### 5. (optional) put the app in /Applications

`LiveTranslate.app` lives inside the project folder and works two ways:

- **portable** (default) — keep the `.app` where it is and just double-click it. the launcher finds the project relative to itself, so you can copy the *whole folder* anywhere, on any mac, and it still works. nothing to configure.
- **installed** — if you'd rather have the app on its own in `/Applications` (Launchpad, Spotlight, Dock), run:

  ```bash
  ./install-app.sh            # copies to /Applications, or: ./install-app.sh ~/Apps
  ```

  this records the project's location (so the detached app can still find your venv + models) and re-signs the copy. if you later move the project, just run `./install-app.sh` again from its new spot.

**first launch on your mac:** the app is ad-hoc signed (no Apple Developer ID), so Gatekeeper may refuse the first double-click with *"can't be opened because it is from an unidentified developer."* this is expected for a local build. just **right-click the app → Open** once and confirm — macOS remembers the choice and double-click works normally afterward. (or: System Settings → Privacy & Security → *Open Anyway*.) you'll also get a one-time microphone prompt — that's BlackHole's loopback audio, allow it.

## the bar at the top

the window is two columns — original on the left, translation on the right — and a thin toolbar. nothing needs explaining, but since you asked:

- **Source / Target** — the language you're listening to and the one you want it in. leave Source on *Auto* and whisper figures it out per sentence (pin it if the audio mixes languages and the detector wobbles).
- **Compact** — hide the original, show only the translation (one wide column) when you just want the gist.
- **Pin** — keep the window floating above everything (on by default). turn off if you want it to behave like a normal window.
- **Opacity** slider + **A- / A+** — how see-through the glass is, and the font size. read it over a bright video or shrink it out of the way.
- **lag: N** — how many audio chunks are waiting. zero-ish = keeping up live; if it climbs and stays up, your machine isn't keeping pace (try `--whisper medium`).
- 🗑 **(trash)** — wipe everything and start clean: both columns, the history, and all the model state behind them.
- **PDF** — dump the whole session (every original + translation pair, not just what's on screen) to a PDF.

the window itself is draggable (grab anywhere) and resizable from the edges; controls on the right tuck away if you make it narrow.

## how it works (the parts i think are interesting)

the naive version of this — chop audio into fixed 5s chunks, transcribe each one independently — is bad. whisper sticks a period at the end of every chunk, cuts words in half at boundaries, and repeats itself. most of the interesting work is in *not* doing that.

- **segmentation that doesn't lose anything (the default).** audio is cut into chunks *at natural pauses* (found with VAD), not at fixed sizes, so words don't get sliced in half. each chunk is transcribed exactly once, with overlap between chunks; an overlap-dedup merges the seams, a sentence assembler regroups text on real punctuation, and a spurious-period stripper removes the period whisper adds at a chunk end when the speaker only paused for breath. the hard rule here is **no dropped audio**: the queues block instead of dropping, so if transcription falls behind for a moment it catches up rather than throwing speech away. for live news i'd rather be a second late than miss a word.

- **a smoother (but lossy) streaming mode, off by default.** there's also a LocalAgreement-2 mode (`whisper-streaming` / WhisperLiveKit style): keep a rolling buffer, re-transcribe it every ~1.5s, and only *commit* a word once two consecutive passes agree on it, showing the unconfirmed tail as a dim live draft. it reads beautifully — text flows and self-corrects instead of appearing in blocks. but re-transcribing the same audio repeatedly is expensive, and to keep up with large-v3 it has to either drop audio or trim its buffer — i.e. **lose pieces**. since i can't afford that, it's behind a flag; the default is the lossless chunker above. (it's the right call if you pair it with a faster model.)

- **surviving whisper's punctuation drift.** on fast, pause-less speech whisper sometimes stops emitting punctuation for a stretch — and because the recently finalized text is fed back as its `initial_prompt`, a punctuation-less run *reinforces itself* and the overlay collapses into one giant paragraph, then snaps back later. two guards: (1) the prompt feedback is dropped whenever that recent context has no sentence punctuation, so the next chunk can re-introduce sentence boundaries instead of inheriting the drift; (2) when a block still arrives with no punctuation at all, it's split into readable paragraphs at clause boundaries (commas/dashes) or, failing that, at word boundaries near a target length — never one wall, never mid-word.

- **VAD before whisper.** whisper *will* invent speech out of silence and music — it was trained on youtube subtitles, so on non-speech it confidently emits "thanks for watching" and friends. so we run [Silero VAD](https://github.com/snakers4/silero-vad) first and simply don't send chunks without enough real speech. this killed most of the hallucinations and also fixed a nasty stall where long pauses (switching videos) would choke the pipeline.

- **a hallucination blocklist** for the ones that sneak through anyway ("gracias", "subtitles by amara.org", etc.), stripped before translation.

- **translation by a real LLM, not word-by-word MT.** the LLM gets whole sentences and the recently-translated context, so it produces coherent sentences instead of word salad. translation runs in the same worker thread that loaded the model — sounds obvious, but MLX uses a thread-local GPU stream, so loading on one thread and generating on another silently explodes on long prompts. lazy-load on first use fixed it.

- the overlay is just an `NSTextView` in a translucent floating window. resizable, pinnable, opacity slider, font size, save-to-PDF, clear button. nothing fancy.

## the models

**transcription: Whisper large-v3, on MLX.** two separate choices here. `large-v3` because it's the most accurate Whisper size — for live foreign-language audio with names, places and accents you want every bit of accuracy you can get, and the cheaper models start dropping words. and **[MLX](https://github.com/ml-explore/mlx) because it's the fastest way to run Whisper on a mac** — it's apple's own array framework, runs on the GPU through Metal with unified memory (no copying audio tensors back and forth), and on apple silicon it beats the CPU-bound options (faster-whisper, whisper.cpp) for this workload. that combination — heaviest model at usable speed — is the *only* reason real-time large-v3 is even feasible on a laptop. on a smaller mac you can still drop to `--whisper medium` for more headroom.

**translation** defaults to **Qwen3.5-9B** (4-bit MLX, on-device). i also wired up **translategemma:12b** (a translation-tuned Gemma, via ollama) and did a little bake-off on real news clips:

- **translategemma**: more fluent, better with idioms and world knowledge ("los Mossos" → "catalan police"). but sometimes *adds* stuff that wasn't said, and ~1.5x slower.
- **Qwen3.5-9B**: faster, more literal/faithful, occasionally clumsy word choice.

for live news i kept Qwen — accuracy + latency win over polish. but it's one line in the launcher to switch (`--translator ollama --ollama-model translategemma:12b`). your mileage will vary by language pair; try both with your ears.

## knobs

it's a CLI under the hood, so everything is tunable. some you'll actually touch:

```
--legacy-chunking          the lossless chunk pipeline (DEFAULT in the .app launcher)
--source / --target        languages (or pick in the UI). source=auto detects per-utterance
--whisper large-v3         transcription model (small/medium are faster, worse)
--mlx-llm REPO             translation model (any MLX-LM repo)
--translator ollama        use ollama instead of MLX (e.g. for translategemma)
--silence-rms 0.006        louder = stricter silence gate
--vad-min-speech-ms 250    min real speech per window before whisper sees it
# streaming mode (omit --legacy-chunking): smoother, but lossy under load
--update-seconds 1.5       how often to re-transcribe the rolling buffer (lower = snappier, heavier)
```

`./.venv/bin/python live_translate_overlay.py --help` for the full list (there are ~30; most you'll never need).

## robustness / things that took embarrassingly long

the "interesting" 20% above is the fun part. the other 80% was making it not fall over, which is never in the demo but is the whole difference between a toy and something you leave running for an hour. a sampler:

- **the audio thread silently dying.** switch youtube videos, CoreAudio re-inits the route, the input callback goes quiet forever, and transcription just... stops, with no error. now there's a watchdog that notices the silence and reopens the stream.
- **the live-vs-complete tradeoff.** my first instinct when whisper fell behind was to drop the oldest audio and stay *live* — great for a clock, terrible when you actually care about every word. so the default flipped the other way: the queues block, nothing gets dropped, and if it falls behind it just runs a second or two late and catches up. "don't lose pieces" won over "always be live."
- **the Clear button only clearing the screen.** turns out "clear" needs to wipe the rolling audio buffer, both worker threads' state, the pending queues, *and* discard the chunk that's mid-flight inside whisper at that exact moment. otherwise old text keeps dribbling in after you hit clear. all of that now resets from one generation counter.
- **workers dying on a single bad frame.** one unguarded exception in the audio loop and the whole pipeline goes dark with nothing in the log. everything is wrapped now and logs `[chunk]`/`[audio]`/`[stream]`/`[ui]` so the next stall names itself.
- **shipping it as a double-clickable app.** this ate an afternoon. macOS LaunchServices will *not* run a `.app` whose executable is a shell script — it just silently does nothing. so the bundle's executable is a tiny compiled C launcher that execs the real script, the whole thing is ad-hoc codesigned (`codesign -s -`), and only then does double-click work. also it kept vanishing from the Dock until i set the activation policy to `Regular`.

none of this is clever. all of it was necessary.

## honest caveats

- macos + apple silicon only. the overlay is pyobjc/Cocoa, the inference is MLX.
- you have to route audio through BlackHole yourself. macos makes capturing system audio annoying on purpose.
- large-v3 is not free. the default lossless chunker keeps up on an M-series mac; if it ever can't, it stays correct but the subtitles drift a second or two behind (it won't drop words). on a smaller mac use `--whisper medium`. the streaming mode is heavier still.
- auto language detection wobbles on mixed-language audio (spanish news with catalan inserts will flip-flop). pin `--source` if you know it.
- it still hallucinates sometimes. it's whisper. we mitigate, we don't cure.
- the code is one big file. it's a personal tool, not a framework. sorry/not sorry.

## layout

```
live_translate_overlay.py   the whole thing (capture, VAD, ASR, translate, overlay)
LiveTranslate.app           double-clickable bundle (just launches the script via your venv)
install-app.sh              put the .app in /Applications, detached from the project folder
setup.sh / Brewfile         install everything
requirements*.txt           python deps
```

## license / spirit

[MIT](LICENSE) — personal hack, take it and do whatever. PRs welcome but i make no promises about keeping this tidy.
