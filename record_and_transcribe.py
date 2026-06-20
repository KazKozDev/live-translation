#!/usr/bin/env python3
"""
record_and_transcribe.py

Records system audio on Mac (anything playing — browser, media player,
calls, any app) and transcribes locally via MLX Whisper,
which uses the GPU on Apple Silicon (fast + accurate).

------------------------------------------------------------------------------
REQUIREMENTS (install once):

    brew install blackhole-2ch        # virtual audio device (loopback)
    brew install ffmpeg               # required by mlx-whisper for audio decoding
    pip install mlx-whisper sounddevice soundfile numpy

AUDIO SETUP (once):
    1. Open "Audio MIDI Setup".
    2. Create a "Multi-Output Device" (bottom "+").
    3. Check your speakers/headphones AND "BlackHole 2ch" in it.
    4. Set this Multi-Output Device as the system audio output
       (volume menu / System Settings -> Sound -> Output).
    Now you hear the audio and it simultaneously goes into BlackHole.
------------------------------------------------------------------------------

EXAMPLES:

    # list audio devices and their indices
    python record_and_transcribe.py --list

    # record until Ctrl+C, then transcribe immediately
    python record_and_transcribe.py --transcribe

    # record exactly 5 minutes of English speech with the large model
    python record_and_transcribe.py --duration 300 --language en --transcribe

    # transcribe an existing audio file (no recording)
    python record_and_transcribe.py --input some_audio.wav --transcribe
"""

import argparse
import datetime as dt
import json
import queue
import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf

# MLX Whisper models from Hugging Face (downloaded automatically on first run).
#   large-v3  -> maximum accuracy (recommended when speed is not a concern)
#   turbo     -> faster than large-v3, slightly less accurate
#   small     -> faster, less accurate
#   tiny      -> very fast, for drafts
MODELS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def list_devices():
    print(sd.query_devices())
    print("\nHint: look for the row with 'BlackHole' and its index on the left.")


def find_blackhole():
    """Automatically finds the BlackHole input device."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
            return idx, dev["name"]
    return None, None


def record(output_path, device, samplerate, channels, duration=None):
    """Streams audio from the input device to a WAV file (constant memory usage)."""
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print(f"\n● Recording -> {output_path}")
    if duration:
        print(f"  Duration: {duration} s")
    else:
        print("  Press Ctrl+C to stop")
    print(
        f"  Device: {sd.query_devices(device)['name']}  "
        f"({samplerate:.0f} Hz, {channels} ch)\n"
    )

    frames_written = 0
    target_frames = int(duration * samplerate) if duration else None

    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=int(samplerate),
        channels=channels,
        subtype="PCM_16",
    ) as f:
        with sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            callback=callback,
        ):
            try:
                while True:
                    block = q.get()
                    f.write(block)
                    frames_written += len(block)
                    if target_frames and frames_written >= target_frames:
                        break
            except KeyboardInterrupt:
                pass

    print(f"✓ Recording saved: {output_path}")
    return output_path


def _fmt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(result, base_path):
    """Saves .txt, .srt and .json next to the base filename."""
    base = Path(base_path).with_suffix("")
    txt = base.with_suffix(".txt")
    srt = base.with_suffix(".srt")
    js = base.with_suffix(".json")

    txt.write_text(result["text"].strip() + "\n", encoding="utf-8")

    with open(srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.get("segments", []), 1):
            f.write(
                f"{i}\n"
                f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n"
                f"{seg['text'].strip()}\n\n"
            )

    js.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ Text:      {txt}")
    print(f"✓ Subtitles: {srt}")
    print(f"✓ JSON:      {js}")


def transcribe(audio_path, model_repo, language):
    """Transcribes via MLX Whisper with hallucination protection on silence."""
    import mlx_whisper  # imported here so --list works without mlx installed

    print(f"\n● Transcribing via {model_repo} ...")
    print("  (first run downloads the model from Hugging Face — one time only)")

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        language=language,  # None => auto-detect language
        # --- guards against looping/hallucinations on silent segments ---
        condition_on_previous_text=False,  # prevents the model from looping on its own output
        hallucination_silence_threshold=2.0,  # skip suspiciously quiet segments
        no_speech_threshold=0.6,  # threshold for "no speech here"
        compression_ratio_threshold=2.4,  # cuts repetitive garbage ("sniff sniff...")
        logprob_threshold=-1.0,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # fallback on failure
        verbose=False,
    )
    print("✓ Done.")
    return result


def main():
    p = argparse.ArgumentParser(
        description="Record Mac system audio + local transcription (MLX Whisper)."
    )
    p.add_argument("--list", action="store_true", help="show audio devices and exit")
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="input device index (default: auto-detect BlackHole)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="how many seconds to record (default: until Ctrl+C)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="output WAV filename (default: timestamped name)",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="transcribe an existing audio file instead of recording",
    )
    p.add_argument("--transcribe", action="store_true", help="transcribe after recording")
    p.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="large-v3",
        help="Whisper model (default: large-v3)",
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="language code, e.g. en / ru (default: auto-detect)",
    )
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    # Transcribe-only mode for an existing file
    if args.input:
        audio = Path(args.input)
        if not audio.exists():
            sys.exit(f"File not found: {audio}")
        result = transcribe(audio, MODELS[args.model], args.language)
        write_outputs(result, audio)
        return

    # Select recording device
    device = args.device
    if device is None:
        device, name = find_blackhole()
        if device is None:
            sys.exit(
                "BlackHole not found. Install it (brew install blackhole-2ch) and\n"
                "set up a Multi-Output Device, or specify --device N (see --list)."
            )
        print(f"Found device: [{device}] {name}")

    info = sd.query_devices(device)
    samplerate = info["default_samplerate"]
    channels = min(2, info["max_input_channels"]) or 1

    if args.output:
        wav_path = args.output
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = f"recording_{stamp}.wav"

    record(wav_path, device, samplerate, channels, args.duration)

    if args.transcribe:
        result = transcribe(wav_path, MODELS[args.model], args.language)
        write_outputs(result, wav_path)
    else:
        print("\nTo transcribe later:")
        print(f"  python {Path(__file__).name} --input {wav_path} --transcribe")


if __name__ == "__main__":
    main()
