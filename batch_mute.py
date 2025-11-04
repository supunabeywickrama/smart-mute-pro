import argparse, csv, json
from pathlib import Path
from pydub import AudioSegment
from mute_words import (
    require_ffmpeg, ffmpeg_extract_audio_pcm, ffmpeg_extract_audio_full, ffmpeg_mux_audio,
    transcribe_with_words, build_hits, merge_intervals_from_hits, build_censored_audio
)

VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".webm",".m4v"}
AUDIO_EXTS = {".mp3",".wav",".m4a",".aac",".flac",".ogg",".wma"}

def is_media(p: Path):
    return p.suffix.lower() in VIDEO_EXTS.union(AUDIO_EXTS)

def main():
    ap = argparse.ArgumentParser("Batch mute words in a folder")
    ap.add_argument("in_dir", type=str)
    ap.add_argument("out_dir", type=str)
    ap.add_argument("--words", type=str, required=True)
    ap.add_argument("--model", type=str, default="small")
    ap.add_argument("--fuzzy", action="store_true")
    ap.add_argument("--margin", type=int, default=60)
    ap.add_argument("--freq", type=int, default=1000)
    ap.add_argument("--gain", type=int, default=-6)
    ap.add_argument("--mute_mode", choices=["replace","duck"], default="replace")
    ap.add_argument("--duck_db", type=int, default=-40)
    args = ap.parse_args()

    require_ffmpeg()
    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.rglob("*") if p.is_file() and is_media(p)]
    print(f"Found {len(files)} media files.")

    banned = [w.strip() for w in args.words.split(",") if w.strip()]

    for src in files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        print(f"-> {src}")
        # asr
        asr_wav = src.with_suffix(".tmp_asr.wav")
        ffmpeg_extract_audio_pcm(src, asr_wav)
        words, _ = transcribe_with_words(asr_wav, model_size=args.model)
        asr_wav.unlink(missing_ok=True)

        hits = build_hits(words, banned, fuzzy=args.fuzzy)
        intervals = merge_intervals_from_hits(hits, margin_ms=args.margin)

        full_wav = src.with_suffix(".tmp_full.wav")
        ffmpeg_extract_audio_full(src, full_wav)
        audio = AudioSegment.from_file(full_wav)

        censored = build_censored_audio(
            audio, intervals,
            freq=args.freq, gain_db=args.gain,
            mode=args.mute_mode, duck_db=args.duck_db
        )

        if src.suffix.lower() in VIDEO_EXTS:
            tmp_wav = src.with_suffix(".tmp_censored.wav")
            censored.export(tmp_wav, format="wav")
            final = dst.with_suffix(".mp4")
            ffmpeg_mux_audio(src, tmp_wav, final)
            tmp_wav.unlink(missing_ok=True)
        else:
            final = dst.with_suffix(".wav")
            censored.export(final, format="wav")

        full_wav.unlink(missing_ok=True)
        print(f"   saved: {final}")

if __name__ == "__main__":
    main()
