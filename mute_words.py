# mute_words.py

import argparse, csv, json, os, re, shutil, subprocess, sys, tempfile, traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

from faster_whisper import WhisperModel
from pydub import AudioSegment, generators
from rapidfuzz import fuzz

# --- Point pydub to ffmpeg on Windows (adjust if installed elsewhere) ---
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

# --- robust ffmpeg resolution (works even if PATH is missing) ---
def _ffmpeg_path() -> str:
    """
    Returns a full path to ffmpeg. Tries PATH, then AudioSegment.converter.
    Raises a clear error if not found.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    conv = getattr(AudioSegment, "converter", None)
    if conv and Path(conv).exists():
        return str(conv)
    raise SystemExit(
        "FFmpeg not found.\n"
        "Set AudioSegment.converter to your ffmpeg.exe or add ffmpeg to PATH.\n"
        "E.g., AudioSegment.converter = r'C:\\ffmpeg\\bin\\ffmpeg.exe'"
    )

# -------------------- Defaults --------------------
DEFAULT_BEEP_FREQ = 1000   # Hz
DEFAULT_BEEP_DB = -6       # dB (relative)
DEFAULT_MARGIN_MS = 60     # ms padding around words
FUZZY_THRESHOLD = 90       # 0..100

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}

# -------------------- Data model --------------------
@dataclass
class WordHit:
    idx: int
    word: str
    start_ms: int
    end_ms: int
    prob: float
    matched: bool
    censor: bool
    matched_with: str

# -------------------- Utilities --------------------
def require_ffmpeg():
    """
    Ensure ffmpeg is callable for both subprocess and pydub.
    """
    # Will raise SystemExit with a clear message if not found
    _ = _ffmpeg_path()

def ffmpeg_extract_audio_pcm(src: Path, dst: Path, sr=16000):
    ff = _ffmpeg_path()
    cmd = [ff, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ffmpeg_extract_audio_full(src: Path, dst: Path):
    ff = _ffmpeg_path()
    cmd = [ff, "-y", "-i", str(src), "-vn", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ffmpeg_mux_audio(video_in: Path, audio_in_wav: Path, out_path: Path):
    ff = _ffmpeg_path()
    cmd = [
        ff, "-y",
        "-i", str(video_in),
        "-i", str(audio_in_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)

def normalize_token(t: str) -> str:
    """
    Unicode-safe: keep only letters and digits; lowercase.
    Avoids unsupported regex classes like \\p{L}/\\p{N}.
    """
    t = t.lower()
    return "".join(ch for ch in t if ch.isalnum())

def transcribe_with_words(wav_path: Path, model_size="small"):
    model = WhisperModel(model_size, compute_type="auto")
    segments, info = model.transcribe(str(wav_path), word_timestamps=True)

    words = []
    k = 0
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            words.append((
                k,
                (w.word or "").strip(),
                max(0, int(w.start * 1000)),
                max(0, int(w.end * 1000)),
                float(getattr(w, "prob", 1.0))
            ))
            k += 1
    return words, info

def build_hits(words, banned: List[str], fuzzy=False) -> List[WordHit]:
    blist = [normalize_token(b) for b in banned if b.strip()]
    hits: List[WordHit] = []
    for (idx, token, s, e, p) in words:
        wn = normalize_token(token)
        matched = False
        matched_with = ""
        for b in blist:
            if not b:
                continue
            if not fuzzy:
                if wn == b:
                    matched, matched_with = True, b
                    break
            else:
                if fuzz.partial_ratio(wn, b) >= FUZZY_THRESHOLD:
                    matched, matched_with = True, b
                    break
        hits.append(WordHit(idx, token, s, e, p, matched, matched, matched_with))
    return hits

def merge_intervals_from_hits(hits: List[WordHit], margin_ms=DEFAULT_MARGIN_MS) -> List[Tuple[int, int]]:
    intervals = []
    for h in hits:
        if h.censor:
            intervals.append((max(0, h.start_ms - margin_ms), h.end_ms + margin_ms))
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

# ---- Old overlay-only method (kept for reference/optional use) ----
def apply_beeps(audio: AudioSegment, intervals, freq=DEFAULT_BEEP_FREQ, gain_db=DEFAULT_BEEP_DB):
    out = audio
    for s, e in intervals:
        dur = max(0, e - s)
        if dur <= 0:
            continue
        beep = generators.Sine(freq).to_audio_segment(duration=dur).apply_gain(gain_db)
        fade = min(10, dur // 8)
        beep = beep.fade_in(fade).fade_out(fade)
        overlay = AudioSegment.silent(duration=s) + beep
        if len(overlay) < len(out):
            overlay += AudioSegment.silent(duration=len(out) - len(overlay))
        out = out.overlay(overlay)
    return out

# ---- Helper: safe append with adaptive crossfade ----
def _safe_append(base: AudioSegment, seg: AudioSegment, max_cf_ms: int = 10) -> AudioSegment:
    """
    Append seg to base with a crossfade that never exceeds either segment's length.
    Handles zero-length segments safely.
    """
    if len(seg) == 0:
        return base
    if len(base) == 0:
        return base + seg
    cf = min(max_cf_ms, len(base), len(seg))
    if cf > 0:
        return base.append(seg, crossfade=cf)
    else:
        return base + seg

# ---- NEW: Proper mute/duck while beeping (uses safe crossfades) ----
def build_censored_audio(audio: AudioSegment, intervals, freq=DEFAULT_BEEP_FREQ, gain_db=DEFAULT_BEEP_DB,
                         mode="replace", duck_db=-40):
    """
    Create a new audio where each interval is either replaced (muted) with a beep,
    or ducked (attenuated) under a beep. Uses safe adaptive crossfades.
    """
    if not intervals:
        return audio

    L = len(audio)
    # clamp & keep only valid intervals
    norm = []
    for s, e in intervals:
        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e > s:
            norm.append((s, e))
    if not norm:
        return audio

    out = AudioSegment.silent(duration=0)
    prev = 0
    for s, e in norm:
        # unchanged region before this interval
        if s > prev:
            pre = audio[prev:s]
            out = _safe_append(out, pre, max_cf_ms=10)

        dur = e - s
        if dur <= 0:
            prev = e
            continue

        beep = generators.Sine(freq).to_audio_segment(duration=dur).apply_gain(gain_db)
        fade = min(10, dur // 8)
        beep = beep.fade_in(fade).fade_out(fade)

        if mode == "duck":
            base = audio[s:e] + duck_db  # attenuate original
            cens = base.overlay(beep)
        else:  # "replace"
            cens = beep  # fully mute original under beep

        out = _safe_append(out, cens, max_cf_ms=10)
        prev = e

    # tail after last interval
    if prev < L:
        tail = audio[prev:]
        out = _safe_append(out, tail, max_cf_ms=10)

    return out

# -------------------- Report Writers --------------------
def write_csv(path: Path, hits: List[WordHit]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "word", "start_ms", "end_ms", "prob", "matched", "censor", "matched_with"])
        for h in hits:
            writer.writerow([
                h.idx,
                h.word,
                h.start_ms,
                h.end_ms,
                f"{h.prob:.3f}",
                int(h.matched),
                int(h.censor),
                h.matched_with
            ])

def write_json(path: Path, hits: List[WordHit], meta: dict):
    data = {"meta": meta, "hits": [asdict(h) for h in hits]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def write_srt(path: Path, hits: List[WordHit]):
    def fmt(ms):
        h = ms // 3600000; ms %= 3600000
        m = ms // 60000;   ms %= 60000
        s = ms // 1000;    ms %= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(str(i))
        lines.append(f"{fmt(h.start_ms)} --> {fmt(h.end_ms)}")
        tag = "CENSOR" if h.censor else "KEEP"
        lines.append(f"[{tag}] {h.word}")
        lines.append("")  # blank line between entries

    path.write_text("\n".join(lines), encoding="utf-8")

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Smart Mute Pro: review -> render censored audio/video.")
    ap.add_argument("input", type=str, help="Input media file")
    ap.add_argument("--words", type=str, default="", help="Comma-separated words to mute")
    ap.add_argument("--model", type=str, default="small", help="Whisper model: tiny/base/small/medium/large-v3")
    ap.add_argument("--fuzzy", action="store_true", help="Fuzzy matching")
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN_MS, help="Padding ms around words")
    ap.add_argument("--freq", type=int, default=DEFAULT_BEEP_FREQ, help="Beep frequency Hz")
    ap.add_argument("--gain", type=int, default=DEFAULT_BEEP_DB, help="Beep gain dB (e.g., -6)")
    ap.add_argument("--report_only", action="store_true", help="Only produce reports (no rendering)")
    ap.add_argument("--apply_from", type=str, help="CSV/JSON file with edited 'censor' flags to apply")
    ap.add_argument("--out", type=str, help="Output path (auto if omitted)")
    ap.add_argument("--mute_mode", type=str, default="replace", choices=["replace","duck"],
                    help="How to treat original audio during beep: replace (mute) or duck (attenuate)")
    ap.add_argument("--duck_db", type=int, default=-40, help="Gain applied to segment when mute_mode=duck")
    args = ap.parse_args()

    try:
        require_ffmpeg()

        src = Path(args.input)
        if not src.exists():
            sys.exit(f"Input not found: {src}")

        print("[1/5] Extracting audio for ASR…", flush=True)
        tmp = Path(tempfile.mkdtemp(prefix="smartmute_"))
        pcm = tmp / "asr.wav"
        ffmpeg_extract_audio_pcm(src, pcm)

        print("[2/5] Running ASR (word timestamps)…", flush=True)
        words, info = transcribe_with_words(pcm, model_size=args.model)
        print(f"      ASR language={getattr(info,'language','unknown')} • tokens={len(words)}", flush=True)

        print("[3/5] Matching words…", flush=True)
        banned = [w.strip() for w in args.words.split(",") if w.strip()]
        hits = build_hits(words, banned, fuzzy=args.fuzzy)

        # ingest edited decisions if provided
        if args.apply_from:
            edited_path = Path(args.apply_from)
            if edited_path.suffix.lower() == ".csv":
                idx2censor = {}
                with open(edited_path, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        idx2censor[int(row["idx"])] = (row.get("censor", "0") in ["1", "true", "True"])
                for h in hits:
                    if h.idx in idx2censor:
                        h.censor = idx2censor[h.idx]
            else:
                data = json.loads(edited_path.read_text(encoding="utf-8"))
                ed = {int(x["idx"]): bool(x["censor"]) for x in data.get("hits", [])}
                for h in hits:
                    if h.idx in ed:
                        h.censor = ed[h.idx]

        print("[4/5] Writing review files…", flush=True)
        meta = {
            "language": getattr(info, "language", "unknown"),
            "input": str(src),
            "model": args.model,
            "banned_words": banned,
            "fuzzy": args.fuzzy,
            "margin_ms": args.margin,
        }
        report_csv = src.with_suffix(".review.csv")
        report_json = src.with_suffix(".review.json")
        report_srt  = src.with_suffix(".review.srt")
        write_csv(report_csv, hits)
        write_json(report_json, hits, meta)
        write_srt(report_srt, hits)
        print(f"      CSV : {report_csv}")
        print(f"      JSON: {report_json}")
        print(f"      SRT : {report_srt}")

        if args.report_only:
            print("Report-only mode; not rendering output.")
            return

        print("[5/5] Rendering censored output…", flush=True)
        full_wav = tmp / "full.wav"
        ffmpeg_extract_audio_full(src, full_wav)
        audio = AudioSegment.from_file(full_wav)
        intervals = merge_intervals_from_hits(hits, margin_ms=args.margin)

        # Proper mute/duck build
        censored = build_censored_audio(
            audio, intervals,
            freq=args.freq, gain_db=args.gain,
            mode=args.mute_mode, duck_db=args.duck_db
        )

        suffix = src.suffix.lower()
        out = Path(args.out) if args.out else src.with_name(
            src.stem + "_censored" + (suffix if suffix else ".mp4")
        )

        if suffix in VIDEO_EXTS:
            tmp_wav = tmp / "censored.wav"
            censored.export(tmp_wav, format="wav")
            if out.suffix.lower() not in VIDEO_EXTS:
                out = out.with_suffix(".mp4")
            ffmpeg_mux_audio(src, tmp_wav, out)
            print(f"✅ Done! Video saved: {out}")
        else:
            out = out.with_suffix(".wav")
            censored.export(out, format="wav")
            print(f"✅ Done! Audio saved: {out}")

    except Exception as e:
        print("❌ Smart Mute Pro failed:\n", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
