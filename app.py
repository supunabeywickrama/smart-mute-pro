import io, os, tempfile, time
from pathlib import Path

import streamlit as st
from pydub import AudioSegment
from mute_words import (
    require_ffmpeg,
    ffmpeg_extract_audio_pcm, ffmpeg_extract_audio_full, ffmpeg_mux_audio,
    transcribe_with_words, build_hits, merge_intervals_from_hits, build_censored_audio,
    DEFAULT_BEEP_FREQ, DEFAULT_BEEP_DB, DEFAULT_MARGIN_MS
)

from pydub import AudioSegment
import os
# hard-point ffmpeg for Streamlit worker process

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")



st.set_page_config(page_title="Smart Mute Pro", page_icon="🔇", layout="wide")

st.title("🔇 Smart Mute Pro")
st.caption("Upload audio/video → choose words → review → render with clean “teeeek” mutes")

require_ffmpeg()

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=2)
    fuzzy = st.checkbox("Fuzzy match", value=True)
    margin = st.number_input("Padding (ms)", min_value=0, max_value=500, value=DEFAULT_MARGIN_MS, step=10)
    freq = st.number_input("Beep frequency (Hz)", min_value=100, max_value=4000, value=DEFAULT_BEEP_FREQ, step=50)
    gain = st.number_input("Beep gain (dB)", min_value=-60, max_value=0, value=DEFAULT_BEEP_DB, step=1)
    mute_mode = st.selectbox("Mute mode", ["replace","duck"], index=0)
    duck_db = st.number_input("Duck gain (dB)", min_value=-80, max_value=0, value=-40, step=1)
    st.markdown("---")
    default_words = st.text_input("Words to mute (comma-separated)", "the,so,we,now,you")
    st.caption("You can also type words below from transcript table and toggle censor per word.")

uploaded = st.file_uploader("Upload audio or video", type=[
    "mp4","mov","mkv","avi","webm","m4v",
    "mp3","wav","m4a","aac","flac","ogg","wma"
])

if uploaded:
    tmpdir = Path(tempfile.mkdtemp(prefix="smp_ui_"))
    src_path = tmpdir / uploaded.name
    src_path.write_bytes(uploaded.read())

    st.info(f"File saved to: `{src_path}`")

    # --- Step 1: ASR to words
    st.write("### 1) Transcribe")
    pcm = tmpdir / "asr.wav"
    t0 = time.time()
    ffmpeg_extract_audio_pcm(src_path, pcm)
    words, info = transcribe_with_words(pcm, model_size=model)
    st.success(f"ASR done in {time.time()-t0:.1f}s • language={getattr(info,'language','?')} • tokens={len(words)}")

    # --- Step 2: build hits
    banned = [w.strip() for w in default_words.split(",") if w.strip()]
    hits = build_hits(words, banned, fuzzy=fuzzy)

    # editable table for censor flags
    import pandas as pd
    df = pd.DataFrame([{
        "idx": h.idx, "word": h.word, "start_ms": h.start_ms, "end_ms": h.end_ms,
        "prob": h.prob, "matched": int(h.matched), "censor": bool(h.censor), "matched_with": h.matched_with
    } for h in hits])

    st.write("### 2) Review & toggle censor")
    st.caption("Tip: filter the `matched==1` rows to see only hits; toggle `censor` as needed.")
    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "idx": st.column_config.NumberColumn(disabled=True),
            "word": st.column_config.TextColumn(disabled=True),
            "start_ms": st.column_config.NumberColumn(disabled=True),
            "end_ms": st.column_config.NumberColumn(disabled=True),
            "prob": st.column_config.NumberColumn(format="%.3f", disabled=True),
            "matched": st.column_config.CheckboxColumn(disabled=True),
            "censor": st.column_config.CheckboxColumn(),
            "matched_with": st.column_config.TextColumn(disabled=True),
        }
    )

    # push edited censor flags back
    censor_map = {int(row["idx"]): bool(row["censor"]) for _, row in edited.iterrows()}
    for h in hits:
        if h.idx in censor_map:
            h.censor = censor_map[h.idx]

    # --- Step 3: render
    st.write("### 3) Render")
    do_render = st.button("Render censored output")
    if do_render:
        # audio extraction
        full_wav = tmpdir / "full.wav"
        ffmpeg_extract_audio_full(src_path, full_wav)
        audio = AudioSegment.from_file(full_wav)

        intervals = merge_intervals_from_hits(hits, margin_ms=int(margin))
        censored = build_censored_audio(
            audio, intervals,
            freq=int(freq), gain_db=int(gain),
            mode=mute_mode, duck_db=int(duck_db)
        )

        # save temp wav
        cens_wav = tmpdir / "censored.wav"
        censored.export(cens_wav, format="wav")

        # if video -> mux; else return wav
        if src_path.suffix.lower() in {".mp4",".mov",".mkv",".avi",".webm",".m4v"}:
            out_path = tmpdir / (src_path.stem + "_censored.mp4")
            ffmpeg_mux_audio(src_path, cens_wav, out_path)
            st.success("Done! Download your censored **video** below.")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download video (MP4)", f, file_name=out_path.name, mime="video/mp4")
        else:
            out_path = tmpdir / (src_path.stem + "_censored.wav")
            with open(cens_wav, "rb") as fsrc, open(out_path, "wb") as fdst:
                fdst.write(fsrc.read())
            st.success("Done! Download your censored **audio** below.")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download audio (WAV)", f, file_name=out_path.name, mime="audio/wav")
