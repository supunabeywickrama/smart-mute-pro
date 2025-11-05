import io, os, tempfile, time, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pydub import AudioSegment

# ---- our pipeline funcs from mute_words.py
from mute_words import (
    require_ffmpeg,
    ffmpeg_extract_audio_pcm, ffmpeg_extract_audio_full, ffmpeg_mux_audio,
    transcribe_with_words, build_hits, merge_intervals_from_hits, build_censored_audio,
    DEFAULT_BEEP_FREQ, DEFAULT_BEEP_DB, DEFAULT_MARGIN_MS
)

# ---------- hard-point ffmpeg for Streamlit worker (safe if already set)
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

st.set_page_config(page_title="VoCleanse", page_icon="🔮", layout="wide")

# -------------------- helpers --------------------
def file_hash(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()[:16]

@st.cache_resource(show_spinner=False)
def load_asr_model(model_name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, compute_type="auto")

@st.cache_data(show_spinner=False)
def cached_transcribe(model_name: str, wav_bytes: bytes):
    model = load_asr_model(model_name)
    tmp = Path(tempfile.mkdtemp(prefix="smp_cache_"))
    wav_path = tmp / "asr.wav"
    wav_path.write_bytes(wav_bytes)
    segments, info = model.transcribe(str(wav_path), word_timestamps=True)
    words = []
    k = 0
    for seg in segments:
        if not getattr(seg, "words", None):
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            words.append({
                "idx": k,
                "word": (w.word or "").strip(),
                "start_ms": max(0, int(w.start*1000)),
                "end_ms": max(0, int(w.end*1000)),
                "prob": float(getattr(w, "prob", 1.0)),
            })
            k += 1
    return words, {"language": getattr(info, "language", "?")}

def clamp(a, lo, hi):
    return max(lo, min(hi, a))

def extract_preview(audio: AudioSegment, start_ms: int, end_ms: int, pad_ms: int = 300):
    L = len(audio)
    s = clamp(start_ms - pad_ms, 0, L)
    e = clamp(end_ms + pad_ms, 0, L)
    return audio[s:e]

# -------------------- sidebar (settings) --------------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Whisper model", ["tiny","base","small","medium","large-v3"], index=2)
    fuzzy = st.toggle(
        "Fuzzy match",
        value=False,
        help="Off = exact word match only. On = tolerant, but short words (≤2 chars) still require exact match."
    )
    margin = st.slider("Padding (ms)", min_value=0, max_value=400, value=DEFAULT_MARGIN_MS, step=10)
    freq = st.slider("Beep frequency (Hz)", min_value=200, max_value=3000, value=DEFAULT_BEEP_FREQ, step=50)
    gain = st.slider("Beep gain (dB)", min_value=-30, max_value=0, value=DEFAULT_BEEP_DB, step=1)
    mute_mode = st.radio("Mute mode", ["replace","duck"], horizontal=True)
    duck_db = st.slider("Duck gain (dB)", min_value=-80, max_value=0, value=-40, step=1, help="Only used if mute mode = duck")

    st.divider()
    st.caption("Upload a file on the main page to choose words & presets.")

# -------------------- branding --------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4cc9f0; font-size: 3rem;'>
        🔮 VoCleanse
    </h1>
    <p style='text-align: center; font-size: 1.2rem; color: #bbb; margin-top: -10px;'>
        AI-powered voice purification in one click.
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

require_ffmpeg()

uploaded = st.file_uploader(
    "Upload audio or video (≤ 200MB)",
    type=["mp4","mov","mkv","avi","webm","m4v","mp3","wav","m4a","aac","flac","ogg","wma"],
    accept_multiple_files=False,
)

if not uploaded:
    st.info("Drop a file above to start. Supports most audio & video formats.")
    st.stop()

# persist file to temp dir
tmpdir = Path(tempfile.mkdtemp(prefix="smp_ui_"))
src_path = tmpdir / uploaded.name
src_path.write_bytes(uploaded.read())
st.success(f"File saved → `{src_path}`")

# ------------- step 1: transcribe (with caching)
with st.status("1) Transcribe…", expanded=True) as status:
    try:
        pcm = tmpdir / "asr.wav"
        ffmpeg_extract_audio_pcm(src_path, pcm)
        wav_bytes = pcm.read_bytes()
        t0 = time.time()
        words, meta = cached_transcribe(model, wav_bytes)
        status.update(label=f"1) Transcribe ✓  ({time.time()-t0:.1f}s)", state="complete")
    except Exception as e:
        status.update(label="1) Transcribe ✗", state="error")
        st.exception(e)
        st.stop()

# -------------------- presets (only visible after upload) --------------------
st.subheader("2) Choose words to mute")
st.info("Enter words to censor, or click presets to add them quickly.")

# initialize session word string once
if "__words" not in st.session_state:
    st.session_state["__words"] = "the,so,we,now,you"

# text input bound to session state
default_words = st.text_input(
    "Words to match (comma-separated)",
    key="__words",
    help="Enter exact words separated by commas. Example: uh, like, you know"
)

# preset chips (now on main page, after upload)
preset_rows = [
    ["um","uh","like","you know","so","okay"],
    ["we","I","you","the","now","and"],
    ["actually","basically","literally","kinda","sorta","right"]
]
for row in preset_rows:
    cols = st.columns(6)
    for i, w in enumerate(row[:6]):
        if cols[i].button(w, use_container_width=True, key=f"preset_{w}"):
            # add to the words input uniquely
            current = [x.strip() for x in st.session_state["__words"].split(",") if x.strip()]
            if w not in current:
                current.append(w)
                st.session_state["__words"] = ",".join(current)

# finalize banned list
banned = [w.strip() for w in st.session_state["__words"].split(",") if w.strip()]
st.write(f"🔍 **Active mute list:** {', '.join(banned) if banned else 'None'}")
st.divider()

# ------------- step 2b: match & review
from mute_words import build_hits as build_hits_raw
hits = build_hits_raw([(w["idx"], w["word"], w["start_ms"], w["end_ms"], w["prob"]) for w in words],
                      banned, fuzzy=fuzzy)
df = pd.DataFrame([{
    "idx": h.idx, "word": h.word, "start_ms": h.start_ms, "end_ms": h.end_ms,
    "prob": h.prob, "matched": int(h.matched), "censor": bool(h.censor), "matched_with": h.matched_with
} for h in hits])

# filters
c1, c2, c3, _ = st.columns([1,1,1,3])
with c1:
    only_matches = st.toggle("Show matched only", value=True)
with c2:
    only_censor = st.toggle("Show censor only", value=False)
with c3:
    prob_min = st.slider("Min prob", 0.0, 1.0, 0.0, 0.05)

fdf = df.copy()
if only_matches:
    fdf = fdf[fdf["matched"] == 1]
if only_censor:
    fdf = fdf[fdf["censor"] == True]
fdf = fdf[fdf["prob"] >= prob_min]

# stats
m1, m2, m3 = st.columns(3)
m1.metric("Tokens", len(df))
m2.metric("Matched", int((df["matched"]==1).sum()))
m3.metric("Selected (censor=1)", int((df["censor"]==True).sum()))

# editable table
edited = st.data_editor(
    fdf,
    use_container_width=True,
    height=360,
    column_config={
        "idx": st.column_config.NumberColumn(disabled=True),
        "word": st.column_config.TextColumn(disabled=True),
        "start_ms": st.column_config.NumberColumn(disabled=True),
        "end_ms": st.column_config.NumberColumn(disabled=True),
        "prob": st.column_config.NumberColumn(format="%.3f", disabled=True),
        "matched": st.column_config.CheckboxColumn(disabled=True),
        "censor": st.column_config.CheckboxColumn(),
        "matched_with": st.column_config.TextColumn(disabled=True),
    },
    key="review_table",
)

# push edits back to full df
censor_map = {int(row["idx"]): bool(row["censor"]) for _, row in edited.iterrows()}
for i in range(len(df)):
    if df.loc[i, "idx"] in censor_map:
        df.loc[i, "censor"] = censor_map[df.loc[i, "idx"]]

# ------------- timeline viz
st.caption("Timeline (blue = selected to mute, gray = other tokens)")
intervals_all = [(int(r.start_ms), int(r.end_ms)) for _, r in df.iterrows()]
intervals_censor = [(int(r.start_ms), int(r.end_ms)) for _, r in df[df["censor"]==True].iterrows()]

chart_df = pd.DataFrame({
    "start": [s/1000 for s,_ in intervals_all],
    "end":   [e/1000 for _,e in intervals_all],
    "type":  ["token"]*len(intervals_all)
})
sel_df = pd.DataFrame({
    "start": [s/1000 for s,_ in intervals_censor],
    "end":   [e/1000 for _,e in intervals_censor],
    "type":  ["censor"]*len(intervals_censor)
})
vis_df = pd.concat([chart_df, sel_df], ignore_index=True)
if not vis_df.empty:
    base = alt.Chart(vis_df).mark_bar().encode(
        x=alt.X("start:Q", title="Time (s)"),
        x2="end:Q",
        y=alt.Y("type:N", title="", sort=["censor","token"]),
        color=alt.Color("type:N", scale=alt.Scale(domain=["censor","token"], range=["#4cc9f0","#999999"])),
        tooltip=["start","end","type"]
    ).properties(height=80, width=900)
    st.altair_chart(base, use_container_width=True)
else:
    st.info("No tokens to display yet.")

# ------------- step 3: preview around a row (before/after)
st.subheader("3) Quick preview around a selection")

try:
    full_wav = tmpdir / "full.wav"
    ffmpeg_extract_audio_full(src_path, full_wav)
    audio = AudioSegment.from_file(full_wav)

    if not df.empty:
        pick_idx = st.selectbox("Pick an idx to preview", df["idx"].tolist())
        row = df[df["idx"]==pick_idx].iloc[0]
        pre_clip = extract_preview(audio, int(row.start_ms), int(row.end_ms), pad_ms=250)

        # build intervals using current df censor flags
        tmp_hits = []
        from collections import namedtuple
        H = namedtuple("H", "idx word start_ms end_ms prob matched censor matched_with")
        for r in df.itertuples(index=False):
            tmp_hits.append(H(r.idx, r.word, int(r.start_ms), int(r.end_ms), float(r.prob), bool(r.matched), bool(r.censor), r.matched_with))
        intervals = merge_intervals_from_hits(tmp_hits, margin_ms=int(margin))
        post_clip_full = build_censored_audio(
            audio, intervals, freq=int(freq), gain_db=int(gain), mode=mute_mode, duck_db=int(duck_db)
        )
        post_clip = extract_preview(post_clip_full, int(row.start_ms), int(row.end_ms), pad_ms=250)

        st.write("**Before:**")
        st.audio(pre_clip.export(io.BytesIO(), format="mp3").getvalue(), format="audio/mp3")
        st.write("**After (with current settings):**")
        st.audio(post_clip.export(io.BytesIO(), format="mp3").getvalue(), format="audio/mp3")
    else:
        st.info("No tokens available to preview.")
except Exception as e:
    st.warning("Preview unavailable for this file.")
    st.caption(str(e))

st.divider()

# ------------- export / import review
st.subheader("Export / Import review")
colA, colB, colC, colD = st.columns(4)
with colA:
    if st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{src_path.stem}.review.csv",
        mime="text/csv",
        use_container_width=True
    ):
        st.toast("CSV exported", icon="✅")
with colB:
    if st.download_button(
        "⬇️ Download JSON",
        json.dumps({"meta": meta, "hits": df.to_dict(orient="records")}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{src_path.stem}.review.json",
        mime="application/json",
        use_container_width=True
    ):
        st.toast("JSON exported", icon="✅")
with colC:
    def srt_fmt(ms):
        h = ms//3600000; ms%=3600000
        m = ms//60000;   ms%=60000
        s = ms//1000;    ms%=1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    srt_lines = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        srt_lines.extend([str(i), f"{srt_fmt(int(r.start_ms))} --> {srt_fmt(int(r.end_ms))}",
                          f"[{'CENSOR' if r.censor else 'KEEP'}] {r.word}", ""])
    srt_blob = "\n".join(srt_lines).encode("utf-8")
    if st.download_button("⬇️ Download SRT", srt_blob, file_name=f"{src_path.stem}.review.srt", mime="text/plain", use_container_width=True):
        st.toast("SRT exported", icon="✅")
with colD:
    edited_csv = st.file_uploader("Upload edited CSV to re-apply", type=["csv"], key="csv_in", label_visibility="collapsed")
    if edited_csv:
        newdf = pd.read_csv(edited_csv)
        if "idx" in newdf.columns and "censor" in newdf.columns:
            map2 = {int(r["idx"]): bool(r["censor"]) for _, r in newdf.iterrows()}
            for i in range(len(df)):
                if df.loc[i, "idx"] in map2:
                    df.loc[i, "censor"] = map2[df.loc[i, "idx"]]
            st.toast("Applied edited CSV", icon="✅")
        else:
            st.toast("CSV must contain 'idx' and 'censor' columns", icon="⚠️")

# ------------- step 4: render
st.subheader("4) Render")
col1, col2 = st.columns([2,1])
with col1:
    render_btn = st.button("🎬 Render censored output", use_container_width=True)
with col2:
    out_name = st.text_input("Output name (optional)", value=f"{src_path.stem}_censored")

if render_btn:
    with st.status("Rendering…", expanded=True) as s:
        try:
            from collections import namedtuple
            H = namedtuple("H", "idx word start_ms end_ms prob matched censor matched_with")
            tmp_hits = [H(r.idx, r.word, int(r.start_ms), int(r.end_ms), float(r.prob), bool(r.matched), bool(r.censor), r.matched_with)
                        for r in df.itertuples(index=False)]
            intervals = merge_intervals_from_hits(tmp_hits, margin_ms=int(margin))

            full_wav = tmpdir / "full.wav"
            ffmpeg_extract_audio_full(src_path, full_wav)
            audio = AudioSegment.from_file(full_wav)
            censored = build_censored_audio(
                audio, intervals,
                freq=int(freq), gain_db=int(gain),
                mode=mute_mode, duck_db=int(duck_db)
            )

            if src_path.suffix.lower() in {".mp4",".mov",".mkv",".avi",".webm",".m4v"}:
                cens_wav = tmpdir / "censored.wav"
                censored.export(cens_wav, format="wav")
                out_path = tmpdir / f"{out_name}.mp4"
                ffmpeg_mux_audio(src_path, cens_wav, out_path)
                s.update(label="Rendering ✓", state="complete")
                st.success("Done! Download your censored video:")
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download MP4", data=f, file_name=out_path.name, mime="video/mp4")
            else:
                out_path = tmpdir / f"{out_name}.wav"
                censored.export(out_path, format="wav")
                s.update(label="Rendering ✓", state="complete")
                st.success("Done! Download your censored audio:")
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download WAV", data=f, file_name=out_path.name, mime="audio/wav")

        except Exception as e:
            s.update(label="Rendering ✗", state="error")
            st.exception(e)

st.caption("tip: tweak padding/freq/gain on the left, use the table to toggle censor per word, preview, then render.")
