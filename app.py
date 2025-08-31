import streamlit as st
import os, tempfile, subprocess, csv, base64, requests
import soundfile as sf
import numpy as np
from datetime import datetime

st.set_page_config(page_title="JA AV Dubbing", layout="wide")
st.title("Japanese AV Dubbing – quick demo")

# ---------- Config (edit quickly) ----------
WHISPER_MODEL = st.selectbox("Whisper model", ["small", "medium", "large-v3"], index=2)
FORCE_LANG = "ja"
RUN_DIARIZATION = st.checkbox("Run speaker diarization (Resemblyzer+Spectral)", value=True)
RUN_TRANSLATION = st.checkbox("Translate to English", value=True)
RUN_TTS = st.checkbox("Generate Murf TTS & stitch", value=False)
RUN_MERGE = st.checkbox("Merge dubbed audio back to video", value=False)
MURF_API_KEY = st.text_input("MURF_API_KEY (needed if TTS on)", type="password")
DEFAULT_VOICE_ID = st.text_input("Default Murf voice_id", "en-US-ken")
DEFAULT_VOICE_STYLE = "Conversational"

# ---------- Small helpers ----------
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg","-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        st.error("FFmpeg not found. Install FFmpeg and re-run.")
        st.stop()

def standardize_to_mono16k(in_path, out_path):
    run(["ffmpeg","-y","-i", in_path, "-ac","1","-ar","16000","-vn", out_path])

def slice_audio(wav, sr, t0, t1):
    i0 = max(0, int(round(t0*sr))); i1 = min(len(wav), int(round(t1*sr)))
    return wav[i0:i1]

@st.cache_data(show_spinner=False)
def load_model(name):
    import torch, whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(name, device=device), device

def transcribe(tmp_wav, model, device):
    kwargs = dict(fp16=(device=="cuda"), word_timestamps=False, condition_on_previous_text=False, language=FORCE_LANG, task="transcribe")
    return model.transcribe(tmp_wav, **kwargs)

def diarize_segments(wav, sr, segments, k_min=2, k_max=6):
    # fast, minimal version of your notebook logic
    from resemblyzer import VoiceEncoder, preprocess_wav
    from spectralcluster import SpectralClusterer
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.metrics import silhouette_score
    def overlaps(a0,a1,b0,b1,eps=1e-6): return not (a1 <= b0+eps or b1 <= a0+eps)

    enc = VoiceEncoder()
    merged = []
    for seg in segments:
        if not merged: merged.append(seg.copy())
        else:
            last = merged[-1]
            dur = seg["end"] - seg["start"]
            if dur < 0.35:
                last["end"] = seg["end"]
                last["text"] = (last.get("text","")+" "+seg.get("text","")).strip()
            else:
                merged.append(seg.copy())

    # embeddings
    embs = []
    wav_mono, _ = sf.read(tmp_wav_global)
    if wav_mono.ndim > 1: wav_mono = wav_mono.mean(axis=1)
    for m in merged:
        chunk = slice_audio(wav_mono, sr, m["start"], m["end"])
        if len(chunk) < int(0.2*sr):
            pad = np.pad(chunk, (0, int(0.2*sr)-len(chunk)), mode="edge")
        else:
            pad = chunk
        mwav = preprocess_wav(pad, source_sr=sr)
        embs.append(enc.embed_utterance(mwav))
    if not embs:
        return ["Speaker 1"]*len(segments), {"Speaker 1":0}

    import numpy as np
    embs = np.vstack(embs)
    def run_spectral(e,k):
        cl = SpectralClusterer(min_clusters=k, max_clusters=k, p_percentile=0.90, gaussian_blur_sigma=1)
        return np.array(cl.predict(e))

    if len(embs) == 1:
        best_labels = np.array([0])
        best_k = 1
    else:
        cos_d = cosine_distances(embs)
        best_k, best_labels, best_score = None, None, -1
        for k in range(k_min, min(k_max, len(embs))+1):
            try:
                labels = run_spectral(embs, k)
                if len(np.unique(labels)) < 2: continue
                score = silhouette_score(cos_d, labels, metric="precomputed")
                if score > best_score: best_k, best_labels, best_score = k, labels, score
            except: pass
        if best_labels is None:
            best_k, best_labels = 1, np.zeros(len(embs), dtype=int)

    # map back
    def assign_labels(segments, merged, mlabels):
        out = [None]*len(segments)
        m_idx = 0
        for i, seg in enumerate(segments):
            s0,s1 = seg["start"], seg["end"]
            while m_idx < len(merged) and merged[m_idx]["end"] <= s0: m_idx += 1
            cands = []
            for mj in (m_idx, m_idx+1):
                if 0 <= mj < len(merged):
                    ms0,ms1 = merged[mj]["start"], merged[mj]["end"]
                    if overlaps(s0,s1,ms0,ms1):
                        ov = max(0.0, min(s1,ms1)-max(s0,ms0))
                        cands.append((ov, mlabels[mj]))
            lbl = sorted(cands, key=lambda x:-x[0])[0][1] if cands else 0
            out[i] = int(lbl)
        return out

    labels_per_seg = assign_labels(segments, merged, best_labels)
    order, id2name = [], {}
    for l in labels_per_seg:
        if l not in order: order.append(l)
    for i,l in enumerate(order):
        id2name[l] = f"Speaker {i+1}"
    names = [id2name[l] for l in labels_per_seg]
    return names, id2name

def translate_texts(texts):
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source="ja", target="en")
    out = []
    for t in texts:
        try: out.append(tr.translate(t) if t else "")
        except: out.append("")
    return out

def murf_tts_lines(rows, out_dir, api_key, default_voice, default_style):
    from murf import Murf
    client = Murf(api_key=api_key)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, row in enumerate(rows, 1):
        text = row.get("english_translation") or row.get("text") or ""
        if not text: continue
        voice_id = row.get("voice_id") or default_voice
        style = row.get("style") or default_style
        resp = client.text_to_speech.generate(text=text, voice_id=voice_id, style=style)
        # normalize bytes/url/base64
        af = getattr(resp, "audio_file", None)
        if isinstance(af, (bytes, bytearray)): audio_bytes = bytes(af)
        elif isinstance(af, str) and af:
            s = af.strip()
            if s.startswith(("http://","https://")):
                r = requests.get(s, timeout=60); r.raise_for_status(); audio_bytes = r.content
            else:
                audio_bytes = base64.b64decode(s)
        else:
            url = getattr(resp, "audio_url", None) or getattr(resp, "url", None)
            if isinstance(url, str) and url.startswith(("http://","https://")):
                r = requests.get(url, timeout=60); r.raise_for_status(); audio_bytes = r.content
            else:
                raise TypeError("Murf response has no audio bytes/url")
        path = os.path.join(out_dir, f"{i:04d}.wav")
        with open(path, "wb") as f: f.write(audio_bytes)
        paths.append(path)
    # stitch
    from pydub import AudioSegment
    GAP_MS = 700
    combined = AudioSegment.silent(duration=0)
    gap = AudioSegment.silent(duration=GAP_MS)
    for idx, p in enumerate(paths):
        combined += AudioSegment.from_file(p)
        if idx < len(paths)-1: combined += gap
    mix = os.path.join(out_dir, "conversation.wav")
    combined.export(mix, format="wav")
    return mix

def merge_video_audio(video_path, audio_path, output_path):
    run(["ffmpeg","-y","-i", video_path, "-i", audio_path, "-c:v","copy","-c:a","aac","-shortest", output_path])

# ---------- UI ----------
ensure_ffmpeg()
uploaded = st.file_uploader("Upload a video (mkv/mp4/mov) or audio (wav/mp3/m4a)", type=["mkv","mp4","mov","wav","mp3","m4a"])
if not uploaded:
    st.info("Upload a file to start.")
    st.stop()

with tempfile.TemporaryDirectory() as td:
    in_path = os.path.join(td, uploaded.name)
    with open(in_path, "wb") as f: f.write(uploaded.read())

    # standardize
    tmp_wav = os.path.join(td, "audio_16k_mono.wav")
    st.write("🎛️ Standardizing audio (mono, 16 kHz)…")
    standardize_to_mono16k(in_path, tmp_wav)

    # transcribe
    st.write("📝 Transcribing with Whisper…")
    model, device = load_model(WHISPER_MODEL)
    global tmp_wav_global
    tmp_wav_global = tmp_wav  # for diarizer helper
    result = transcribe(tmp_wav, model, device)
    segments = result.get("segments", [])
    if not segments:
        st.error("No speech detected / transcription empty.")
        st.stop()

    # diarize (optional)
    names = ["Speaker 1"]*len(segments)
    id2name = {"Speaker 1": 0}
    if RUN_DIARIZATION:
        st.write("🧑‍🤝‍🧑 Diarizing speakers…")
        wav_arr, sr = sf.read(tmp_wav)
        if wav_arr.ndim > 1: wav_arr = wav_arr.mean(axis=1)
        names, id2name = diarize_segments(wav_arr, 16000, segments)

    # build CSV rows
    csv_rows = []
    texts = []
    for seg, name in zip(segments, names):
        row = {
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "duration": 0.0,
            "speaker": name,
            "text": (seg.get("text","") or "").strip().replace("\n"," ")
        }
        row["duration"] = round(row["end"] - row["start"], 3)
        csv_rows.append(row); texts.append(row["text"])

    # translate (optional)
    if RUN_TRANSLATION:
        st.write("🌐 Translating JA → EN…")
        en = translate_texts(texts)
        for r, t in zip(csv_rows, en):
            r["english_translation"] = t

    # save CSV
    out_dir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "transcript.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    st.success("CSV ready ✅")
    st.download_button("Download CSV", data=open(csv_path,"rb").read(), file_name="transcript.csv", mime="text/csv")

    # TTS (optional)
    mix_path = None
    if RUN_TTS:
        if not MURF_API_KEY:
            st.error("Provide MURF_API_KEY to run TTS.")
        else:
            st.write("🔊 Generating Murf TTS & stitching…")
            # map speakers → voices (simple default mapping)
            for r in csv_rows:
                r["voice_id"] = DEFAULT_VOICE_ID
                r["style"] = DEFAULT_VOICE_STYLE
            mix_path = murf_tts_lines(csv_rows, os.path.join(out_dir, "tts"), MURF_API_KEY, DEFAULT_VOICE_ID, DEFAULT_VOICE_STYLE)
            st.audio(mix_path)

    # Merge back to video (optional)
    if RUN_MERGE:
        if not mix_path:
            st.error("Run TTS first to create conversation.wav.")
        else:
            st.write("🎬 Merging dubbed audio back to video…")
            final_path = os.path.join(out_dir, "final_with_audio.mkv")
            merge_video_audio(in_path, mix_path, final_path)
            with open(final_path, "rb") as f:
                st.download_button("Download final video", f.read(), file_name="final_with_audio.mkv", mime="video/x-matroska")

st.caption("Tip: uncheck diarization/translation if you’re short on time or RAM.")
