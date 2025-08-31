#!/usr/bin/env python3
"""
demux_media.py — Separate audio, video, and subtitles from any media container without re-encoding.

Usage:
  python demux_media.py /path/to/input.mp4 --outdir out --subs
"""

import argparse
import json
import subprocess
from pathlib import Path

# Map FFmpeg codec names -> common file extensions
AUDIO_EXT = {
    "aac": "m4a",
    "mp3": "mp3",
    "flac": "flac",
    "opus": "opus",
    "vorbis": "ogg",
    "eac3": "eac3",
    "ac3": "ac3",
    "dts": "dts",
    "pcm_s16le": "wav",
    "pcm_s24le": "wav",
    "alac": "m4a",
}
SUB_EXT = {
    "subrip": "srt",
    "ass": "ass",
    "ssa": "ssa",
    "webvtt": "vtt",
    "mov_text": "srt",
}

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def ffprobe_streams(infile):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language,title",
        "-of", "json", str(infile)
    ]
    out = run(cmd).stdout.decode("utf-8")
    data = json.loads(out)
    return data.get("streams", [])

def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="Input media file (any container: mp4, mkv, avi, mov, …)")
    p.add_argument("--outdir", type=Path, default=Path("."), help="Output directory")
    p.add_argument("--subs", action="store_true", help="Also extract subtitle tracks")
    args = p.parse_args()

    infile = args.input
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise SystemExit(f"Input not found: {infile}")

    streams = ffprobe_streams(infile)
    base = sanitize(infile.stem)
    container_ext = infile.suffix.lstrip(".")  # keep same container type

    # --- Extract VIDEO only ---
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if video_streams:
        video_out = outdir / f"{base}_video_only.{container_ext}"
        cmd = ["ffmpeg", "-y", "-i", str(infile), "-map", "0:v:0", "-c", "copy", str(video_out)]
        print("Extracting video ->", video_out.name)
        subprocess.run(cmd, check=True)
    else:
        print("No video streams found.")

    # --- Extract AUDIO tracks ---
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    for s in audio_streams:
        stream_index = s["index"]
        codec = s.get("codec_name", "audio")
        tags = s.get("tags", {}) or {}
        lang = tags.get("language", "und").lower()
        title = tags.get("title", "")
        ext = AUDIO_EXT.get(codec, codec)

        title_part = f"_{sanitize(title)}" if title else ""
        out_path = outdir / f"{base}_a{stream_index}_{lang}{title_part}.{ext}"

        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(infile), "-map", f"0:{stream_index}", "-c", "copy", str(out_path)]
        print(f"Extracting audio #{stream_index} ({codec}, {lang}) -> {out_path.name}")
        subprocess.run(ffmpeg_cmd, check=True)

    if not audio_streams:
        print("No audio streams found.")

    # --- Extract SUBTITLES (optional) ---
    if args.subs:
        sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
        for s in sub_streams:
            stream_index = s["index"]
            codec = s.get("codec_name", "sub")
            tags = s.get("tags", {}) or {}
            lang = tags.get("language", "und").lower()
            title = tags.get("title", "")
            ext = SUB_EXT.get(codec, f"{codec}.sub")

            title_part = f"_{sanitize(title)}" if title else ""
            out_path = outdir / f"{base}_s{stream_index}_{lang}{title_part}.{ext}"

            ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(infile), "-map", f"0:{stream_index}", "-c", "copy", str(out_path)]
            print(f"Extracting subtitle #{stream_index} ({codec}, {lang}) -> {out_path.name}")
            subprocess.run(ffmpeg_cmd, check=True)

        if not sub_streams:
            print("No subtitle streams found or --subs not set.")

    print("Done.")

if __name__ == "__main__":
    main()
