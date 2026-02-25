"""
Motion video conversion for BehaveAI.

Converts standard video to a chromatic motion-enhanced video where RGB channels
encode temporal distance of movement:
  - Blue:  recent motion
  - Green: medium-term motion
  - Red:   older motion
"""

import os
import glob
import platform
import shutil
import subprocess
import time
from pathlib import Path

import click
import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm", ".ts", ".mpg", ".mpeg"}

# OpenCV's standard pip wheel on Windows has no H.264 encoder — all H.264 variants
# remap to avc1/openh264 which requires a DLL that is rarely present. Use mp4v
# directly on Windows to avoid noisy failed attempts.
# On Linux/macOS, try H.264 first for smaller files.
if platform.system() == "Windows":
    _CODEC_PREFERENCE = ["mp4v"]
else:
    _CODEC_PREFERENCE = ["avc1", "X264", "H264", "mp4v"]


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    click.echo("")
    click.echo(click.style(text, bold=True, fg="cyan"))


def _item(label: str, value: str) -> None:
    click.echo(f"  {click.style(label, dim=True)}: {value}")


def _warn(text: str) -> None:
    click.echo(click.style(f"  Warning: {text}", fg="yellow"), err=True)


def _progress(current: int, total: int) -> None:
    pct    = 100 * current / total if total > 0 else 0
    bar_w  = 30
    filled = int(bar_w * current / total) if total > 0 else 0
    bar    = click.style("#" * filled, fg="cyan") + click.style("-" * (bar_w - filled), dim=True)
    width  = len(str(total))
    counter = click.style(f"{current:{width}d}/{total}", dim=True)
    click.echo(f"\r  [{bar}] {counter} {pct:5.1f}%", nl=False)


def _done_line() -> None:
    click.echo("")


# ---------------------------------------------------------------------------
# FFmpeg compression
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _compress_with_ffmpeg(path: str, crf: int = 23) -> None:
    """Re-encode a video with FFmpeg H.264 in-place, replacing the original."""
    tmp = path + ".tmp_compress.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-vcodec", "libx264", "-crf", str(crf),
        "-preset", "fast",
        "-movflags", "+faststart",
        tmp,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, path)
    except subprocess.CalledProcessError as e:
        _warn(f"FFmpeg compression failed: {e.stderr.decode(errors='replace').strip()}")
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Codec selection
# ---------------------------------------------------------------------------

def _pick_codec(output_path: str, w: int, h: int, fps: float):
    """
    Try codecs in preference order, validate by writing a test frame,
    and return (codec_name, writer) for the first one that actually works.
    """
    test_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for codec in _CODEC_PREFERENCE:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer.release()
            continue
        writer.write(test_frame)
        # If the write silently failed the file will be empty or missing
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return codec, writer
        writer.release()
        # Remove the broken file before trying the next codec
        try:
            os.remove(output_path)
        except OSError:
            pass
    # Hard fallback — mp4v always works with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return "mp4v", cv2.VideoWriter(output_path, fourcc, fps, (w, h))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_motion_video(
    input_path: str,
    output_path: str,
    strategy: str = "exponential",
    exp_a: float = 0.5,
    exp_b: float = 0.8,
    lum_weight: float = 0.7,
    rgb_multipliers: tuple = (4.0, 4.0, 4.0),
    chromatic_tail_only: bool = False,
    scale_factor: float = 1.0,
    frame_skip: int = 0,
    motion_threshold: int = 0,
    compress: bool = False,
    crf: int = 23,
) -> None:
    """
    Convert a single video to a motion-enhanced video.

    The output encodes motion history chromatically: blue channel shows the most
    recent movement, green shows medium-term, and red shows older movement.

    :param input_path: Path to the input video file.
    :param output_path: Path for the output video file.
    :param strategy: Frame accumulation strategy, either 'exponential' or 'sequential'.
    :param exp_a: Exponential decay for the green (medium) channel (0-1).
    :param exp_b: Exponential decay for the red (older) channel (0-1).
    :param lum_weight: Blend weight of the original luminance into the output (0-1).
    :param rgb_multipliers: Scaling factors for the (red, green, blue) motion channels.
    :param chromatic_tail_only: If True, suppress base luminance from motion channels.
    :param scale_factor: Resize factor applied to each frame before processing.
    :param frame_skip: Number of frames to skip between processed frames.
    :param motion_threshold: Brightness offset applied to the output.
    :param compress: If True, re-encode the output with FFmpeg H.264 after writing.
    :param crf: H.264 quality level for compression (lower = better, 18-28 is typical).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w = int(src_w * scale_factor)
    h = int(src_h * scale_factor)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    codec, writer = _pick_codec(output_path, w, h, fps)

    _header(Path(input_path).name)
    _item("resolution", f"{src_w}x{src_h}" + (f" -> {w}x{h}" if scale_factor != 1.0 else ""))
    _item("frames",     str(total_frames))
    _item("fps",        f"{fps:.2f}")
    _item("strategy",   strategy)
    _item("codec",      codec)
    _item("output",     output_path)

    exp_a2           = 1.0 - exp_a
    exp_b2           = 1.0 - exp_b
    threshold_offset = -1 * abs(motion_threshold)
    step             = frame_skip + 1

    prev_frames = None
    frame_idx   = 0
    written     = 0
    start_time  = time.perf_counter()

    click.echo("")
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        if scale_factor != 1.0:
            raw_frame = cv2.resize(raw_frame, (w, h))

        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        if prev_frames is None:
            # Pre-allocate accumulator frames to allow in-place addWeighted
            prev_frames = [gray, gray.copy(), gray.copy()]
            frame_idx += 1
            continue

        diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

        if strategy == "exponential":
            prev_frames[0] = gray
            cv2.addWeighted(prev_frames[1], exp_a, gray, exp_a2, 0, dst=prev_frames[1])
            cv2.addWeighted(prev_frames[2], exp_b, gray, exp_b2, 0, dst=prev_frames[2])
        else:  # sequential
            prev_frames[2] = prev_frames[1]
            prev_frames[1] = prev_frames[0]
            prev_frames[0] = gray

        if chromatic_tail_only:
            b_src = cv2.subtract(diffs[0], diffs[1])
            g_src = cv2.subtract(diffs[1], diffs[0])
            r_src = cv2.subtract(diffs[2], diffs[1])
        else:
            b_src, g_src, r_src = diffs[0], diffs[1], diffs[2]

        blue  = cv2.addWeighted(gray, lum_weight, b_src, rgb_multipliers[2], threshold_offset)
        green = cv2.addWeighted(gray, lum_weight, g_src, rgb_multipliers[1], threshold_offset)
        red   = cv2.addWeighted(gray, lum_weight, r_src, rgb_multipliers[0], threshold_offset)

        writer.write(cv2.merge((blue, green, red)))
        written   += 1
        frame_idx += 1

        if written % 30 == 0:
            _progress(frame_idx, total_frames)

    _done_line()
    cap.release()
    writer.release()

    elapsed = time.perf_counter() - start_time
    out_mb  = Path(output_path).stat().st_size / 1_048_576 if Path(output_path).exists() else 0
    _item("written", f"{written} frames in {elapsed:.1f}s ({written / elapsed:.0f} fps)")
    _item("size",    f"{out_mb:.1f} MB")

    if compress:
        if _ffmpeg_available():
            _item("compressing", f"FFmpeg H.264 (crf={crf})...")
            _compress_with_ffmpeg(output_path, crf=crf)
            compressed_mb = Path(output_path).stat().st_size / 1_048_576 if Path(output_path).exists() else 0
            _item("compressed", f"{compressed_mb:.1f} MB ({100 * compressed_mb / out_mb:.0f}% of original)")
        else:
            _warn("--compress requested but ffmpeg not found in PATH; skipping compression.")


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def process_motion_batch(
    input_path: str,
    output_path: str,
    **kwargs,
) -> None:
    """
    Convert one video or a folder of videos to motion-enhanced output.

    If input_path is a directory, all video files within it are processed and
    written to output_path (created as a directory). If input_path is a single
    file, output_path is treated as the output file path.

    :param input_path: Path to a video file or directory of video files.
    :param output_path: Output file path (single video) or output directory (folder input).
    :param kwargs: Passed through to :func:`process_motion_video`, including
        ``compress`` and ``crf``.
    """
    input_path  = str(input_path)
    output_path = str(output_path)

    if os.path.isdir(input_path):
        videos = sorted(
            f for f in glob.glob(os.path.join(input_path, "*"))
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise ValueError(f"No video files found in: {input_path}")
        os.makedirs(output_path, exist_ok=True)
        click.echo(click.style(f"Processing {len(videos)} video(s) -> {output_path}", bold=True))
        for i, vid in enumerate(videos, 1):
            out = os.path.join(output_path, Path(vid).stem + "_motion.mp4")
            click.echo(click.style(f"\n[{i}/{len(videos)}]", dim=True), nl=False)
            process_motion_video(vid, out, **kwargs)
    else:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        process_motion_video(input_path, output_path, **kwargs)

    click.echo("")
    click.echo(click.style("Done.", bold=True, fg="cyan"))