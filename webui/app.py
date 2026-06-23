import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "webui" / "jobs"
UPLOADS_DIR = ROOT / "webui" / "uploads"
UPLOAD_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEFAULT_CHUNK_SECONDS = 8
DEFAULT_MAX_WIDTH = 360
VIDEO_ENCODER = os.environ.get("VRT_VIDEO_ENCODER", "libopenh264")
VIDEO2X_ROOT = ROOT / "tools" / "video2x" / "squashfs-root"
VIDEO2X_BIN = VIDEO2X_ROOT / "usr" / "bin" / "video2x"


@dataclass(frozen=True)
class Treatment:
    label: str
    task: str
    tile: tuple[int, int, int]
    overlap: tuple[int, int, int]
    sigma: int = 0
    notes: str = ""
    kind: str = "vrt"
    ffmpeg_filter: str = ""
    video2x_processor: str = ""
    video2x_model: str = ""
    video2x_scale: int = 2


TREATMENTS = {
    "vcr_cleanup": Treatment(
        "VCR cleanup",
        "",
        (0, 0, 0),
        (0, 0, 0),
        notes="Deinterlaces, reduces tape noise/color bleed, restores contrast, and applies mild sharpening.",
        kind="ffmpeg",
        ffmpeg_filter=(
            "bwdif=mode=send_field:parity=auto:deint=all,"
            "hqdn3d=4:3:6:4,"
            "chromanr=thres=24:sizew=5:sizeh=5,"
            "eq=contrast=1.05:saturation=1.08:gamma=1.02,"
            "unsharp=5:5:0.55:3:3:0.25"
        ),
    ),
    "vcr_cleanup_bright": Treatment(
        "VCR cleanup, brighten",
        "",
        (0, 0, 0),
        (0, 0, 0),
        notes="For dark VHS/VCR footage. Deinterlaces, reduces tape noise/color bleed, lifts brightness, restores contrast, and sharpens.",
        kind="ffmpeg",
        ffmpeg_filter=(
            "bwdif=mode=send_field:parity=auto:deint=all,"
            "hqdn3d=4:3:6:4,"
            "chromanr=thres=24:sizew=5:sizeh=5,"
            "eq=brightness=0.06:contrast=1.12:saturation=1.12:gamma=1.08,"
            "unsharp=5:5:0.7:3:3:0.3"
        ),
    ),
    "vcr_cleanup_sharp": Treatment(
        "VCR cleanup, sharper",
        "",
        (0, 0, 0),
        (0, 0, 0),
        notes="A stronger clarity pass for soft VHS/VCR footage. Use when the normal cleanup still looks blurry.",
        kind="ffmpeg",
        ffmpeg_filter=(
            "bwdif=mode=send_field:parity=auto:deint=all,"
            "hqdn3d=3:2:5:3,"
            "chromanr=thres=20:sizew=5:sizeh=5,"
            "eq=brightness=0.04:contrast=1.14:saturation=1.1:gamma=1.05,"
            "unsharp=7:7:0.95:5:5:0.45"
        ),
    ),
    "vcr_cleanup_upscale_2x": Treatment(
        "VCR cleanup + 2x upscale",
        "",
        (0, 0, 0),
        (0, 0, 0),
        notes="Best next test for low-res tape. Cleans, brightens, sharpens, then upscales 2x with Lanczos.",
        kind="ffmpeg",
        ffmpeg_filter=(
            "bwdif=mode=send_field:parity=auto:deint=all,"
            "hqdn3d=4:3:6:4,"
            "chromanr=thres=24:sizew=5:sizeh=5,"
            "eq=brightness=0.06:contrast=1.12:saturation=1.12:gamma=1.08,"
            "unsharp=5:5:0.7:3:3:0.3,"
            "scale=iw*2:ih*2:flags=lanczos,"
            "unsharp=5:5:0.45:3:3:0.2"
        ),
    ),
    "video2x_realesrgan_x2": Treatment(
        "Video2X Real-ESRGAN x2",
        "",
        (0, 0, 0),
        (0, 0, 0),
        notes="Newer Vulkan/ncnn upscaler. Best first AI test for real-world VHS footage.",
        kind="video2x",
        video2x_processor="realesrgan",
        video2x_model="realesrgan-plus",
        video2x_scale=2,
    ),
    "video2x_realesrgan_x4": Treatment(
        "Video2X Real-ESRGAN x4",
        "",
        (0, 0, 0),
        (0, 0, 0),
        notes="Stronger AI upscale through Video2X. Slower and more artifact-prone than x2.",
        kind="video2x",
        video2x_processor="realesrgan",
        video2x_model="realesrgan-plus",
        video2x_scale=4,
    ),
    "sr_reds": Treatment(
        "4x super-resolution, REDS",
        "001_VRT_videosr_bi_REDS_6frames",
        (40, 128, 128),
        (2, 20, 20),
        notes="Best for bicubic low-resolution video.",
    ),
    "sr_vimeo_bicubic": Treatment(
        "4x super-resolution, Vimeo bicubic",
        "003_VRT_videosr_bi_Vimeo_7frames",
        (32, 128, 128),
        (2, 20, 20),
        notes="A general bicubic super-resolution option.",
    ),
    "sr_vimeo_blur": Treatment(
        "4x super-resolution, Vimeo blur-downsampled",
        "004_VRT_videosr_bd_Vimeo_7frames",
        (32, 128, 128),
        (2, 20, 20),
        notes="Use when the source is low-resolution and softly blurred.",
    ),
    "deblur_gopro": Treatment(
        "Motion deblur, GoPro",
        "006_VRT_videodeblurring_GoPro",
        (18, 192, 192),
        (2, 20, 20),
        notes="Best for camera-motion blur.",
    ),
    "deblur_dvd": Treatment(
        "Motion deblur, DVD",
        "005_VRT_videodeblurring_DVD",
        (12, 256, 256),
        (2, 20, 20),
        notes="Best for dynamic-scene blur.",
    ),
    "deblur_reds": Treatment(
        "Motion deblur, REDS",
        "007_VRT_videodeblurring_REDS",
        (12, 256, 256),
        (2, 20, 20),
        notes="Best for REDS-like motion blur.",
    ),
    "denoise_10": Treatment(
        "Denoise, light",
        "008_VRT_videodenoising_DAVIS",
        (12, 256, 256),
        (2, 20, 20),
        sigma=10,
        notes="For mild Gaussian noise.",
    ),
    "denoise_30": Treatment(
        "Denoise, medium",
        "008_VRT_videodenoising_DAVIS",
        (12, 256, 256),
        (2, 20, 20),
        sigma=30,
        notes="For visible noise.",
    ),
    "denoise_50": Treatment(
        "Denoise, heavy",
        "008_VRT_videodenoising_DAVIS",
        (12, 256, 256),
        (2, 20, 20),
        sigma=50,
        notes="For strong Gaussian noise.",
    ),
}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("VRT_MAX_UPLOAD_MB", "15360")) * 1024 * 1024
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.errorhandler(413)
def upload_too_large(_error):
    max_mb = app.config["MAX_CONTENT_LENGTH"] // 1024 // 1024
    return f"Upload is larger than the current {max_mb} MB limit.", 413


def job_path(job_id: str) -> Path:
    return JOBS_DIR / job_id


def upload_path(upload_id: str) -> Path:
    return UPLOADS_DIR / upload_id


def status_path(job_id: str) -> Path:
    return job_path(job_id) / "status.json"


def write_status(job_id: str, **updates) -> None:
    path = status_path(job_id)
    status = {}
    if path.exists():
        status = json.loads(path.read_text())
    status.update(updates)
    status["updated_at"] = time.time()
    path.write_text(json.dumps(status, indent=2))


def read_status(job_id: str) -> dict:
    path = status_path(job_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def chunks_path(job_id: str) -> Path:
    return job_path(job_id) / "chunks.json"


def read_chunks(job_id: str) -> list[dict]:
    path = chunks_path(job_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def write_chunks(job_id: str, chunks: list[dict]) -> None:
    chunks_path(job_id).write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def update_chunk(job_id: str, index: int, **updates) -> list[dict]:
    chunks = read_chunks(job_id)
    for chunk in chunks:
        if chunk["index"] == index:
            chunk.update(updates)
            break
    write_chunks(job_id, chunks)
    return chunks


def compact_error(message: str) -> str:
    if "ran out of memory" in message or "can't allocate memory" in message or "Cannot allocate memory" in message:
        return (
            "Out of memory. For 4x SR, re-prepare chunks with max input width 320-360, "
            "or use denoise/deblur on full-size chunks."
        )
    return message[:500]


def append_log(job_id: str, message: str) -> None:
    with (job_path(job_id) / "process.log").open("a", encoding="utf-8") as log:
        log.write(message)


def run_command(job_id: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    append_log(job_id, f"\n$ {' '.join(cmd)}\n")
    recent_output = []
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        append_log(job_id, line)
        recent_output.append(line)
        recent_output = recent_output[-30:]
    return_code = process.wait()
    if return_code != 0:
        details = "".join(recent_output)
        if "can't allocate memory" in details or "Cannot allocate memory" in details:
            raise RuntimeError(
                "VRT ran out of memory. For 4x super-resolution, prepare chunks with a smaller max input width "
                "(try 320-360 for 720p-ish output), or use denoise/deblur on the original resolution."
            )
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def ffprobe_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_float = float(denominator)
        return float(numerator) / denominator_float if denominator_float else 30.0
    return float(value or 30.0)


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def ffprobe_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if "x" not in value:
        return 0, 0
    width, height = value.split("x", 1)
    return int(width), int(height)


def has_audio(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return bool(result.stdout.strip())


def process_video_segment(
    job_id: str,
    treatment: Treatment,
    input_video: Path,
    segment_dir: Path,
    output_video: Path,
    label: str,
) -> int:
    frames_dir = segment_dir / "frames" / "upload"
    result_dir = segment_dir / "result_frames"

    frames_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    fps = ffprobe_fps(input_video)
    run_command(
        job_id,
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vsync",
            "0",
            str(frames_dir / "%08d.png"),
        ],
    )

    frame_count = len(list(frames_dir.glob("*.png")))
    if frame_count == 0:
        raise RuntimeError(f"No frames were extracted from {label}.")

    command = [
        sys.executable,
        "main_test_vrt.py",
        "--task",
        treatment.task,
        "--folder_lq",
        str(segment_dir / "frames"),
        "--tile",
        *[str(value) for value in treatment.tile],
        "--tile_overlap",
        *[str(value) for value in treatment.overlap],
        "--num_workers",
        "0",
        "--save_result",
        "--save_dir",
        str(result_dir),
    ]
    if treatment.sigma:
        command.extend(["--sigma", str(treatment.sigma)])
    run_command(job_id, command)

    generated_dir = result_dir / "upload"
    generated_count = len(list(generated_dir.glob("*.png")))
    if generated_count == 0:
        raise RuntimeError(f"VRT finished but did not produce PNG frames for {label}.")

    run_command(
        job_id,
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-start_number",
            "1",
            "-i",
            str(generated_dir / "%08d.png"),
            "-c:v",
            VIDEO_ENCODER,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
    )
    return frame_count


def process_ffmpeg_treatment(
    job_id: str,
    treatment: Treatment,
    input_video: Path,
    output_video: Path,
    include_audio: bool,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
    ]
    if include_audio:
        cmd.extend(["-map", "0:a:0?"])
    else:
        cmd.append("-an")
    cmd.extend(
        [
            "-vf",
            treatment.ffmpeg_filter,
            "-c:v",
            VIDEO_ENCODER,
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if include_audio:
        cmd.extend(["-c:a", "aac", "-shortest"])
    cmd.extend(["-movflags", "+faststart", str(output_video)])
    run_command(job_id, cmd)


def process_video2x_treatment(
    job_id: str,
    treatment: Treatment,
    input_video: Path,
    output_video: Path,
) -> None:
    if not VIDEO2X_BIN.exists():
        raise RuntimeError("Video2X is not installed. Expected tools/video2x/squashfs-root/usr/bin/video2x.")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    video2x_lib = str(VIDEO2X_ROOT / "usr" / "lib")
    env["LD_LIBRARY_PATH"] = f"{video2x_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    command = [
        str(VIDEO2X_BIN),
        "-i",
        str(input_video),
        "-o",
        str(output_video),
        "-p",
        treatment.video2x_processor,
        "--realesrgan-model",
        treatment.video2x_model,
        "-s",
        str(treatment.video2x_scale),
        "-c",
        "libx264",
        "--pix-fmt",
        "yuv420p",
        "--no-progress",
    ]
    run_command(job_id, command, env=env)


def split_video(job_id: str, input_video: Path, chunks_dir: Path, chunk_seconds: int) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        job_id,
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            VIDEO_ENCODER,
            "-force_key_frames",
            f"expr:gte(t,n_forced*{chunk_seconds})",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            str(chunks_dir / "chunk_%04d.mp4"),
        ],
    )
    chunks = sorted(chunks_dir.glob("chunk_*.mp4"))
    if not chunks:
        raise RuntimeError("FFmpeg did not create any chunks.")
    return chunks


def resize_chunks(job_id: str, chunks: list[Path], max_width: int) -> list[Path]:
    if max_width <= 0:
        return chunks

    resized_dir = chunks[0].parent.parent / "resized_chunks"
    resized_dir.mkdir(parents=True, exist_ok=True)
    resized_chunks = []
    for index, chunk in enumerate(chunks):
        resized = resized_dir / f"chunk_{index:04d}.mp4"
        run_command(
            job_id,
            [
                "ffmpeg",
                "-y",
                "-i",
                str(chunk),
                "-vf",
                f"scale='min({max_width},iw)':-2",
                "-an",
                "-c:v",
                VIDEO_ENCODER,
                "-pix_fmt",
                "yuv420p",
                str(resized),
            ],
        )
        resized_chunks.append(resized)
    return resized_chunks


def resize_video(job_id: str, input_video: Path, output_video: Path, max_width: int) -> Path:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        job_id,
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-an",
            "-c:v",
            VIDEO_ENCODER,
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ],
    )
    return output_video


def chunk_source_path(job_id: str, chunk_index: int, prefer_resized: bool = False) -> Path:
    chunks = read_chunks(job_id)
    chunk = next((item for item in chunks if item["index"] == chunk_index), None)
    if chunk is None:
        raise RuntimeError(f"Chunk {chunk_index} does not exist.")

    job_dir = job_path(job_id)
    source = chunk.get("source") or f"chunk_{chunk_index - 1:04d}.mp4"
    folders = ["resized_chunks", "chunks"] if prefer_resized else ["chunks", "resized_chunks"]
    for folder in folders:
        candidate = job_dir / folder / source
        if candidate.exists():
            return candidate
    fallback = job_dir / "chunks" / f"chunk_{chunk_index - 1:04d}.mp4"
    if fallback.exists():
        return fallback
    raise RuntimeError(f"Chunk {chunk_index} source file is missing.")


def prepare_working_chunk(job_id: str, chunk_index: int, treatment: Treatment) -> Path:
    chunk_path = chunk_source_path(job_id, chunk_index)
    width, height = ffprobe_dimensions(chunk_path)
    status = read_status(job_id)
    max_width = int(status.get("max_width") or DEFAULT_MAX_WIDTH)

    if "videosr" not in treatment.task or width <= max_width:
        return chunk_path

    working_path = job_path(job_id) / "working_chunks" / f"chunk_{chunk_index:04d}_w{max_width}.mp4"
    if not working_path.exists():
        write_status(
            job_id,
            state="processing_chunk",
            message=f"Auto-resizing chunk {chunk_index} from {width}x{height} to max width {max_width}",
            chunk_index=chunk_index,
            treatment=treatment.label,
        )
        resize_video(job_id, chunk_path, working_path, max_width)

    new_width, new_height = ffprobe_dimensions(working_path)
    update_chunk(
        job_id,
        chunk_index,
        working_width=new_width,
        working_height=new_height,
        auto_resized=True,
    )
    return working_path


def concat_videos(job_id: str, videos: list[Path], output_video: Path) -> None:
    list_path = output_video.parent / "concat.txt"
    lines = [f"file '{video.resolve().as_posix()}'" for video in videos]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_command(
        job_id,
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_video),
        ],
    )


def mux_original_audio(job_id: str, video_path: Path, original_video: Path, output_video: Path) -> None:
    if not has_audio(original_video):
        shutil.copy2(video_path, output_video)
        return
    run_command(
        job_id,
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(original_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
    )


def job_summary(job_id: str) -> dict:
    status = read_status(job_id)
    chunks = read_chunks(job_id)
    completed = sum(1 for chunk in chunks if chunk.get("state") == "complete")
    return {
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}",
        "filename": status.get("filename", job_id),
        "state": status.get("state", "unknown"),
        "message": status.get("message", ""),
        "treatment": status.get("treatment", ""),
        "chunk_count": len(chunks) or status.get("chunk_count", 0),
        "processed_chunks": completed or status.get("processed_chunks", 0),
        "updated_at": status.get("updated_at", 0),
    }


def process_job(
    job_id: str,
    treatment_key: str,
    input_video: Path,
    chunk_seconds: int = 0,
    max_width: int = 0,
) -> None:
    try:
        treatment = TREATMENTS[treatment_key]
        job_dir = job_path(job_id)
        output_video = job_dir / "output.mp4"
        chunk_seconds = max(0, int(chunk_seconds))
        max_width = max(0, int(max_width))

        if chunk_seconds:
            chunks_dir = job_dir / "chunks"

            duration = ffprobe_duration(input_video)
            write_status(
                job_id,
                state="chunking",
                message=f"Splitting video into {chunk_seconds}-second chunks",
                treatment=treatment.label,
                duration=duration,
            )
            chunks = split_video(job_id, input_video, chunks_dir, chunk_seconds)
            write_chunks(
                job_id,
                [
                    {
                        "index": index,
                        "source": chunk.name,
                        "state": "ready",
                        "download_url": None,
                    }
                    for index, chunk in enumerate(chunks, start=1)
                ],
            )
            write_status(
                job_id,
                state="ready_for_chunks",
                message=f"Prepared {len(chunks)} chunks. Select one chunk and process it with the current treatment.",
                chunk_count=len(chunks),
            )
            return
        else:
            if treatment.kind == "ffmpeg":
                write_status(
                    job_id,
                    state="processing",
                    message=f"Running {treatment.label}",
                    treatment=treatment.label,
                    step_started_at=time.time(),
                )
                process_ffmpeg_treatment(job_id, treatment, input_video, output_video, include_audio=True)
                write_status(
                    job_id,
                    state="complete",
                    message="Processing complete",
                    download_url=f"/jobs/{job_id}/download",
                    output_name=f"vrt-{treatment_key}-{job_id[:8]}.mp4",
                )
                return
            if treatment.kind == "video2x":
                write_status(
                    job_id,
                    state="processing",
                    message=f"Running {treatment.label}",
                    treatment=treatment.label,
                    step_started_at=time.time(),
                )
                process_video2x_treatment(job_id, treatment, input_video, output_video)
                write_status(
                    job_id,
                    state="complete",
                    message="Processing complete",
                    download_url=f"/jobs/{job_id}/download",
                    output_name=f"vrt-{treatment_key}-{job_id[:8]}.mp4",
                )
                return

            write_status(job_id, state="extracting", message="Extracting video frames")
            video_only_output = job_dir / "single-video-only.mp4"
            frame_count = process_video_segment(
                job_id,
                treatment,
                input_video,
                job_dir / "single",
                video_only_output,
                "uploaded video",
            )
            write_status(job_id, frame_count=frame_count)
            write_status(job_id, state="encoding", message="Reattaching original audio")
            mux_original_audio(job_id, video_only_output, input_video, output_video)

        write_status(
            job_id,
            state="complete",
            message="Processing complete",
            download_url=f"/jobs/{job_id}/download",
            output_name=f"vrt-{treatment_key}-{job_id[:8]}.mp4",
        )
    except Exception as exc:
        append_log(job_id, f"\nERROR: {exc}\n")
        write_status(job_id, state="failed", message=str(exc))


def process_chunk_job(job_id: str, chunk_index: int, treatment_key: str) -> None:
    try:
        treatment = TREATMENTS[treatment_key]
        job_dir = job_path(job_id)
        chunk_path = prepare_working_chunk(job_id, chunk_index, treatment)

        processed_dir = job_dir / "processed_chunks"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_chunk = processed_dir / f"processed_{chunk_index:04d}.mp4"

        update_chunk(job_id, chunk_index, state="processing", treatment=treatment.label)
        write_status(
            job_id,
            state="processing_chunk",
            message=f"Running {treatment.label} on chunk {chunk_index}",
            chunk_index=chunk_index,
            treatment=treatment.label,
            step_started_at=time.time(),
        )
        update_chunk(job_id, chunk_index, state="processing", treatment=treatment.label, started_at=time.time())
        if treatment.kind == "ffmpeg":
            process_ffmpeg_treatment(job_id, treatment, chunk_path, processed_chunk, include_audio=False)
            frame_count = 0
        elif treatment.kind == "video2x":
            process_video2x_treatment(job_id, treatment, chunk_path, processed_chunk)
            frame_count = 0
        else:
            frame_count = process_video_segment(
                job_id,
                treatment,
                chunk_path,
                job_dir / "segments" / f"chunk_{chunk_index:04d}",
                processed_chunk,
                f"chunk {chunk_index}",
            )
        output_width, output_height = ffprobe_dimensions(processed_chunk)
        chunks = update_chunk(
            job_id,
            chunk_index,
            state="complete",
            treatment=treatment.label,
            frame_count=frame_count,
            output_width=output_width,
            output_height=output_height,
            download_url=f"/jobs/{job_id}/chunks/{chunk_index}/download",
            completed_at=time.time(),
        )
        for chunk in chunks:
            if chunk["index"] == chunk_index and chunk.get("started_at") and chunk.get("completed_at"):
                chunk["duration_seconds"] = chunk["completed_at"] - chunk["started_at"]
                break
        write_chunks(job_id, chunks)
        completed = sum(1 for chunk in chunks if chunk.get("state") == "complete")
        write_status(
            job_id,
            state="ready_for_chunks",
            message=f"Chunk {chunk_index} complete. {completed} of {len(chunks)} chunks processed.",
            processed_chunks=completed,
            chunk_count=len(chunks),
        )
    except Exception as exc:
        error = compact_error(str(exc))
        append_log(job_id, f"\nERROR: {exc}\n")
        update_chunk(job_id, chunk_index, state="failed", error=error)
        write_status(job_id, state="ready_for_chunks", message=f"Chunk {chunk_index} failed: {error}")


def process_all_chunks_job(job_id: str, treatment_key: str) -> None:
    try:
        treatment = TREATMENTS[treatment_key]
        chunks = read_chunks(job_id)
        if not chunks:
            raise RuntimeError("This job has no chunks to process.")

        completed_treatments = {
            chunk.get("treatment")
            for chunk in chunks
            if chunk.get("state") == "complete" and chunk.get("treatment")
        }
        if completed_treatments and completed_treatments != {treatment.label}:
            choices = ", ".join(sorted(completed_treatments))
            raise RuntimeError(
                f"Some chunks are already complete with a different treatment ({choices}). "
                f"Select that treatment to resume, or process individual chunks to intentionally replace them."
            )

        total = len(chunks)
        for chunk in chunks:
            chunk_treatment = chunk.get("treatment")
            if chunk.get("state") == "complete" and chunk_treatment == treatment.label:
                continue

            chunk_index = int(chunk["index"])
            chunk_path = prepare_working_chunk(job_id, chunk_index, treatment)
            job_dir = job_path(job_id)
            processed_dir = job_dir / "processed_chunks"
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_chunk = processed_dir / f"processed_{chunk_index:04d}.mp4"

            started_at = time.time()
            completed = sum(1 for item in read_chunks(job_id) if item.get("state") == "complete" and item.get("treatment") == treatment.label)
            update_chunk(
                job_id,
                chunk_index,
                state="processing",
                treatment=treatment.label,
                error=None,
                started_at=started_at,
            )
            write_status(
                job_id,
                state="processing_all_chunks",
                message=f"Running {treatment.label} on chunk {chunk_index} of {total}",
                chunk_index=chunk_index,
                treatment=treatment.label,
                processed_chunks=completed,
                chunk_count=total,
                step_started_at=started_at,
            )

            if treatment.kind == "ffmpeg":
                process_ffmpeg_treatment(job_id, treatment, chunk_path, processed_chunk, include_audio=False)
                frame_count = 0
            elif treatment.kind == "video2x":
                process_video2x_treatment(job_id, treatment, chunk_path, processed_chunk)
                frame_count = 0
            else:
                frame_count = process_video_segment(
                    job_id,
                    treatment,
                    chunk_path,
                    job_dir / "segments" / f"chunk_{chunk_index:04d}",
                    processed_chunk,
                    f"chunk {chunk_index}",
                )

            completed_at = time.time()
            output_width, output_height = ffprobe_dimensions(processed_chunk)
            chunks = update_chunk(
                job_id,
                chunk_index,
                state="complete",
                treatment=treatment.label,
                frame_count=frame_count,
                output_width=output_width,
                output_height=output_height,
                download_url=f"/jobs/{job_id}/chunks/{chunk_index}/download",
                completed_at=completed_at,
                duration_seconds=completed_at - started_at,
            )
            completed = sum(1 for item in chunks if item.get("state") == "complete" and item.get("treatment") == treatment.label)
            write_status(
                job_id,
                state="processing_all_chunks",
                message=f"Chunk {chunk_index} complete. {completed} of {total} chunks processed with {treatment.label}.",
                processed_chunks=completed,
                chunk_count=total,
            )

        write_status(job_id, state="stitching", message="All chunks complete. Stitching processed chunks.")
        stitch_job(job_id)
    except Exception as exc:
        error = compact_error(str(exc))
        append_log(job_id, f"\nERROR: {exc}\n")
        write_status(job_id, state="ready_for_chunks", message=f"Automatic processing failed: {error}")


def stitch_job(job_id: str) -> None:
    try:
        job_dir = job_path(job_id)
        chunks = read_chunks(job_id)
        missing = [chunk["index"] for chunk in chunks if chunk.get("state") != "complete"]
        if missing:
            raise RuntimeError(f"Process all chunks before stitching. Missing: {missing}")
        treatments = {chunk.get("treatment") for chunk in chunks}
        if len(treatments) > 1:
            raise RuntimeError("All chunks must use the same treatment before stitching.")

        processed_chunks = [job_dir / "processed_chunks" / f"processed_{chunk['index']:04d}.mp4" for chunk in chunks]
        stitched_video = job_dir / "stitched-video-only.mp4"
        output_video = job_dir / "output.mp4"
        input_candidates = sorted(job_dir.glob("input.*"))
        if not input_candidates:
            raise RuntimeError("Original uploaded video is missing.")

        write_status(job_id, state="stitching", message="Stitching processed chunks")
        concat_videos(job_id, processed_chunks, stitched_video)
        write_status(job_id, state="encoding", message="Reattaching original audio")
        mux_original_audio(job_id, stitched_video, input_candidates[0], output_video)
        write_status(
            job_id,
            state="complete",
            message="Processing complete",
            download_url=f"/jobs/{job_id}/download",
            output_name=f"vrt-chunked-{job_id[:8]}.mp4",
        )
    except Exception as exc:
        append_log(job_id, f"\nERROR: {exc}\n")
        write_status(job_id, state="ready_for_chunks", message=str(exc))


@app.get("/")
def index():
    treatments = {
        key: {"label": value.label, "notes": value.notes}
        for key, value in TREATMENTS.items()
    }
    return render_template(
        "index.html",
        treatments=treatments,
        default_chunk_seconds=DEFAULT_CHUNK_SECONDS,
        default_max_width=DEFAULT_MAX_WIDTH,
    )


@app.get("/repo-assets/<path:filename>")
def repo_assets(filename: str):
    return send_from_directory(ROOT / "assets", filename)


@app.post("/uploads/start")
def start_upload():
    payload = request.get_json(force=True, silent=True) or {}
    filename = secure_filename(payload.get("filename", ""))
    if not filename:
        abort(400, "Missing filename")

    extension = Path(filename).suffix.lower()
    if extension not in UPLOAD_EXTENSIONS:
        abort(400, "Unsupported video type")

    size = int(payload.get("size") or 0)
    if size <= 0:
        abort(400, "Missing file size")
    if size > app.config["MAX_CONTENT_LENGTH"]:
        max_gb = app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024 / 1024
        abort(413, f"Upload is larger than the current {max_gb:.1f} GB limit.")

    treatment_key = payload.get("treatment", "")
    if treatment_key not in TREATMENTS:
        abort(400, "Unknown treatment")

    upload_id = uuid.uuid4().hex
    path = upload_path(upload_id)
    path.mkdir(parents=True)
    metadata = {
        "filename": filename,
        "size": size,
        "extension": extension,
        "treatment": treatment_key,
        "chunk_seconds": int(payload.get("chunk_seconds") or 0),
        "max_width": int(payload.get("max_width") or 0),
        "created_at": time.time(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (path / "upload.part").touch()
    return jsonify({"upload_id": upload_id, "received": 0})


@app.put("/uploads/<upload_id>/part")
def upload_part(upload_id: str):
    path = upload_path(upload_id)
    metadata_path = path / "metadata.json"
    part_path = path / "upload.part"
    if not metadata_path.exists() or not part_path.exists():
        abort(404, "Upload session not found")

    metadata = json.loads(metadata_path.read_text())
    expected_size = int(metadata["size"])
    expected_offset = part_path.stat().st_size
    offset = int(request.headers.get("X-Upload-Offset") or -1)
    if offset != expected_offset:
        abort(409, f"Upload offset mismatch. Server has {expected_offset} bytes.")

    written = 0
    with part_path.open("ab") as output:
        while True:
            data = request.stream.read(1024 * 1024)
            if not data:
                break
            output.write(data)
            written += len(data)

    received = expected_offset + written
    if received > expected_size:
        abort(400, "Uploaded data exceeds declared file size")
    return jsonify({"received": received, "size": expected_size})


@app.post("/uploads/<upload_id>/finish")
def finish_upload(upload_id: str):
    path = upload_path(upload_id)
    metadata_path = path / "metadata.json"
    part_path = path / "upload.part"
    if not metadata_path.exists() or not part_path.exists():
        abort(404, "Upload session not found")

    metadata = json.loads(metadata_path.read_text())
    size = int(metadata["size"])
    received = part_path.stat().st_size
    if received != size:
        abort(409, f"Upload incomplete. Received {received} of {size} bytes.")

    treatment_key = metadata["treatment"]
    if treatment_key not in TREATMENTS:
        abort(400, "Unknown treatment")

    job_id = uuid.uuid4().hex
    job_dir = job_path(job_id)
    job_dir.mkdir(parents=True)
    input_video = job_dir / f"input{metadata['extension']}"
    shutil.move(str(part_path), input_video)

    chunk_seconds = int(metadata.get("chunk_seconds") or 0)
    max_width = int(metadata.get("max_width") or 0)
    write_status(
        job_id,
        state="queued",
        message="Queued",
        filename=metadata["filename"],
        treatment=TREATMENTS[treatment_key].label,
        chunk_seconds=chunk_seconds,
        max_width=max_width,
        created_at=time.time(),
    )
    shutil.rmtree(path, ignore_errors=True)
    thread = threading.Thread(
        target=process_job,
        args=(job_id, treatment_key, input_video, chunk_seconds, max_width),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "status_url": f"/jobs/{job_id}"})


@app.post("/jobs")
def create_job():
    upload = request.files.get("video")
    treatment_key = request.form.get("treatment", "")
    if treatment_key not in TREATMENTS:
        abort(400, "Unknown treatment")
    if not upload or not upload.filename:
        abort(400, "Upload a video file")

    filename = secure_filename(upload.filename)
    extension = Path(filename).suffix.lower()
    if extension not in UPLOAD_EXTENSIONS:
        abort(400, "Unsupported video type")

    job_id = uuid.uuid4().hex
    job_dir = job_path(job_id)
    job_dir.mkdir(parents=True)
    input_video = job_dir / f"input{extension}"
    upload.save(input_video)
    chunk_seconds = int(request.form.get("chunk_seconds") or 0)
    max_width = int(request.form.get("max_width") or 0)

    write_status(
        job_id,
        state="queued",
        message="Queued",
        filename=filename,
        treatment=TREATMENTS[treatment_key].label,
        chunk_seconds=chunk_seconds,
        max_width=max_width,
        created_at=time.time(),
    )
    thread = threading.Thread(
        target=process_job,
        args=(job_id, treatment_key, input_video, chunk_seconds, max_width),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "status_url": f"/jobs/{job_id}"})


@app.get("/jobs")
def list_jobs():
    jobs = []
    for path in JOBS_DIR.glob("*/status.json"):
        jobs.append(job_summary(path.parent.name))
    jobs.sort(key=lambda item: item["updated_at"], reverse=True)
    return jsonify({"jobs": jobs})


@app.get("/jobs/latest")
def latest_job():
    candidates = []
    for path in JOBS_DIR.glob("*/status.json"):
        job_id = path.parent.name
        status = json.loads(path.read_text())
        if read_chunks(job_id):
            candidates.append((status.get("updated_at", path.stat().st_mtime), job_id))
    if not candidates:
        abort(404)
    _, job_id = max(candidates)
    return jsonify({"job_id": job_id, "status_url": f"/jobs/{job_id}"})


@app.get("/jobs/<job_id>")
def get_job(job_id: str):
    path = status_path(job_id)
    if not path.exists():
        abort(404)
    status = json.loads(path.read_text())
    status["chunks"] = read_chunks(job_id)
    log_path = job_path(job_id) / "process.log"
    status["log"] = log_path.read_text(encoding="utf-8")[-5000:] if log_path.exists() else ""
    return jsonify(status)


@app.post("/jobs/<job_id>/chunks/<int:chunk_index>/process")
def process_chunk(job_id: str, chunk_index: int):
    if not status_path(job_id).exists():
        abort(404)
    treatment_key = request.form.get("treatment", "")
    if treatment_key not in TREATMENTS:
        abort(400, "Unknown treatment")
    status = read_status(job_id)
    if status.get("state") in {"processing", "processing_chunk", "chunking", "stitching", "encoding"}:
        abort(409, "A job step is already running")
    chunks = read_chunks(job_id)
    if not any(chunk["index"] == chunk_index for chunk in chunks):
        abort(404, "Unknown chunk")
    thread = threading.Thread(target=process_chunk_job, args=(job_id, chunk_index, treatment_key), daemon=True)
    thread.start()
    return jsonify({"ok": True, "status_url": f"/jobs/{job_id}"})


@app.post("/jobs/<job_id>/chunks/process-all")
def process_all_chunks(job_id: str):
    if not status_path(job_id).exists():
        abort(404)
    treatment_key = request.form.get("treatment", "")
    if treatment_key not in TREATMENTS:
        abort(400, "Unknown treatment")
    status = read_status(job_id)
    if status.get("state") in {"processing", "processing_chunk", "processing_all_chunks", "chunking", "stitching", "encoding"}:
        abort(409, "A job step is already running")
    if not read_chunks(job_id):
        abort(400, "This job has no chunks")
    thread = threading.Thread(target=process_all_chunks_job, args=(job_id, treatment_key), daemon=True)
    thread.start()
    return jsonify({"ok": True, "status_url": f"/jobs/{job_id}"})


@app.get("/jobs/<job_id>/chunks/<int:chunk_index>/download")
def download_chunk(job_id: str, chunk_index: int):
    path = job_path(job_id) / "processed_chunks" / f"processed_{chunk_index:04d}.mp4"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"vrt-chunk-{chunk_index:04d}.mp4")


@app.get("/jobs/<job_id>/samples/<path:filename>/download")
def download_sample(job_id: str, filename: str):
    path = job_path(job_id) / "samples" / secure_filename(filename)
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.post("/jobs/<job_id>/stitch")
def stitch_chunks(job_id: str):
    if not status_path(job_id).exists():
        abort(404)
    status = read_status(job_id)
    if status.get("state") in {"processing", "processing_chunk", "chunking", "stitching", "encoding"}:
        abort(409, "A job step is already running")
    thread = threading.Thread(target=stitch_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "status_url": f"/jobs/{job_id}"})


@app.get("/jobs/<job_id>/download")
def download(job_id: str):
    path = job_path(job_id) / "output.mp4"
    if not path.exists():
        abort(404)
    status = json.loads(status_path(job_id).read_text())
    return send_file(path, as_attachment=True, download_name=status.get("output_name", "vrt-output.mp4"))


@app.post("/jobs/<job_id>/delete")
def delete_job(job_id: str):
    path = job_path(job_id)
    if path.exists():
        shutil.rmtree(path)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "17495")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
