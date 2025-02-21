#frame_extraction.py
import os
import subprocess
import logging

from yt_dlp import YoutubeDL
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
import fiftyone as fo  # Import FiftyOne
import fiftyone.brain as fob

# Configure logging (keep your existing configuration)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("frame_extraction")

def download_video(url):
    # (Keep your existing download_video function - no changes needed)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': '%(id)s.%(ext)s',
        'merge_output_format': 'mp4',
        # 'logger': logger,  # Uncomment if you want ydl_opts to use the logger.
        # 'progress_hooks': [lambda d: logger.info(f"Download progress: {d}")],
    }
    with YoutubeDL(ydl_opts) as ydl:
        logger.info(f"Downloading video from URL: {url}")
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        video_ext = info.get('ext', 'mp4')
        video_path = f"{video_id}.{video_ext}"
        logger.info(f"Downloaded video {video_id}")
        return video_id, video_path


def detect_scenes(video_path, threshold=30.0):
    # (Keep your existing detect_scenes function - no changes needed)
    logger.info(f"Detecting scenes in {video_path}")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor(1)
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    logger.info(f"Scene detection complete: {len(scene_list)} scenes found")
    return scene_list


def extract_keyframes_with_timestamps(video_path, scenes, output_dir="frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}")

    frames = []
    num_scenes = len(scenes)

    # --- Enhanced Scene Logging ---
    logger.info(f"Number of scenes detected: {num_scenes}")
    for idx, scene in enumerate(scenes):
        start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
        logger.info(f"Scene {idx:04d}: Start Time: {start_time:.3f}, End Time: {end_time:.3f}")

    if scenes:
        video_duration_seconds = scenes[-1][1].get_seconds()
        logger.info(f"Estimated video duration: {video_duration_seconds:.3f} seconds")
    else:
        video_duration_seconds = 600  # Default, or fetch from yt-dlp info later if needed.
        logger.warning("No scenes detected, using default video duration.")

    for idx, scene in enumerate(scenes):
        start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()

        # Offsets: start, middle.  Skip end time for the *last* scene.
        offsets = [0, (end_time - start_time) / 2]
        if idx < num_scenes - 1:  # Add end time offset *unless* it's the last scene
            offsets.append(end_time - start_time)

        for t_offset in offsets:
            t_actual = start_time + t_offset
            frame_timestamp = max(0, min(t_actual, video_duration_seconds))
            frame_path = os.path.join(output_dir, f"frame_{idx:04d}_{frame_timestamp:.2f}.png")
            command = f'ffmpeg -y -loglevel error -ss {frame_timestamp:.2f} -i "{video_path}" -frames:v 1 "{frame_path}"'
            logger.info(f"FFMPEG command: {command}")
            subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            frames.append({"timestamp": frame_timestamp, "path": frame_path, "scene_index": idx})
            optimize_frame(frame_path)  # Optimize all extracted frames.

    logger.info(f"Extracted {len(frames)} keyframes")
    return frames


def optimize_frame(frame_path, resolution=(1280, 720), quality=85):
    # (Keep your existing optimize_frame function with debug logs)
    logger.info(f"optimize_frame called for path: {frame_path}")  # Debug log 1: Function called

    try:
        logger.info(f"optimize_frame attempting to open: {frame_path}") # Debug log 2: Before Image.open
        img = Image.open(frame_path)
        logger.info(f"optimize_frame successfully opened: {frame_path}") # Debug log 3: After Image.open success
        img = img.resize(resolution)
        img.save(frame_path, quality=quality)
    except Exception as e:
        logger.error(f"Optimization failed for {frame_path}: {e}") # Original error log
        logger.error(f"optimize_frame exception details: Path: {frame_path}, Error: {e}") # Debug log 4: Exception details


def load_frames_into_fiftyone(frames, video_id):
    dataset_name = f"video-frames-{video_id}"

    # --- Delete dataset if it exists ---
    if dataset_name in fo.list_datasets():
        logger.info(f"Deleting existing FiftyOne dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    for frame_info in frames:
        sample = fo.Sample(filepath=frame_info["path"])
        sample["timestamp"] = frame_info["timestamp"]
        sample["scene_index"] = frame_info["scene_index"]
        dataset.add_sample(sample)
    return dataset


def select_unique_frames(dataset, uniqueness_threshold=0.7):
    # (Keep your existing select_unique_frames function)
    fob.compute_uniqueness(dataset)
    unique_view = dataset.match(fo.ViewField("uniqueness") > uniqueness_threshold)

    # Sort by scene index, then by timestamp within each scene.  This restores chronological order.
    sorted_view = unique_view.sort_by("scene_index").sort_by("timestamp")

    selected_frames = []
    for sample in sorted_view:
        selected_frames.append({
            "timestamp": sample.timestamp,
            "path": sample.filepath,
            "scene_index": sample.scene_index,
            "uniqueness": sample.uniqueness
        })
    return selected_frames

def align_frames_to_transcript(frames, transcript_segments):
    # (Keep your existing align_frames_to_transcript function)
    aligned_frames = []
    for frame in frames:
        frame_time = frame['timestamp']
        # Find the closest transcript segment
        closest_segment = min(transcript_segments, key=lambda seg: abs(seg['start'] - frame_time))

        aligned_item = {
            'text': closest_segment['text'],
            'path': frame['path'],
            'timestamp': frame['timestamp'],
            'uniqueness': frame['uniqueness'],
            'scene_index': frame['scene_index'],
        }
        aligned_frames.append(aligned_item)
    logger.info("Alignment of frames with transcript complete")
    return aligned_frames