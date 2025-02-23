"""
Includes helper functions for downloading videos, detecting scenes, extracting and optimizing 
keyframes, loading frames into a FiftyOne dataset, and aligning frames to transcript segments.
"""

import os
import subprocess
import logging
from typing import List, Dict, Tuple, Any

from yt_dlp import YoutubeDL
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
import fiftyone as fo
import fiftyone.brain as fob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("frame_extraction")


def download_video(url: str) -> Tuple[str, str]:
    """
    Download a video from the given URL using yt-dlp.

    Args:
        url (str): The URL of the video to download.

    Returns:
        Tuple[str, str]: A tuple containing video ID and video path.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': '%(id)s.%(ext)s',
        'merge_output_format': 'mp4',
    }
    with YoutubeDL(ydl_opts) as ydl:
        logger.info("Downloading video from URL: %s", url)
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        video_ext = info.get('ext', 'mp4')
        video_path = f"{video_id}.{video_ext}"
        logger.info("Downloaded video %s", video_id)
        return video_id, video_path


def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Tuple[Any, Any]]:
    """
    Detect scenes in a video file using scenedetect.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold for scene detection.

    Returns:
        List[Tuple[Any, Any]]: List of detected scenes.
    """
    logger.info("Detecting scenes in %s", video_path)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor(1)
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    logger.info("Scene detection complete: %s scenes found", len(scene_list))
    return scene_list


def extract_keyframes_with_timestamps(video_path: str,
                                      scenes: List[Any],
                                      output_dir: str = "frames") -> List[Dict[str, Any]]:
    """
    Extract keyframes from video based on scene detection and save them to disk.
    Optimizes extracted frames and returns metadata for each frame.

    Args:
        video_path (str): Path to the video file.
        scenes (List[Any]): List of detected scenes.
        output_dir (str): Directory to store extracted frames.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing frame metadata.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created directory %s", output_dir)

    frames: List[Dict[str, Any]] = []
    num_scenes = len(scenes)

    # --- Enhanced Scene Logging ---
    logger.info("Number of scenes detected: %s", num_scenes)
    for idx, scene in enumerate(scenes):
        start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
        logger.info("Scene %04d: Start Time: %.3f, End Time: %.3f", idx, start_time, end_time)

    if scenes:
        video_duration_seconds = scenes[-1][1].get_seconds()
        logger.info("Estimated video duration: %.3f seconds", video_duration_seconds)
    else:
        video_duration_seconds = 600  # Default, or fetch from yt-dlp info later if needed.
        logger.warning("No scenes detected, using default video duration.")

    for idx, scene in enumerate(scenes):
        start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()

        # Offsets: start, middle. Skip end time for the *last* scene.
        offsets = [0, (end_time - start_time) / 2]
        if idx < num_scenes - 1:  # Add end time offset unless it's the last scene
            offsets.append(end_time - start_time)

        for t_offset in offsets:
            t_actual = start_time + t_offset
            frame_timestamp = max(0, min(t_actual, video_duration_seconds))
            frame_path = os.path.join(output_dir, f"frame_{idx:04d}_{frame_timestamp:.2f}.png")
            command = f'ffmpeg -y -loglevel error -ss {frame_timestamp:.2f} -i "{video_path}" -frames:v 1 "{frame_path}"'  # pylint: disable=line-too-long
            logger.info("FFMPEG command: %s", command)
            subprocess.run(command, shell=True, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            frames.append({"timestamp": frame_timestamp, "path": frame_path, "scene_index": idx})
            optimize_frame(frame_path)

    logger.info("Extracted %s keyframes", len(frames))
    return frames


def optimize_frame(frame_path: str,
                   resolution: Tuple[int, int] = (1280, 720),
                   quality: int = 85) -> None:
    """
    Optimize a frame image by resizing it to the specified resolution and saving with given quality.

    Args:
        frame_path (str): Path to the image file.
        resolution (Tuple[int, int]): Target resolution as (width, height).
        quality (int): Quality level for saved image.
    """
    logger.info("optimize_frame called for path: %s", frame_path)
    try:
        logger.info("optimize_frame attempting to open: %s", frame_path)
        img = Image.open(frame_path)
        logger.info("optimize_frame successfully opened: %s", frame_path)
        img = img.resize(resolution)
        img.save(frame_path, quality=quality)
    except (IOError, OSError, Image.UnidentifiedImageError) as e:
        logger.error("Optimization failed for %s: %s", frame_path, e)
        logger.error("optimize_frame exception details: Path: %s, Error: %s", frame_path, e)


def load_frames_into_fiftyone(frames: List[Dict[str, Any]], video_id: str) -> fo.Dataset:
    """
    Load frame metadata into a FiftyOne dataset.

    Args:
        frames (List[Dict[str, Any]]): List of frame metadata dictionaries.
        video_id (str): Identifier for the video.

    Returns:
        fo.Dataset: The created FiftyOne dataset.
    """
    dataset_name = f"video-frames-{video_id}"

    # --- Delete dataset if it exists ---
    if dataset_name in fo.list_datasets():
        logger.info("Deleting existing FiftyOne dataset: %s", dataset_name)
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    for frame_info in frames:
        sample = fo.Sample(filepath=frame_info["path"])
        sample["timestamp"] = frame_info["timestamp"]
        sample["scene_index"] = frame_info["scene_index"]
        dataset.add_sample(sample)
    return dataset


def select_unique_frames(dataset: fo.Dataset,
                         uniqueness_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Select unique frames from a FiftyOne dataset based on a uniqueness threshold.

    Args:
        dataset (fo.Dataset): FiftyOne dataset containing frame samples.
        uniqueness_threshold (float): Threshold for uniqueness.

    Returns:
        List[Dict[str, Any]]: List of unique frame metadata dictionaries.
    """
    fob.compute_uniqueness(dataset)
    unique_view = dataset.match(fo.ViewField("uniqueness") > uniqueness_threshold)

    # Sort by scene index, then by timestamp within each scene.
    sorted_view = unique_view.sort_by("scene_index").sort_by("timestamp")

    selected_frames: List[Dict[str, Any]] = []
    for sample in sorted_view:
        selected_frames.append({
            "timestamp": sample.timestamp,
            "path": sample.filepath,
            "scene_index": sample.scene_index,
            "uniqueness": sample.uniqueness
        })
    return selected_frames


def align_frames_to_transcript(frames: List[Dict[str, Any]],
                               transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Align frames to transcript segments by matching frame timestamps to the closest transcript 
    segment.

    Args:
        frames (List[Dict[str, Any]]): List of frame metadata dictionaries.
        transcript_segments (List[Dict[str, Any]]): List of transcript segments with 
        'start' & 'text' keys.

    Returns:
        List[Dict[str, Any]]: List of aligned frame dictionaries including transcript text.
    """
    aligned_frames: List[Dict[str, Any]] = []
    for frame in frames:
        frame_time = frame['timestamp']
        # Find the closest transcript segment, capturing frame_time to avoid cell variable issues.
        closest_segment = min(transcript_segments,
                              key=lambda seg,
                              t=frame_time: abs(seg['start'] - t))
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
