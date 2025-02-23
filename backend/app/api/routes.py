"""
Module for generating a course from a YouTube video by extracting transcripts
and keyframes, then processing them with LangGraph agent.
"""

import os
import subprocess
import asyncio
import logging
from typing import Any, Dict, List, Tuple
import whisper
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from dotenv import load_dotenv
from app.api import frame_extraction, agentic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter()
load_dotenv()


class VideoRequest(BaseModel):
    """
    Request model for video URL.
    """
    videoUrl: str


def save_transcript(transcript: str, video_id: str) -> str:
    """
    Save the transcript text to a file.

    Args:
        transcript: The transcript text.
        video_id: YouTube video identifier.

    Returns:
        The path to the saved transcript file.
    """
    transcript_folder: str = "transcripts"
    os.makedirs(transcript_folder, exist_ok=True)
    transcript_file: str = os.path.join(transcript_folder, f"{video_id}.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    logger.info("Transcript saved to %s", transcript_file)
    return transcript_file


@router.post("/generate-course")
async def generate_course(request: VideoRequest) -> Dict[str, Any]:
    """
    Generate course content from a YouTube video by extracting the transcript,
    frames, and then invoking the LangGraph agent for course generation.

    Args:
        request: VideoRequest object containing the YouTube video URL.

    Returns:
        A dictionary with course content details.
    """
    try:
        video_id: str = request.videoUrl.split("v=")[-1].split("&")[0]
        logger.info("Extracted video_id: %s", video_id)
    except Exception as e:
        logger.error("Error extracting video_id: %s", e)
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.") from e

    def get_transcript() -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract transcript from YouTube or via Whisper as fallback.

        Returns:
            A tuple of formatted transcript text and its segments.
        """
        try:
            transcript_data: List[Dict[str, Any]] = YouTubeTranscriptApi.get_transcript(video_id)
            segments: List[Dict[str, Any]] = [
                {
                    "start": item["start"],
                    "duration": item["duration"],
                    "text": item["text"]
                }
                for item in transcript_data
            ]
            transcript_formatted: str = "\n".join(
                [f"{seg['start']:.2f}s: {seg['text']}" for seg in segments]
            )
            logger.info(
                "Transcript extracted via YouTubeTranscriptApi (first 200 chars): %s",
                transcript_formatted[:200],
            )
            return transcript_formatted, segments
        except NoTranscriptFound as e:
            logger.info("YouTubeTranscriptApi failed: %s", e)
            audio_file: str = f"/tmp/{video_id}.mp3"
            command: str = (
                f"yt-dlp -x --audio-format mp3 -o '/tmp/%(id)s.%(ext)s' "
                f"{request.videoUrl}"
            )
            subprocess.run(command, shell=True, check=True)
            model = whisper.load_model("base")
            result: Dict[str, Any] = model.transcribe(audio_file)
            segments_whisper: List[Dict[str, Any]] = result.get("segments", [])
            transcript_formatted = "\n".join(
                [f"{seg['start']:.2f}-{seg['end']:.2f}s: {seg['text']}" for seg in segments_whisper]
            )
            logger.info(
                "Transcript extracted via Whisper (first 200 chars): %s",
                transcript_formatted[:200],
            )
            os.remove(audio_file)
            segments_formatted: List[Dict[str, Any]] = [
                {
                    "start": seg["start"],
                    "duration": seg["end"] - seg["start"],
                    "text": seg["text"]
                }
                for seg in segments
            ]
            return transcript_formatted, segments_formatted

    def get_frames() -> List[Dict[str, Any]]:
        """
        Download video, extract scenes and keyframes, then load them into a FiftyOne dataset.

        Returns:
            A list of selected unique frames with their paths.
        """
        vid_id, video_path = frame_extraction.download_video(request.videoUrl)
        scenes: List[Any] = frame_extraction.detect_scenes(video_path, threshold=20.0)
        frames: List[Any] = frame_extraction.extract_keyframes_with_timestamps(
            video_path, scenes, output_dir="frames"
        )
        dataset = frame_extraction.load_frames_into_fiftyone(frames, video_id=vid_id)
        selected_frames: List[Dict[str, Any]] = frame_extraction.select_unique_frames(
            dataset, uniqueness_threshold=0.7
        )
        for frame in selected_frames:
            frame["path"] = os.path.join(
                "/frames", os.path.basename(frame["path"])
            ).replace("\\", "/")
        return selected_frames

    try:
        transcript_future: asyncio.Future = asyncio.to_thread(get_transcript)
        frames_future: asyncio.Future = asyncio.to_thread(get_frames)

        transcript_formatted, transcript_segments = await transcript_future
        frames: List[Dict[str, Any]] = await frames_future

        logger.info("Transcript and frame extraction completed.")

        aligned_frames: List[Any] = frame_extraction.align_frames_to_transcript(
            frames, transcript_segments
        )
        logger.info("Frame and transcript alignment done.")
    except Exception as e:
        logger.error("Error during extraction: %s", e)
        raise HTTPException(status_code=500, detail="Extraction error.") from e

    save_transcript(transcript_formatted, video_id)

    try:
        logger.info("Running LangGraph agent.")
        inputs: Dict[str, Any] = {"transcript": transcript_formatted, "frames": aligned_frames}
        results: Dict[str, Any] = agentic.app.invoke(
            inputs, {"configurable": {"thread_id": video_id}}
        )
        logger.info("LangGraph agent completed successfully.")
        print("Raw results from agentic.app.invoke:", results)

        structured_content_dict = results.get("structured_content")
        course_content_dict = results.get("course_content")
        quiz_content_dict = results.get("quiz_content")
        retention_plan_dict = results.get("retention_plan")

        logger.info("structured_content_str: %s", structured_content_dict)
        logger.info("course_content_str: %s", course_content_dict)
        logger.info("quiz_content_str: %s", quiz_content_dict)
        logger.info("retention_plan_str: %s", retention_plan_dict)
    except Exception as e:
        logger.error("Error during agent processing: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing error: {e}"
        ) from e
    return {
        "course": {
            "structured_content": structured_content_dict,
            "course_content": course_content_dict,
            "quiz_content": quiz_content_dict,
            "retention_plan": retention_plan_dict,
        }
    }
