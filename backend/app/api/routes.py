# routes.py
import os
import subprocess
import whisper
import asyncio
import logging
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from app.api import frame_extraction, agentic
import json
import fiftyone as fo


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter()
load_dotenv()

class VideoRequest(BaseModel):
    videoUrl: str


def save_transcript(transcript: str, video_id: str) -> str:
    transcript_folder = "transcripts"
    os.makedirs(transcript_folder, exist_ok=True)
    transcript_file = os.path.join(transcript_folder, f"{video_id}.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    logger.info(f"Transcript saved to {transcript_file}")
    return transcript_file


@router.post("/generate-course")
async def generate_course(request: VideoRequest):
    try:
        video_id = request.videoUrl.split("v=")[-1].split("&")[0]
        logger.info(f"Extracted video_id: {video_id}")
    except Exception as e:
        logger.error(f"Error extracting video_id: {e}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    def get_transcript():
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            segments = [
                {"start": item["start"], "duration": item["duration"], "text": item["text"]}
                for item in transcript_data
            ]
            transcript_formatted = "\n".join(
                [f"{seg['start']:.2f}s: {seg['text']}" for seg in segments]
            )
            logger.info(
                "Transcript extracted via YouTubeTranscriptApi (first 200 chars): %s",
                transcript_formatted[:200],
            )
            return transcript_formatted, segments
        except Exception as e:
            logger.info(f"YouTubeTranscriptApi failed: {e}")
            audio_file = f"/tmp/{video_id}.mp3"
            command = (
                f"yt-dlp -x --audio-format mp3 -o '/tmp/%(id)s.%(ext)s' {request.videoUrl}"
            )
            subprocess.run(command, shell=True, check=True)
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            segments = result.get("segments", [])
            transcript_formatted = "\n".join(
                [f"{seg['start']:.2f}-{seg['end']:.2f}s: {seg['text']}" for seg in segments]
            )
            logger.info(
                "Transcript extracted via Whisper (first 200 chars): %s",
                transcript_formatted[:200],
            )
            os.remove(audio_file)
            segments_formatted = [
                {"start": seg['start'], "duration": seg['end'] - seg['start'], "text": seg['text']}
                for seg in segments
            ]
            return transcript_formatted, segments_formatted

    def get_frames():
        vid_id, video_path = frame_extraction.download_video(request.videoUrl)
        scenes = frame_extraction.detect_scenes(video_path, threshold=30.0)
        frames = frame_extraction.extract_keyframes_with_timestamps(video_path, scenes, output_dir="frames")
        dataset = frame_extraction.load_frames_into_fiftyone(frames, video_id=vid_id)
        selected_frames = frame_extraction.select_unique_frames(dataset, uniqueness_threshold=0.7)

        for frame in selected_frames:
            frame['path'] = os.path.join("/frames", os.path.basename(frame["path"])).replace("\\","/")
        return selected_frames


    try:
        transcript_future = asyncio.to_thread(get_transcript)
        frames_future = asyncio.to_thread(get_frames)

        transcript_formatted, transcript_segments = await transcript_future
        frames = await frames_future  

        logger.info("Transcript and frame extraction completed.")

        aligned_frames = frame_extraction.align_frames_to_transcript(frames, transcript_segments)
        logger.info("Frame and transcript alignment done.")

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise HTTPException(status_code=500, detail="Extraction error.")

    transcript_file = save_transcript(transcript_formatted, video_id)

    try:
        logger.info("Running LangGraph agent.")
        inputs = {"transcript": transcript_formatted, "frames": aligned_frames}
        results = agentic.app.invoke(inputs, {"configurable": {"thread_id": video_id}})
        logger.info("LangGraph agent completed successfully.")

        # --- Debugging: Print the raw results ---
        print("Raw results from agentic.app.invoke:", results)

        # --- Access state keys DIRECTLY from results ---
        structured_content_dict = results.get("structured_content")
        course_content_dict = results.get("course_content")
        quiz_content_dict = results.get("quiz_content")
        retention_plan_dict = results.get("retention_plan")


        logger.info(f"structured_content_str: {structured_content_dict}")
        logger.info(f"course_content_str: {course_content_dict}")
        logger.info(f"quiz_content_str: {quiz_content_dict}")
        logger.info(f"retention_plan_str: {retention_plan_dict}")

    except Exception as e:
        logger.error(f"Error during agent processing: {e}")
        raise HTTPException(status_code=500, detail=f"Agent processing error: {e}")

    return {
        "course": {
            "structured_content": structured_content_dict,
            "course_content": course_content_dict,
            "quiz_content": quiz_content_dict,
            "retention_plan": retention_plan_dict,
        }
    }