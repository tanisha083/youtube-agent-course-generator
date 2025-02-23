"""
agentic.py: Agentic Workflow for Video-to-Course Conversion

Implements a LangGraph state graph to automate conversion of video content into structured courses.
Takes video transcripts and extracted frames as input and orchestrates an agentic workflow to:
- Structure content into modules and sections.
- Select relevant frames as visual aids.
- Generate detailed course content (lessons).
- Design quizzes for assessment.
- Develop retention-focused learning strategies.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, TypedDict
from uuid import uuid4
from dotenv import load_dotenv
import google.api_core.exceptions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from maxim import Maxim, Config
from maxim.logger import LoggerConfig
from maxim.logger.components.generation import GenerationConfig
from maxim.logger.components.span import SpanConfig
from maxim.logger.components.trace import TraceConfig
from maxim.types import GenerationRequestMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.api.utils import clean_json_string, encode_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("langgraph_agentic")
load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")
maxim_api_key = os.environ.get("MAXIM_API_KEY")
log_repository_id = os.environ.get("LOG_REPO_ID")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")
if not maxim_api_key:
    raise ValueError("MAXIM_API_KEY environment variable not set.")
if not log_repository_id:
    raise ValueError("LOG_REPO_ID environment variable not set.")

maxim_client = Maxim(Config(api_key=maxim_api_key))
maxim_logger = maxim_client.logger(LoggerConfig(id=log_repository_id))
trace = maxim_logger.trace(TraceConfig(id=str(uuid4()), name="user-course-generation"))

# pylint: disable=line-too-long

class GraphState(TypedDict):
    """State representation for the content generation graph."""
    transcript: str
    frames: List[Dict[str, str]]
    structured_content: Dict
    course_content: Dict
    quiz_content: Dict
    retention_plan: Dict
    trace_id: Optional[str]

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=gemini_api_key
)
logger.info("Initialized ChatGoogleGenerativeAI with model gemini-1.5-pro.")

groq_llm = ChatGroq(
    model="llama-3.2-90b-vision-preview",
    temperature=0.7,
    api_key=groq_api_key
)
logger.info("Initialized Groq with llama-3.2-90b-vision-preview.")

memory = MemorySaver()

def log_retry_error(retry_state):
    """Log retry error."""
    logger.error("Retrying: %s", retry_state)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    retry_error_callback=log_retry_error
)
def call_gemini_with_retry(chain, input_data, generation_config: GenerationConfig, span):
    """Call Gemini with retries and log the generation result."""
    generation = span.generation(generation_config)
    try:
        result = chain.invoke(input_data)
        content = result.content
        gemini_response = {
            "id": str(uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": generation_config.model,
            "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(str(input_data)),
                "completion_tokens": len(content),
                "total_tokens": len(str(input_data)) + len(content),
            },
        }
        generation.result(gemini_response)
        return result
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        generation.result({"error": str(e)})
        raise
    finally:
        generation.end()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error
)
def call_groq_with_retry(chain, input_data, generation_config: GenerationConfig, span):
    """Call Groq with retries and log the generation result."""
    generation = span.generation(generation_config)
    try:
        result = chain.invoke(input_data)
        content = result.content
        gemini_response = {
            "id": str(uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": generation_config.model,
            "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(str(input_data)),
                "completion_tokens": len(content),
                "total_tokens": len(str(input_data)) + len(content),
            },
        }
        generation.result(gemini_response)
        return result
    except Exception as e:
        logger.exception("Groq call failed: %s", e)
        generation.result({"error": str(e)})
        raise
    finally:
        generation.end()

def content_structurer(state: GraphState) -> GraphState:
    """Generate a blog-style course structure from the given transcript."""
    logger.info("Running Content Structurer Node")
    content_structurer_span = trace.span(SpanConfig(id=str(uuid4()), name="Content Structurer"))
    content_structurer_span.event("content_structurer_start", "Content Structurer Start")

    prompt_template = """
Analyze the following video transcript and create a detailed, blog-style course structure.  
The goal is to create a structure that is easy to understand and helps in selecting relevant images later.

1. **Identify Main Modules:** Divide the video into distinct modules. Each module should represent a major topic or theme.
   -  Use **highly descriptive and specific module titles**.  Avoid generic titles like "Module 1" or "Introduction."  
   -  Example:  Instead of "Introduction", use "Famotidine:  Understanding the Basics and Mechanism of Action".

2. **Structure within Modules (Sections):**  Within each module, identify logical sections.
   - Use **highly descriptive and specific section titles**.  Avoid generic titles like "Section 1" or "Overview."
   - Example: Instead of "Section 1", use "What is Famotidine and What Conditions Does it Treat?".
   - Provide accurate `start_ts` and `end_ts` (timestamps) for each section.

3. **Extract Global Concepts:** Identify 5-10 key overarching concepts that are central to the entire video. These should be concise and informative.

4. **Output Format:**  Output the structure as plain JSON, *without* any markdown formatting or code blocks. Follow the exact format below:

```json
{{
    "modules": [
        {{
            "module_title": "Descriptive Module Title Here",
            "sections": [
                {{
                    "section_title": "Descriptive Section Title Here",
                    "start_ts": 0.00,  
                    "end_ts": 10.50,
                    frames: [],//return it empty for now
                }},
                ... more sections ...
            ]
        }},
        ... more modules ...
    ],
    "global_concepts": ["Concept 1", "Concept 2", ...]
}}

Transcript:
{transcript}
"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_llm # pylint: disable=unsupported-binary-operation

    messages_content = prompt_template.format(transcript=state["transcript"])
    messages = [GenerationRequestMessage(role="user", content=messages_content)]
    generation_config = GenerationConfig(
        id=str(uuid4()),
        name="Content Structure Generation",
        provider="google",
        model="gemini-1.5-flash",
        model_parameters={"temperature": 0.7},
        messages=messages,
    )
    generation = content_structurer_span.generation(generation_config)

    try:
        response = chain.invoke({"transcript": state["transcript"]})
        logger.info("Content Structurer output: %s", response)
        cleaned_content = clean_json_string(response.content)
        structured_content_json = json.loads(cleaned_content)
        state["structured_content"] = structured_content_json

        gemini_response = {
            "id": str(uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": "gemini-1.5-flash",
            "choices": [{"index": 0, "text": cleaned_content, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(str(state["transcript"])),
                "completion_tokens": len(cleaned_content),
                "total_tokens": len(str(state["transcript"])) + len(cleaned_content),
            },
        }
        logger.info("gemini response: %s", gemini_response)
        generation.result(gemini_response)
        content_structurer_span.event("content_structurer_success", "Content Structurer Succeeded")
    except (json.JSONDecodeError, ValueError, google.api_core.exceptions.GoogleAPIError) as e:
        logger.exception("Content Structurer error: %s", e)
        content_structurer_span.event("content_structurer_error",
                                    "Error in Content Structurer",
                                    {"error": str(e)})
        generation.result({"error": str(e)})
        state["structured_content"] = {}
    finally:
        generation.end()
        content_structurer_span.end()
    return state

def frame_selector(state: GraphState) -> GraphState:
    """Select frames from the video based on relevancy."""
    logger.info("Running Frame Selector Node")
    frame_selector_span = trace.span(SpanConfig(id=str(uuid4()), name="Frame Selector"))
    frame_selector_span.event("frame_selector_start", "Frame Selector Start")

    frames = state["frames"]
    selected_frames_with_info = []

    for module_index, module in enumerate(state["structured_content"]["modules"]):
        module_start_ts = int(float(module["sections"][0]["start_ts"]))
        module_end_ts = int(float(module["sections"][-1]["end_ts"]))
        module_content = state["transcript"][module_start_ts:module_end_ts]

        for section_index, section in enumerate(module["sections"]):
            start_ts = float(section["start_ts"])
            end_ts = float(section["end_ts"])
            section_content = state["transcript"][int(start_ts):int(end_ts)]
            section_frames = []

            for frame in frames:
                frame_ts = float(frame["timestamp"])
                if start_ts <= frame_ts <= end_ts:
                    section_frames.append(frame) # Add frame to section if timestamp is within range

            if "frames" not in section:
                section["frames"] = []

            if section_frames: # Process section_frames only if it's not empty
                logger.info("Processing %d frames for section: '%s'",
                            len(section_frames),
                            section['section_title'])
                for frame_index, frame in enumerate(section_frames): #Iterate through section_frames
                    logger.info("Processing frame: %s (Index within section: %d) for section: '%s'",
                                frame['path'], frame_index, section['section_title'])

                    image_path = os.path.join(os.getcwd(), frame["path"].lstrip('/'))
                    base64_image = encode_image(image_path)
                    if not base64_image:
                        logger.warning("Skipping frame %s due to encoding failure.", frame["path"])
                        continue

                    image_data_item = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }

                    user_prompt_text = (
                        "**You MUST respond *only* with valid JSON. Do not include any introductory text, explanations, \
                        or markdown formatting. Just the JSON object.**\n\n"
                        f"Module content:\n{module_content}\n\n"
                        f"Section content:\n{section_content}\n\n"
                        "Given this module and section context, and the provided image, determine if the image is relevant "
                        "to the *section*. "
                        "If the image IS relevant, ALSO generate a short caption (1-2 sentences) for it. "
                        "Return a JSON object in one of these two formats:\n"
                        "- If the image is relevant: `{\"relevant\": true, \"info\": \"Your 1-2 sentence reason\", \
                        \"caption\": \"Generated caption\"}`\n"
                        "- If the image is NOT relevant: `{\"relevant\": false}`\n\n"
                        "Example of a relevant response: `{\"relevant\": true, \"info\": \"This image shows the device setup.\", \
                        \"caption\": \"The experimental setup with the sensor attached.\"}`\n"
                        "Example of a not relevant response: `{\"relevant\": false}`\n\n"
                        "**Remember, ONLY output the JSON.  No other text. you are strictly told to only \
                            return expected json else u will be heavily penalized**"
                    )

                    message_content = [
                        {"type": "text", "text": user_prompt_text},
                        image_data_item,
                    ]
                    messages = [GenerationRequestMessage(role="user", content=message_content)]
                    generation_config = GenerationConfig(
                        id=str(uuid4()),
                        name=f"Frame Relevance - Mod {module_index}, Sec {section_index}, Frame {frame_index}",
                        provider="groq",
                        model="llama-3.2-90b-vision-preview",
                        model_parameters={"temperature": 0.7},
                        messages=messages,
                    )
                    message_to_llama = [HumanMessage(content=message_content)]

                    try:
                        response = call_groq_with_retry(groq_llm,
                                                        message_to_llama,
                                                        generation_config,
                                                        frame_selector_span)
                        cleaned_response = clean_json_string(response.content)
                        logger.info("Frame processing response: %s", cleaned_response)
                        result = json.loads(cleaned_response)
                        if isinstance(result, dict) and result.get("relevant") is True:
                            if "info" in result and "caption" in result:
                                section["frames"].append({
                                    "frame_path": frame["path"],
                                    "caption": result["caption"],
                                    "info": result["info"],
                                    "timestamp": float(frame["timestamp"]),
                                })
                                selected_frames_with_info.append(frame)
                                logger.info("Timestamp %s appended to section: %s",
                                            frame["timestamp"], section["section_title"])
                        frame_selector_span.event(
                                    "frame_selection_success",
                                    "Successfully generated frame relevance result"
                                )
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.exception("Frame processing error: %s", e)
                        frame_selector_span.event("frame_selection_error",
                                                  "Error in frame selection",
                                                  {"error": str(e)})
            else:
                frame_selector_span.event("no_frames_in_range",
                                          f"No frames in range for section: {section['section_title']}")
    frame_selector_span.end()
    state["frames"] = selected_frames_with_info
    return state

def course_content_generator(state: GraphState) -> GraphState:
    """Generate course content based on structured content and selected frames."""
    logger.info("Running Course Content Generator Node")
    course_content_span = trace.span(SpanConfig(id=str(uuid4()), name="Course Content Generator"))
    course_content_span.event("course_content_generation_start", "Course Content Generation Start")

    structured_content_with_frames = state["structured_content"].copy()
    for module_index, module in enumerate(structured_content_with_frames["modules"]):
        for section_index, section in enumerate(module["sections"]):
            if "frames" in section:
                structured_content_with_frames["modules"][module_index]["sections"][section_index]["media"] = section["frames"]
            else:
                structured_content_with_frames["modules"][module_index]["sections"][section_index]["media"] = []



    system_message = """
You are an Educational Content Designer. Create blog-style course content based on the provided structure, transcript, and selected frames (if any).

Output JSON in the specified format.  Do NOT include markdown code blocks.
"""
    prompt_template = """
    {structured_content_with_frames}
    Transcript: {transcript}

    For each section:
    1.  Write a 300-500 word explanation of the concept.
    2.  If frames are provided for the section, include a "media" field with the frame_path and caption *exactly* as provided.
    3. Include a "Key Insight" callout.
    4. Include timestamps (only where required) in the generated section content (format: [HH:MM:SS]).

    Output JSON:
    {{        
        "modules": [
            {{
                "module_title": "...",
                "sections": [
                    {{
                        "section_title": "...",
                        "content": "Markdown formatted content...",
                        "media": [{{ "frame_path": "...", "caption": "...", "info": "...", "timestamp": x.xx }}],
                    }},
                    ...
                ]
            }},
            ...
        ]
    }}
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", prompt_template)
    ])
    chain = prompt | gemini_llm # pylint: disable=unsupported-binary-operation
    messages = [
        GenerationRequestMessage(role="system", content=system_message),
        GenerationRequestMessage(role="user", content=prompt_template.format(
            structured_content_with_frames=json.dumps(structured_content_with_frames),
            transcript=state["transcript"]
        )),
    ]
    generation_config = GenerationConfig(
        id=str(uuid4()),
        name="Course Content Generation",
        provider="google",
        model="gemini-1.5-flash",
        model_parameters={"temperature": 0.7},
        messages=messages,
    )

    try:
        course_content_response = call_gemini_with_retry(
            chain,
            {
                "structured_content_with_frames": json.dumps(structured_content_with_frames),
                "transcript": state["transcript"],
            },
            generation_config,
            course_content_span
        )
        cleaned_content = clean_json_string(course_content_response.content)
        logger.info("Course Content Generation output: %s", cleaned_content)
        state["course_content"] = json.loads(cleaned_content)
        course_content_span.event("course_content_generation_success", "Course Content Generation Succeeded")
    except (json.JSONDecodeError, ValueError, google.api_core.exceptions.GoogleAPIError) as e:
        logger.exception("Course Content Generation error: %s", e)
        course_content_span.event("course_content_error", "Error in course content generation", {"error": str(e)})
        state["course_content"] = {"modules": []}
    finally:
        course_content_span.end()
    return state

def quiz_architect(state: GraphState) -> GraphState:
    """Generate quiz questions based on the structured content."""
    logger.info("Running Quiz Architect Node")
    quiz_architect_span = trace.span(SpanConfig(id=str(uuid4()), name="Quiz Architect"))
    quiz_architect_span.event("quiz_architect_start", "Quiz Architect Start")

    prompt_template = """
    You are an Assessment Designer creating quizzes for a blog-style course. Design multiple-choice questions (MCQs) for each section.

    Quiz Requirements:

    Placement: 1-2 MCQs per section.
    Question Details:
        Difficulty Rating (1-5)
        Rationale: Explanation of correct answer and distractors.
        Review Timestamp: Reference a timestamp in the video.

    Output Format: Plain JSON. Do NOT include markdown code blocks.

    {{
      "quizzes": [
        {{
            "module_title": "...",
            "section_title": "...",
            "questions": [
                {{
                    "type": "multiple_choice",
                    "text": "...",
                    "options": [...],
                    "correct_answer": "...",
                    "rationale": "...",
                    "review_timestamp": "...",
                    "difficulty": X
                }},
                ...
            ]
        }},
        ...
      ]
    }}

    Structured Content:
    {structured_content}
    """

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_llm # pylint: disable=unsupported-binary-operation
    messages = [GenerationRequestMessage(role="user", content=prompt_template.format(
        structured_content=state["structured_content"]
    ))]
    generation_config = GenerationConfig(
        id=str(uuid4()),
        name="Quiz Generation",
        provider="google",
        model="gemini-1.5-flash",
        model_parameters={"temperature": 0.7},
        messages=messages,
    )
    try:
        quiz_content = call_gemini_with_retry(
            chain,
            {"structured_content": state["structured_content"]},
            generation_config,
            quiz_architect_span
        )
        cleaned_content = clean_json_string(quiz_content.content)
        logger.info("Quiz Architect output: %s", cleaned_content)
        state["quiz_content"] = json.loads(cleaned_content)
        quiz_architect_span.event("quiz_architect_success", "Quiz Architect Succeeded")
    except (json.JSONDecodeError, ValueError, google.api_core.exceptions.GoogleAPIError) as e:
        logger.exception("Quiz Architect error: %s", e)
        quiz_architect_span.event("quiz_architect_error", "Error in Quiz Architect", {"error": str(e)})
        state["quiz_content"] = {"quizzes": []}
    finally:
        quiz_architect_span.end()
    return state

def retention_designer(state: GraphState) -> GraphState:
    """Design a retention plan based on structured content."""
    logger.info("Running Retention Designer Node")
    retention_designer_span = trace.span(SpanConfig(id=str(uuid4()), name="Retention Designer"))
    retention_designer_span.event("retention_designer_start", "Retention Designer Start")

    prompt_template = """
You are a Learning Experience Designer focused on enhancing retention. Design a retention plan, focusing on text-based strategies, for a course based on the provided structured content.

Retention Features (at the MODULE level):

Retention Tips: 2-3 per MODULE. Use the SPECIFIC JSON format below. Choose tip types appropriate for the content.
Spaced Repetition Prompts: 1-2 per MODULE. Short questions/prompts to review material.
Scenario Examples: 1-2 per MODULE. Short scenarios applying the concepts.
Summary: A text-based summary, with comparison tables if relevant (at the end of the entire course).

Output Format: Plain JSON. Do NOT include markdown code blocks.

{{
    "retention_plan": {{
        "module_retention": [
            {{
                "module_title": "...",
                "retention_tips": [
                    {{
                        "type": "analogy",  // MUST be one of: analogy, real_world_example, table_creation, mnemonic_device, categorization, prioritization, role_playing, example, explanation, summary, comparison, question_generation
                        "description": "..." // Description of the tip, related to the MODULE content.
                    }},
                    ... // More tips
                ],
                "spaced_repetition_prompts": [...],
                "scenario_examples": [...]
            }},
            ...
        ],
        "overall_summary": "..."
    }}
}}

Structured Content:
{structured_content}
"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_llm # pylint: disable=unsupported-binary-operation
    messages = [GenerationRequestMessage(role="user", content=prompt_template.format(
        structured_content=state["structured_content"]
    ))]
    generation_config = GenerationConfig(
        id=str(uuid4()),
        name="Retention Plan Generation",
        provider="google",
        model="gemini-1.5-flash",
        model_parameters={"temperature": 0.7},
        messages=messages,
    )
    try:
        retention_plan = call_gemini_with_retry(
            chain,
            {"structured_content": state["structured_content"]},
            generation_config,
            retention_designer_span
        )
        cleaned_content = clean_json_string(retention_plan.content)
        logger.info("Retention Designer output: %s", cleaned_content)
        state["retention_plan"] = json.loads(cleaned_content)
        retention_designer_span.event("retention_design_success", "Retention Designer Succeeded")
    except (json.JSONDecodeError, ValueError, google.api_core.exceptions.GoogleAPIError) as e:
        logger.exception("Retention Designer error: %s", e)
        retention_designer_span.event("retention_design_error", "Error in retention design", {"error": str(e)})
        state["retention_plan"] = {"retention_plan": {"module_retention": [], "overall_summary": ""}}
    finally:
        retention_designer_span.end()
    return state

def retention_tip_executor(state: GraphState) -> GraphState:
    """Execute retention tips by generating additional content based on tip type."""
    logger.info("Running Retention Tip Executor Node")
    retention_tip_executor_span = trace.span(SpanConfig(id=str(uuid4()), name="Retention Tip Executor"))
    retention_tip_executor_span.event("retention_tip_executor_start", "Retention Tip Executor Start")

    retention_plan_data = state["retention_plan"]
    if not retention_plan_data or "retention_plan" not in retention_plan_data:
        logger.warning("No retention plan. Skipping Retention Tip Executor.")
        retention_tip_executor_span.event("retention_plan_missing", "No Retention Plan Found")
        retention_tip_executor_span.end()
        return state

    modules_retention = retention_plan_data["retention_plan"]["module_retention"]

    for module_index, module_retention in enumerate(modules_retention):
        for tip_index, retention_tip_dict in enumerate(module_retention.get("retention_tips", [])):
            instruction = None
            executed_content = None
            prompt = None
            tip_type = retention_tip_dict.get("type", "Unknown")
            tip_description = retention_tip_dict.get("description", "")

            if tip_type == "table_creation":
                instruction = tip_description
                prompt_template = "Create a markdown table: {instruction}. Respond with only the markdown table."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "real_world_example":
                instruction = tip_description
                prompt_template = "Provide a real-world example: {instruction}. Respond in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "analogy":
                instruction = tip_description
                prompt_template = "Expand on the following analogy: {instruction}. Explain in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "mnemonic_device":
                instruction = tip_description
                prompt_template = "Explain this mnemonic and how it helps: {instruction}. Explain in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "categorization":
                instruction = tip_description
                prompt_template = "Provide an example of categorization: {instruction}. Give a short example in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "prioritization":
                instruction = tip_description
                prompt_template = "Explain how to prioritize based on: {instruction}. Explain in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "role_playing":
                instruction = tip_description
                prompt_template = "Suggest a role-playing scenario: {instruction}. Describe a brief scenario (2-4 sentences)."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "example":
                instruction = tip_description
                prompt_template = "Provide an example: {instruction}. Respond in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "explanation":
                instruction = tip_description
                prompt_template = "Provide an explanation: {instruction}. Respond in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "summary":
                instruction = tip_description
                prompt_template = "Provide a summary: {instruction}. Respond in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "comparison":
                instruction = tip_description
                prompt_template = "Provide a comparison: {instruction}. Respond in 2-3 sentences."
                prompt = PromptTemplate.from_template(prompt_template)
            elif tip_type == "question_generation":
                instruction = tip_description
                prompt_template = "Generate 1-2 questions related to: {instruction}. Provide only the questions, not the answers."
                prompt = PromptTemplate.from_template(prompt_template)

            if prompt:
                chain = prompt | gemini_llm # pylint: disable=unsupported-binary-operation
                messages = [GenerationRequestMessage(role="user",
                                                     content=prompt_template.format(instruction=instruction))]
                generation_config = GenerationConfig(
                    id=str(uuid4()),
                    name=f"Retention Tip - {tip_type} - Mod {module_index} - Tip {tip_index}",
                    provider="google",
                    model="gemini-1.5-flash",
                    model_parameters={"temperature": 0.7},
                    messages=messages,
                )
                try:
                    response = call_gemini_with_retry(
                        chain,
                        {"instruction": instruction},
                        generation_config,
                        retention_tip_executor_span
                    )
                    executed_content = response.content
                    retention_tip_dict["executed_content"] = executed_content
                    logger.info("Executed retention tip '%s' for module '%s'.",
                                 tip_type,
                                 module_retention["module_title"])
                    retention_tip_executor_span.event("retention_tip_execution_success", "Retention Tip Execution Succeeded")
                except (google.api_core.exceptions.GoogleAPIError,
                        json.JSONDecodeError, ValueError) as e:
                    logger.exception("Retention tip execution error (%s): %s", tip_type, e)
                    retention_tip_dict["executed_content"] = "Error"
                    retention_tip_executor_span.event(
                        f"retention_tip_error_{tip_type}",
                        "Retention Tip Error",
                        {"error": str(e)}
                    )
            else:
                retention_tip_executor_span.event(f"unknown_tip_type_{tip_type}",
                                                  "Unknown Tip Type")
                retention_tip_dict["executed_content"] = "Unknown"
    retention_tip_executor_span.end()
    state["retention_plan"] = retention_plan_data
    return state

graph = StateGraph(GraphState)
graph.add_node("content_structurer", content_structurer)
graph.add_node("frame_selector", frame_selector)
graph.add_node("course_content_generator", course_content_generator)
graph.add_node("quiz_architect", quiz_architect)
graph.add_node("retention_designer", retention_designer)
graph.add_node("retention_tip_executor", retention_tip_executor)
graph.add_edge("content_structurer", "frame_selector")
graph.add_edge("frame_selector", "course_content_generator")
graph.add_edge("course_content_generator", "quiz_architect")
graph.add_edge("quiz_architect", "retention_designer")
graph.add_edge("retention_designer", "retention_tip_executor")
graph.add_edge("retention_tip_executor", END)
graph.set_entry_point("content_structurer")

app = graph.compile(checkpointer=memory)
