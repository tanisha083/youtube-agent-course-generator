"""
This module provides helper functions to clean JSON strings extracted from markdown 
and to encode images to base64.
"""

import json
import logging
from typing import Optional
import base64
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("langgraph_agentic")


def clean_json_string(json_string_with_markdown: Optional[str]) -> str:
    """
    Clean a JSON string extracted from markdown text.

    Parameters:
        json_string_with_markdown (Optional[str]): The JSON string possibly embedded in markdown.

    Returns:
        str: Cleaned JSON string if valid, otherwise an empty string.
    """
    if json_string_with_markdown is None:
        return ""
    cleaned_string: str = json_string_with_markdown.strip()
    json_match: Optional[re.Match[str]] = re.search(r"\{.*\}", cleaned_string, re.DOTALL)
    if json_match:
        extracted_json: str = json_match.group(0)
        try:
            json.loads(extracted_json)  # Validate JSON
            logger.info("Extracted JSON using regex: %s", extracted_json)
            return extracted_json
        except json.JSONDecodeError:
            logger.warning(
                "Regex extracted potential JSON, but it's invalid: %s. Falling back to basic cleaning.", # pylint: disable=line-too-long
                extracted_json
            )

    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string[7:]
    elif cleaned_string.startswith("```"):
        cleaned_string = cleaned_string[3:]
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[:-3]
    cleaned_string = cleaned_string.strip()

    try:
        json.loads(cleaned_string)
        return cleaned_string
    except json.JSONDecodeError:
        logger.warning(
            "Basic cleaning failed to produce valid JSON from: '%s'. Returning empty string.",
            json_string_with_markdown
        )
        return ""


def encode_image(image_path: str) -> Optional[str]:
    """
    Encodes an image to base64.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        Optional[str]: The base64 encoded string if successful, otherwise None.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error("Image file not found: %s", image_path)
        return None
    except Exception as e: # pylint: disable=broad-except
        logger.error("Error encoding image: %s", e)
        return None




# from tenacity import retry, stop_after_attempt, wait_exponential
# @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=5, max=20))
# def google_image_search(query: str, api_key: str, cse_id: str) -> Optional[str]:
#     """
#     Performs a Google Custom Search for images.
#
#     Parameters:
#         query (str): The search query.
#         api_key (str): Google API key.
#         cse_id (str): Google Custom Search Engine ID.
#
#     Returns:
#         Optional[str]: The URL of the first image result if found, otherwise None.
#     """
#     url: str = "https://www.googleapis.com/customsearch/v1"
#     params: dict[str, Any] = {
#         "key": api_key,
#         "cx": cse_id,
#         "q": query,
#         "searchType": "image",
#         "num": 1,
#     }
#     try:
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         results = response.json()
#
#         if "items" in results and results["items"]:
#             return results["items"][0]["link"]
#         else:
#             logger.warning("No image results found for query: %s", query)
#             return None
#
#     except requests.exceptions.RequestException as e:
#         logger.error("Error during Google Custom Search API call: %s", e)
#         return None
#     except (KeyError, IndexError) as e:
#         logger.error("Error parsing Google Custom Search API response: %s", e)
#         return None
