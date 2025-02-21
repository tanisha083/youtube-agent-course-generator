
import json
import logging
from typing import TypedDict, List, Dict, Optional, Any
import base64
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("langgraph_agentic")


def clean_json_string(json_string_with_markdown):
    if json_string_with_markdown is None:
        return ""
    cleaned_string = json_string_with_markdown.strip()
    json_match = re.search(r"\{.*\}", cleaned_string, re.DOTALL)
    if json_match:
        extracted_json = json_match.group(0)
        try:
            json.loads(extracted_json) # Try to parse to validate it's actually JSON
            logger.info(f"Extracted JSON using regex: {extracted_json}") # Log when regex extraction works
            return extracted_json # Return the extracted JSON if valid
        except json.JSONDecodeError:
            logger.warning(f"Regex extracted potential JSON, but it's invalid: {extracted_json}. Falling back to basic cleaning.")
            # Fallback to basic cleaning if regex extraction fails to parse

    # Basic cleaning (original logic - still helpful for responses that are *almost* JSON)
    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string[7:]
    elif cleaned_string.startswith("```"):
        cleaned_string = cleaned_string[3:]
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[:-3]
    cleaned_string = cleaned_string.strip()

    try:
        json.loads(cleaned_string) # Try to parse after basic cleaning
        return cleaned_string # Return if basic cleaning works
    except json.JSONDecodeError:
        logger.warning(f"Basic cleaning failed to produce valid JSON from: '{json_string_with_markdown}'. Returning empty string.")
        return ""


# --- Helper function to encode images to base64 ---
def encode_image(image_path: str) -> Optional[str]:
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None
    
# # --- Helper function for Google Custom Search API Calls ---
# @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=5, max=20))
# def google_image_search(query: str, api_key: str, cse_id: str) -> Optional[str]:
#     """Performs a Google Custom Search for images."""
#     url = "https://www.googleapis.com/customsearch/v1"
#     params = {
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

#         if "items" in results and results["items"]:
#             return results["items"][0]["link"]
#         else:
#             logger.warning(f"No image results found for query: {query}")
#             return None

#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error during Google Custom Search API call: {e}")
#         return None
#     except (KeyError, IndexError) as e:
#         logger.error(f"Error parsing Google Custom Search API response: {e}")
#         return None
