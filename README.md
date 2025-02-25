# YouTube Video to Structured Course Generator

🚀 Transform YouTube videos into interactive educational courses with AI-powered content generation, quizzes, and retention strategies.

## Key Features

- **Automated Video Processing**: Download videos, extract transcripts, detect scenes
- **Smart Keyframe Extraction**: Optimized frame selection with FiftyOne uniqueness analysis
- **AI-Powered Content Generation**:
  - Blog-style course structuring
  - Visual-text alignment with Gemini/Groq models
  - Quiz generation with difficulty ratings
  - Retention-focused learning strategies
- **Full-Stack Architecture**: FastAPI backend + React frontend
- **Maxim Integration:** Integrates [Maxim](https://www.getmaxim.ai/) for logging and observability. Traces, spans, and generations are logged to a Maxim repository, providing detailed insights into the agentic workflow execution.

## Installation
### Clone repository
```
gh repo clone tanisha083/youtube-agent-course-generator
cd youtube-agent-course-generator
```
### Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate
```
### Install Dependencies:
Ensure you have pip installed, then run:
```
pip install -r requirements.txt
```
### Set Up Environment Variables:
Create a .env file in the project root with the following:
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
MAXIM_API_KEY=your_maxim_api_key
LOG_REPO_ID=your_log_repository_id
```
## Usage
### Run the backend server:
```
cd backend
uvicorn main:app --reload
```
This starts the server locally on port 8000.

### Run the frontend:
```
cd frontend
npm start
```
This starts the server locally on port 3000.

## Demo
https://drive.google.com/file/d/16gN4C70lhdyf4rVMzDzx9rqFxmA8i7x7/view

