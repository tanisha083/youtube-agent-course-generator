"""
Module main.py

This module initializes the FastAPI application, configures CORS middleware,
mounts the static files directory, and includes API routes for the YouTube Agent.
"""

import os
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

app: FastAPI = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Serve Static Files
frames_dir: str = os.path.join(os.getcwd(), "frames")
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)
app.mount("/frames", StaticFiles(directory=frames_dir), name="frames")

app.include_router(router, prefix="/api", tags=["YouTube Agent"])

@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """
    Root endpoint.

    Returns:
        Dict[str, str]: A dictionary containing a welcome message.
    """
    return {"message": "Welcome to YouTube Agent!"}
