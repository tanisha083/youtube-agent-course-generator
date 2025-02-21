#main.py
from fastapi import FastAPI, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# --- Serve Static Files ---
# Create the 'frames' directory if it doesn't exist
frames_dir = os.path.join(os.getcwd(), "frames")  # Correct path
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Mount the 'frames' directory to be served at '/frames'
app.mount("/frames", staticfiles.StaticFiles(directory=frames_dir), name="frames")
# --- End Serve Static Files ---

app.include_router(router, prefix="/api", tags=["YouTube Agent"])

@app.get("/")
async def root():
    return {"message": "Welcome to YouTube Agent!"}
