"""
Hugging Face Spaces entry point for BaseAL
Serves both the FastAPI backend and React frontend
"""
import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

# Import the FastAPI app from api module
from api.main import app

# Update CORS for production (allow HF Spaces domain)
# Remove existing CORS middleware and add new one
app.middleware_stack = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.build_middleware_stack()

# Get paths
BASE_DIR = Path(__file__).parent
FRONTEND_BUILD_DIR = BASE_DIR / "app" / "dist"

# Serve static files from React build
if FRONTEND_BUILD_DIR.exists():
    # Mount static assets (js, css, etc.)
    app.mount("/assets", StaticFiles(directory=FRONTEND_BUILD_DIR / "assets"), name="assets")

    # Remove the existing root route from api/main.py and replace with frontend
    # Find and remove the existing "/" route
    app.routes[:] = [route for route in app.routes if not (hasattr(route, 'path') and route.path == "/")]

    # Serve frontend at root
    @app.get("/")
    async def serve_index():
        """Serve the React SPA index"""
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")

    # Catch-all route for SPA - must be after API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes"""
        # Skip API routes
        if full_path.startswith("api/"):
            return None
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Return index.html for SPA routing
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")

if __name__ == "__main__":
    print("Starting BaseAL on port 7860...")
    print(f"Frontend build dir: {FRONTEND_BUILD_DIR}")
    print(f"Frontend exists: {FRONTEND_BUILD_DIR.exists()}")
    uvicorn.run(app, host="0.0.0.0", port=7860)
