"""
Entry point for BaseAL
Serves both the FastAPI backend and React frontend
"""
import os
import threading
import uvicorn
from pathlib import Path
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
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

# ---------------------------------------------------------------------------
# Startup readiness — show a maintenance page until umap JIT is compiled
# ---------------------------------------------------------------------------

_app_ready = False

_MAINTENANCE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="15">
  <title>Starting up…</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0f1923;
      color: white;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
    }
    h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.6rem; }
    p  { color: rgba(255,255,255,0.55); font-size: 0.875rem; line-height: 1.7; }
  </style>
</head>
<body>
  <div style="text-align:center">
    <h1>Starting up</h1>
    <p>The application is warming up and will be ready shortly.<br>
       This page refreshes automatically.</p>
  </div>
</body>
</html>"""


@app.middleware("http")
async def startup_check(request: Request, call_next):
    if not _app_ready:
        return HTMLResponse(
            content=_MAINTENANCE_HTML,
            status_code=503,
            headers={"Retry-After": "15"},
        )
    return await call_next(request)


def _warmup():
    """Trigger umap/numba JIT compilation so the first real request is fast."""
    global _app_ready
    try:
        import umap
        import numpy as np
        umap.UMAP(n_components=2, n_neighbors=3).fit_transform(np.random.rand(20, 5))
    except Exception:
        pass
    _app_ready = True


@app.on_event("startup")
async def on_startup():
    threading.Thread(target=_warmup, daemon=True).start()


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

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

    _no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    # Serve frontend at root
    @app.get("/")
    async def serve_index():
        """Serve the React SPA index"""
        return FileResponse(FRONTEND_BUILD_DIR / "index.html", headers=_no_cache_headers)

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
        return FileResponse(FRONTEND_BUILD_DIR / "index.html", headers=_no_cache_headers)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting BaseAL on port {port}...")
    print(f"Frontend build dir: {FRONTEND_BUILD_DIR}")
    print(f"Frontend exists: {FRONTEND_BUILD_DIR.exists()}")
    uvicorn.run(app, host="0.0.0.0", port=port)
