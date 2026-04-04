import uvicorn
from app.main import app

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    # Standard OpenEnv port is often expected to be 7860 or as defined in the environment
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
