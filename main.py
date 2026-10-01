"""Entry point: start the VectorDB web server (equivalent of the C++ main()).

    python main.py          -> http://localhost:8080
"""

import uvicorn

from src import config
from src.api.server import create_app


def main() -> None:
    app = create_app()
    ollama = app.state.ollama
    online = ollama.is_available()

    print("=== VectorDB Engine (Python) ===")
    print(f"http://localhost:{config.PORT}")
    print(f"{len(app.state.db)} demo vectors | {config.DEMO_DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama:", "ONLINE" if online else "OFFLINE (install from ollama.com, then: ollama serve)")
    if online:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
