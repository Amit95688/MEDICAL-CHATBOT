"""Simple Flask app for Medical RAG."""
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from flask import Flask, render_template
from app.api.routes import api


def create_app():
    """Create Flask app."""
    app = Flask(__name__)
    app.register_blueprint(api)
    
    @app.route("/")
    def index():
        """Serve home page."""
        print("Serving index page")
        return render_template("index.html")
    
    return app


if __name__ == "__main__":
    app = create_app()
    print("Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
