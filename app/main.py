"""Flask app entry (medical RAG transferred from notebook)."""
import sys
from pathlib import Path

# Ensure project root is on path (run from app/ or GEN-AI/)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Flask, render_template

from app.api.routes import api


def create_app():
    app = Flask(__name__)
    app.register_blueprint(api)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
