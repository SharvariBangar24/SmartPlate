import os
from flask import Flask, jsonify, send_from_directory

from flask_cors import CORS

# Resolve the absolute path to the frontend directory (one level up from backend/)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)
CORS(app)


# ── Serve Frontend ──────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


# ── API Endpoints ───────────────────────────────────────────
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        "message": "Smart Plate backend is working!"
    })


if __name__ == '__main__':
    app.run(debug=True)
