from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from backend.predictor import predict_image


# --------------------------------------------------
# Project folders
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Flask app
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Serve the frontend
# --------------------------------------------------


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# --------------------------------------------------
# Prediction API
# --------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    # Check that an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    # Check that a file was actually selected
    if image.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Save uploaded image temporarily
    image_path = UPLOADS_DIR / image.filename
    image.save(image_path)

    try:
        # Run the trained ASL model
        result = predict_image(image_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Remove the uploaded image after prediction
        if image_path.exists():
            image_path.unlink()


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
