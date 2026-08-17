import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

load_dotenv(Path(__file__).resolve().parent / ".env")

from backend.feedback import generate_asl_feedback
from backend.predictor import NoHandDetectedError, predict_image
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


app = Flask(__name__)


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


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
    filename = secure_filename(image.filename)

    if filename == "":
        return jsonify({"error": "Invalid filename"}), 400

    image_path = UPLOADS_DIR / filename
    image.save(image_path)

    attempted_letter = request.form.get("letter") or None

    try:
        # Run the trained ASL model
        result = predict_image(image_path)

        top3 = result.pop("top3")

        feedback = generate_asl_feedback(
            predicted_letter=result["prediction"],
            confidence=result["confidence"],
            top3=top3,
            attempted_letter=attempted_letter,
        )

        if feedback:
            result["feedback"] = feedback

        return jsonify(result)

    except NoHandDetectedError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Remove the uploaded image after prediction
        if image_path.exists():
            image_path.unlink()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
