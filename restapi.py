import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"

# Initialize model as None
model = None

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Check if the model is initialized
        if model is not None:
            results = model(img, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient="records")
        else:
            return "Model not initialized"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--weights', default='best.pt', help='path to model weights file, i.e., --weights best.pt')
    args = parser.parse_args()

    # Load the YOLOv5 model from the specified weights file
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=True)
    model.eval()

    app.run(host="0.0.0.0", port=args.port)

