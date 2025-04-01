from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo previamente entrenado
model = load_model("brain_tumor_detector.h5")

# Preprocesar imagen
def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        return None

    c = max(cnts, key=cv.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_image

# Predicción del tumor
def predict_tumor(image):
    try:
        image = preprocess_image(image)
        if image is None:
            return "No tumor detected"
        
        image_resized = cv.resize(image, (240, 240)) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)
        
        res = model.predict(image_resized)
        return "Tumor Detected" if res[0][0] > 0.5 else "No Tumor"
    except Exception as e:
        return f"Error: {str(e)}"

# Ruta principal para cargar imágenes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            image = Image.open(filepath)
            image = np.array(image)
            
            result = predict_tumor(image)
            return render_template("index.html", result=result, image_url=filepath)
        except Exception as e:
            return render_template("index.html", error=str(e))
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
