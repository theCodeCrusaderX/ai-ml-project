from flask import Flask, request, render_template
import pickle
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (64, 64))  # Resize image to 64x64
    img = img.flatten().reshape(1, -1)  # Flatten and reshape
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']  # Get the uploaded file
        if file:
            image = Image.open(file)  # Open the image
            img = preprocess_image(image)  # Preprocess the image
            prediction = model.predict(img)  # Predict using the loaded model
            labels = ['COVID', 'Normal', 'Viral Pneumonia']  # Possible labels
            result = labels[prediction[0]]  # Get the result label
            return render_template('result.html', result=result)  # Show the result
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
