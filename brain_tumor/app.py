from flask import Flask, request, jsonify, render_template
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')
model = load_model("brain_tumor_model.h5")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = np.array(img)
    return img

def get_className(class_index):
    class_labels = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
    return class_labels[class_index]

def highlight_tumor(image_path, mask):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img[mask == 255] = [0, 0, 255]  
    return img

@app.route('/', methods=['GET', 'POST'])
def predict_and_highlight_tumor():
    highlighted_image_path = ""
    if request.method == 'POST':
        image_files = request.files.getlist('file')        
        image_dir = "static"      
        os.makedirs(image_dir, exist_ok=True)      
        results = []  
        for image in image_files:
            image_path = os.path.join(image_dir, image.filename)
            image.save(image_path)
            input_image = preprocess_image(image_path)
            pred_probs = model.predict(np.expand_dims(input_image, axis=0))[0]
            pred_class_index = np.argmax(pred_probs)
            result = get_className(pred_class_index)
            confidence = float(pred_probs[pred_class_index]) * 100 

            results.append({"result": result, "confidence": confidence})
            tumor_mask = np.zeros_like(input_image, dtype=np.uint8)
            tumor_mask[input_image > 128] = 255
            highlighted_image = highlight_tumor(image_path, tumor_mask)
            highlighted_image_path = os.path.join(image_dir, f"highlighted_{image.filename}")
            cv2.imwrite(highlighted_image_path, highlighted_image)

        return jsonify(results=results, highlighted_image_path=highlighted_image_path)

    return render_template('index.html', results=None, highlighted_image_path=None, symptoms=None, precautions=None)

if __name__ == '__main__':
    app.run(debug=True)
