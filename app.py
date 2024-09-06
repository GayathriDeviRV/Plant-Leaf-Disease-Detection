from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/best_model.keras')
UPLOAD_FOLDER = r'C:\Users\GAYATHRI DEVI R V\Desktop\pd\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess the image


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array


# Dictionary with diagnoses for each class and actions
diagnoses = {
    'Healthy': {
        'diagnosis': 'No visible symptoms of disease. Leaves are green and intact. Plant growth is normal.',
        'actions': [
            'Continue regular watering and maintenance.',
            'Ensure adequate sunlight and proper spacing between plants.',
            'Monitor plants regularly for any signs of stress or disease.',
            'Apply balanced fertilizers as needed.'
        ]
    },
    'Rust': {
        'diagnosis': 'Small, yellow to orange pustules on the underside of leaves. Leaves may turn yellow and drop prematurely. In severe cases, plant growth may be stunted.',
        'actions': [
            'Remove and destroy infected leaves to prevent the spread.',
            'Improve air circulation around plants by proper spacing and pruning.',
            'Avoid overhead watering to reduce leaf wetness.',
            'Apply fungicides as recommended by agricultural extensions or local experts.',
            'Rotate crops to prevent the buildup of rust pathogens in the soil.'
        ]
    },
    'Powdery': {
        'diagnosis': 'White, powdery fungal growth on leaves, stems, and buds. Infected leaves may curl, turn yellow, and drop. Flower and fruit production may be affected.',
        'actions': [
            'Remove and destroy affected plant parts.',
            'Water plants at the base to keep foliage dry.',
            'Improve air circulation and reduce humidity around plants.',
            'Apply fungicides labeled for powdery mildew control.',
            'Choose resistant plant varieties if available.',
            'Ensure plants are not overcrowded to reduce moisture buildup.'
        ]
    }
}

# Predict the class of the image


def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    class_indices = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
    predicted_class = class_indices[np.argmax(prediction)]
    confidence = np.max(prediction)
    diagnosis_info = diagnoses[predicted_class]
    return predicted_class, confidence, diagnosis_info


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")  # Debugging line
            predicted_class, confidence, diagnosis_info = predict_image(
                file_path)
            return render_template('index.html', prediction=predicted_class, confidence=confidence, diagnosis_info=diagnosis_info, image_path=file.filename)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
