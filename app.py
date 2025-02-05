from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Memuat Model CNN
model = load_model('model/cnn_model.h5')

# Label Kelas
class_labels = ['Sehat', 'Terinfeksi', 'Kekurangan Nutrisi']

# Fungsi untuk memeriksa tipe file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi Prediksi
def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100

        return class_labels[class_index], confidence
    except Exception as e:
        return "Error during prediction", 0

# Route Halaman Utama
@app.route('/', methods=['GET', 'POST'])
def index():
    accuracy_data = []
    loss_data = []
    filename = None
    label = None
    confidence = None

    if request.method == 'POST':
        if 'clear' in request.form:  # Tombol Clear ditekan
            return redirect(url_for('index'))  # Reset halaman tanpa data

        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Prediksi Gambar
            label, confidence = model_predict(file_path, model)

            # Membaca data akurasi dan loss dari JSON (hasil training)
            try:
                with open('model/training_history.json', 'r') as f:
                    history = json.load(f)
                    accuracy_data = history.get('accuracy', [])
                    loss_data = history.get('loss', [])
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # Abaikan jika file tidak ditemukan atau JSON error

            return render_template('index.html',
                                   filename=filename,
                                   label=label,
                                   confidence=confidence,
                                   accuracy_data=accuracy_data,
                                   loss_data=loss_data)

        else:
            return render_template('index.html', error="Invalid file type. Please upload an image (png, jpg, jpeg).")

    return render_template('index.html',
                           filename=filename,
                           label=label,
                           confidence=confidence,
                           accuracy_data=accuracy_data,
                           loss_data=loss_data)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Pastikan folder upload ada
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
