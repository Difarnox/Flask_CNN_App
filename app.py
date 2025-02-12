# Import library yang dibutuhkan
from flask import Flask, render_template, request, jsonify, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from datetime import datetime

# Inisialisasi Flask
app = Flask(__name__)

# Load model untuk prediksi (pastikan file model .h5 ada di direktori model/)
modelnasnet = load_model("model/NASNetMobile.h5")
modelvgg = load_model("model/VGG16.h5")
modelxception = load_model("model/Xception.h5")
modelcnn = load_model("model/scratchCNN.h5")

# Konfigurasi folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ekstensi file yang diperbolehkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

# Daftar kelas yang digunakan dalam klasifikasi
class_names = ['Sehat', 'Rusak', 'Terinfeksi']

# Fungsi untuk mengecek apakah file yang diunggah memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route utama untuk halaman depan
@app.route("/")
def main():
    return render_template("cnn_model.html")  # Halaman utama

# Route untuk menangani unggahan dan prediksi gambar
@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:  # Cek apakah ada file dalam request
        return jsonify({'message': 'No image in the request'}), 400

    file = request.files['file']  # Ambil file yang diunggah

    if not (file and allowed_file(file.filename)):  # Validasi ekstensi file
        return jsonify({'message': 'Invalid file type'}), 400

    # Simpan file dengan nama unik berdasarkan timestamp
    now = datetime.now().strftime("%d%m%y-%H%M%S")
    filename = f"{now}.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Konversi gambar ke RGB dan simpan kembali
    img = Image.open(file_path).convert('RGB')
    img.save(file_path, format="png")
    img.close()

    # Persiapan gambar untuk prediksi
    img = image.load_img(file_path, target_size=(128, 128, 3))  # Resize gambar
    x = image.img_to_array(img)  # Konversi ke array
    x = x / 127.5 - 1  # Normalisasi nilai piksel
    x = np.expand_dims(x, axis=0)  # Ubah ke bentuk batch
    images = np.vstack([x])  # Susun dalam vektor

    # Prediksi dengan semua model
    prediction_array_nasnet = modelnasnet.predict(images)
    prediction_array_vgg = modelvgg.predict(images)
    prediction_array_xception = modelxception.predict(images)
    prediction_array_cnn = modelcnn.predict(images)

    # Hasil prediksi dalam format JSON
    result = {
        "img_path": url_for('static', filename=f'uploads/{filename}'),
        "predictionnasnet": class_names[np.argmax(prediction_array_nasnet)],
        "confidencenasnet": '{:2.0f}%'.format(100 * np.max(prediction_array_nasnet)),
        "predictionvgg": class_names[np.argmax(prediction_array_vgg)],
        "confidencvgg": '{:2.0f}%'.format(100 * np.max(prediction_array_vgg)),
        "predictionxception": class_names[np.argmax(prediction_array_xception)],
        "confidencexception": '{:2.0f}%'.format(100 * np.max(prediction_array_xception)),
        "predictioncnn": class_names[np.argmax(prediction_array_cnn)],
        "confidencecnn": '{:2.0f}%'.format(100 * np.max(prediction_array_cnn))
    }

    return jsonify(result)

# Menjalankan aplikasi Flask dalam mode debug
if __name__ == '__main__':
    app.run(debug=True)