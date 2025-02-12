import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load Xception dengan pre-trained weights dari ImageNet
base_model = Xception(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Bekukan sebagian layer awal Xception agar tidak berubah saat training
for layer in base_model.layers[:-4]:  # Freeze semua layer kecuali 4 layer terakhir
    layer.trainable = False

# Tambahkan lapisan tambahan untuk klasifikasi 3 kelas
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dense(3, activation="softmax")(x)  # Output untuk 3 kelas

# Buat model akhir
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Data augmentation untuk training dan validasi
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset training
train_generator = train_datagen.flow_from_directory(
    "dataset/train", target_size=(128, 128), batch_size=16, class_mode="categorical"
)

# Load dataset validasi
val_generator = val_datagen.flow_from_directory(
    "dataset/validation", target_size=(128, 128), batch_size=16, class_mode="categorical"
)

# Folder untuk menyimpan grafik
static_images_dir = "static/images"
if not os.path.exists(static_images_dir):
    os.makedirs(static_images_dir)  # Membuat folder static/images jika belum ada

# Implementasikan EarlyStopping dan ReduceLROnPlateau untuk meningkatkan akurasi
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6)

# Training model dengan validasi
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# Simpan model dalam format .h5
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save(os.path.join(model_dir, "Xception.h5"))
print("Model Xception telah disimpan di 'model/Xception.h5'")

# Evaluasi model dengan data testing
test_generator = val_datagen.flow_from_directory(
    "dataset/test", target_size=(128, 128), batch_size=16, class_mode="categorical"
)

y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ambil label kelas yang benar
y_true = test_generator.classes

# Hitung confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot grafik akurasi dan loss
# Grafik Akurasi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)  # Menambahkan grid pada grafik

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)  # Menambahkan grid pada grafik

# Simpan grafik akurasi dan loss
plt.savefig(os.path.join(static_images_dir, 'training_validation_graphs_xception.png'))
plt.close()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Label Asli')

# Simpan confusion matrix sebagai gambar PNG
plt.savefig(os.path.join(static_images_dir, 'confusion_matrix_xception.png'))
plt.close()

# Menambahkan ROC Curve
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(y_true)
fpr, tpr, thresholds = roc_curve(y_true_bin.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)  # Menambahkan grid pada grafik

# Simpan ROC curve sebagai gambar PNG
plt.savefig(os.path.join(static_images_dir, 'roc_curve_xception.png'))
plt.close()

print("Grafik, confusion matrix, dan ROC curve telah disimpan di folder static/images.")
