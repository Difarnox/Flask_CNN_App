import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Pastikan TensorFlow mengenali GPU jika tersedia
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Direktori dataset
train_dir = "dataset/train"
test_dir = "dataset/test"
val_dir = "dataset/validation"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load data training
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=64, class_mode="categorical")

# Load data validasi
test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=64, class_mode="categorical")

val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=64, class_mode="categorical")

# Jumlah kelas
num_classes = len(train_generator.class_indices)
print(f"Jumlah kelas: {num_classes}, Label: {train_generator.class_indices}")

# Membangun model CNN sesuai Percobaan C
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Menampilkan arsitektur model
model.summary()

# Latih model
history = model.fit(train_generator, epochs=50, validation_data=val_generator)

# Simpan model
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "scratchCNN.h5"))
print("Model telah disimpan sebagai scratchCNN.h5")

# Evaluasi model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Akurasi Model pada Data Testing: {test_acc:.2%}")

# Simpan grafik
static_images_dir = "static/images"
os.makedirs(static_images_dir, exist_ok=True)

# Plot grafik akurasi dan loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(static_images_dir, 'training_validation_graphs.png'))
plt.close()

# Prediksi untuk confusion matrix
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Label Asli')
plt.savefig(os.path.join(static_images_dir, 'confusion_matrix.png'))
plt.close()

# ROC Curve
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(y_true)
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(static_images_dir, 'roc_curve.png'))
plt.close()

print("Grafik, confusion matrix, dan ROC curve telah disimpan di folder static/images.")
