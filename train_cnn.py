import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# Dataset direktori
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Menggunakan ImageDataGenerator untuk preprocessing data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Menyimpan model
model.save('model/cnn_model.h5')

# Menyimpan riwayat akurasi dan loss
history_dict = history.history
with open('model/training_history.json', 'w') as f:
    json.dump(history_dict, f)

# Menentukan epoch terbaik berdasarkan akurasi validasi tertinggi
best_epoch = max(enumerate(history_dict['val_accuracy']), key=lambda x: x[1])[0] + 1

# Membuat grafik akurasi
plt.figure(figsize=(8, 6))
plt.plot(history_dict['accuracy'], label='Akurasi Pelatihan', marker='o')
plt.plot(history_dict['val_accuracy'], label='Akurasi Validasi', marker='x')
plt.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title('Grafik Akurasi Model CNN')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)
plt.savefig('static/accuracy_loss_plot.png')
plt.close()

# Membuat grafik loss
plt.figure(figsize=(8, 6))
plt.plot(history_dict['loss'], label='Loss Pelatihan', marker='o')
plt.plot(history_dict['val_loss'], label='Loss Validasi', marker='x')
plt.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title('Grafik Loss Model CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('static/accuracy_loss_plot_loss.png')
plt.close()
