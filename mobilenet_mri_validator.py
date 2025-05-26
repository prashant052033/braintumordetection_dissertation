# -------------------------------
# MRI vs Non-MRI Classifier using MobileNetV2
# -------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------------------
# STEP 1: Image Setup
# -------------------------------

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Set dataset paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Image augmentation for training
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -------------------------------
# STEP 2: MobileNetV2 Model
# -------------------------------

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# Custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# STEP 3: Train the Model
# -------------------------------

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------------
# STEP 4: Save the Model
# -------------------------------

model.save("mobilenet_mri_validator.h5")
print("✅ Model saved as mobilenet_mri_validator.h5")

# -------------------------------
# STEP 5: Predict Single Image
# -------------------------------

def is_mri_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)[0][0]
    return pred > 0.5  # True = MRI, False = not

# Test
test_path = "test_image.jpg"  # Change this to your test image
if os.path.exists(test_path):
    result = is_mri_image(test_path)
    print("✅ MRI Image" if result else "❌ Not an MRI Image")
else:
    print(f"⚠️ {test_path} not found.")
