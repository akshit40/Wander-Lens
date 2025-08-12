import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 5       # Start with fewer epochs to run faster initially
EPOCHS_FINE_TUNE = 10    # Epochs for fine-tuning more layers
LEARNING_RATE_INITIAL = 0.001
LEARNING_RATE_FINE_TUNE = 0.00005

base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'data', 'train')
test_dir = os.path.join(base_dir, 'data', 'test')
model_save_path = os.path.join(base_dir, 'saved_model', 'landmark_model.h5')

# --- 1. Data Preparation and Augmentation ---
# Create a data generator for the training set with aggressive augmentation
# This simulates "lazy" photos (rotations, zooms, etc.)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# The validation/test set should not be augmented, only rescaled
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators that will read images from the sub-folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- 2. Model Building (Transfer Learning) ---
# Load the MobileNetV2 model, pre-trained on ImageNet, without its top classifier
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, # We are providing our own top layer
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Freeze the layers of the base model so they are not trained initially
base_model.trainable = False

# Add our custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Dropout layer to prevent overfitting
x = Dense(1024, activation='relu')(x)
num_classes = len(train_generator.class_indices)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Initial Training (Training just the new top layers) ---
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_INITIAL),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("--- 🚀 Starting Initial Training (Transfer Learning) ---")
history_initial = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=validation_generator
)

# --- 4. Fine-Tuning ---
# Unfreeze more layers of the base model to fine-tune it on our specific images
base_model.trainable = True
fine_tune_at = 100 # Freeze the first 100 layers and fine-tune the rest
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with a much lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- 🚀 Starting Fine-Tuning ---")
total_epochs = EPOCHS_INITIAL + EPOCHS_FINE_TUNE
history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history_initial.epoch[-1], # Continue from where we left off
    validation_data=validation_generator
)

# --- 5. Save Model and Plot History ---
print("\n--- ✅ Training complete. Saving model... ---")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Combine training histories to create a single plot
acc = history_initial.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history_initial.history['loss'] + history_fine_tune.history['loss']
val_loss = history_initial.history['val_loss'] + history_fine_tune.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(total_epochs), acc, label='Training Accuracy')
plt.plot(range(total_epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(total_epochs), loss, label='Training Loss')
plt.plot(range(total_epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('accuracy_loss_plot.png')
print("📈 Accuracy and Loss plot saved as accuracy_loss_plot.png")