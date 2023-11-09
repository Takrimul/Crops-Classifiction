import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

# Load the CSV file
csv_path = 'D:\ML_Project\DATA-CODE\cropsdataset.csv'
data = pd.read_csv(csv_path)

# Assuming 'path' column contains image file paths and 'label' column contains labels
image_paths = data['path'].values
labels = data['label'].values

# Load and preprocess images
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # Resize images to a common size
    image = tf.keras.applications.resnet.preprocess_input(image)  # Preprocess based on the chosen CNN architecture
    return image, label

# Create a dataset from the image paths and labels
image_dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
image_dataset = image_dataset.map(load_and_preprocess_image)

# Split the dataset into training and validation sets
batch_size = 32
train_dataset = image_dataset.take(3600).batch(batch_size)
val_dataset = image_dataset.skip(3600).batch(batch_size)

# Define the ResNet-152 model
base_model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 5
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Save the model
model.save('trained_resnet152_model.h5')
print("Model saved.")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Generate predictions
val_images = np.array([image for image, label in val_dataset.unbatch()])
val_labels = np.array([label for image, label in val_dataset.unbatch()])
predictions = model.predict(val_images)
predicted_labels = np.argmax(predictions, axis=1)

# Generate and print the classification report
target_names = ['Jute', 'Paddy', 'Sugercane', 'Wheat', 'Corn', 'Potato', 'Lentil', 'Chilli', 'Mustard', 'Onion']
report = classification_report(val_labels, predicted_labels, target_names=target_names)
print(report)
