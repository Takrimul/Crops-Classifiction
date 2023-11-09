import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the saved model
loaded_model = keras.models.load_model('trained_resnet152_model.h5')


# Function to preprocess the input image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet.preprocess_input(image)
    return image


# Load and preprocess a single image
def load_and_preprocess_single_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image)
    return image


# List of image paths
input_image_paths = [
    r"E:\7th Semester\A test\paddy1.jpeg",
    r"E:\7th Semester\A test\paddy.jpeg",
    r"E:\7th Semester\A test\chilli1.jpeg",
    r"E:\7th Semester\A test\chilli2.jpeg",
    r"E:\7th Semester\A test\corn.jpeg",
    r"E:\7th Semester\A test\lentil.jpeg",
    r"E:\7th Semester\A test\lentil1.jpeg",
    r"E:\7th Semester\A test\mustard.jpeg",
    r"E:\7th Semester\A test\onion1.jpeg",
    r"E:\7th Semester\A test\onion2.jpeg",
    r"E:\7th Semester\A test\potato.jpeg",
    r"E:\7th Semester\A test\potato1.jpeg",
    r"E:\7th Semester\A test\sug.jpeg",
    r"E:\7th Semester\A test\sugarcane.jpeg",
    r"E:\7th Semester\A test\wheat.jpeg",
    r"E:\7th Semester\A test\whet.jpeg"



    # Add more image paths here
]

# Process and predict for each image
for input_image_path in input_image_paths:
    input_image = load_and_preprocess_single_image(input_image_path)

    # Expand dimensions to match the model's input shape (batch_size, height, width, channels)
    input_image = np.expand_dims(input_image, axis=0)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_image)
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Display the predicted label
    class_names = ['Jute', 'Paddy', 'Sugarcane', 'Wheat', 'Corn', 'Potato', 'Lentil', 'Chilli', 'Mustard', 'Onion']
    print("Image:", input_image_path)
    print("Predicted Label:", class_names[predicted_label])
    print()
