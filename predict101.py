import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the saved model
loaded_model = keras.models.load_model('trained_resnet101_model.h5')

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

# Replace 'your_image_path.jpg' with the path to your image file
input_image_path = r"E:\7th Semester\A test\wheat.jpeg"
input_image = load_and_preprocess_single_image(input_image_path)

# Expand dimensions to match the model's input shape (batch_size, height, width, channels)
input_image = np.expand_dims(input_image, axis=0)

# Make predictions using the loaded model
predictions = loaded_model.predict(input_image)
predicted_label = np.argmax(predictions, axis=1)[0]

# Display the predicted label
class_names = ['Jute', 'Paddy', 'Sugercane', 'Wheat', 'Corn', 'Potato', 'Lentil', 'Chilli', 'Mustard', 'Onion']
print("Predicted Label:", class_names[predicted_label])
