import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Function to load and preprocess a single image
def load_and_process_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input_shape
    return img_array

# Define the path to the saved model
model_path = 'diabetic_retinopathy_model.h5'

# Load the saved model
model = load_model(model_path)

# Function to predict class label for a single image
def predict_class(image_path, model):
    # Load and preprocess the image
    img_array = load_and_process_image(image_path)
    
    # Predict the class probabilities
    predictions = model.predict(img_array)
    
    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(predictions)
    
    # Mapping from index to class label
    class_mapping = {
        0: "No_DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferate_DR"
    }
    
    # Get the predicted class label
    predicted_class = class_mapping[predicted_class_index]
    
    return predicted_class, predictions[0]

# Example usage:
image_path = "D:\\diabetic retinopathy dataset\\colored_images\\Mild\\00cb6555d108.png"  # Replace with the path to your image
predicted_class, class_probabilities = predict_class(image_path, model)

print(f"Predicted Class: {predicted_class}")
print(f"Class Probabilities: {class_probabilities}")

image_path = "D:\\diabetic retinopathy dataset\\colored_images\\Moderate\\000c1434d8d7.png"  # Replace with the path to your image
predicted_class, class_probabilities = predict_class(image_path, model)

print(f"Predicted Class: {predicted_class}")
print(f"Class Probabilities: {class_probabilities}")
