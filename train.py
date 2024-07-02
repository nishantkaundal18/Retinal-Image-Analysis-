import pandas as pd

# Load the CSV file with headers
csv_path = "D:\\diabetic retinopathy dataset\\train.csv"
df = pd.read_csv(csv_path)

# Display the first few rows to check if it's correct
print(df.head())


import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define the path to the images
image_base_path = "D:\\diabetic retinopathy dataset\\colored_images"

# Mapping from class label to folder name
class_folder_mapping = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# Function to load images
def load_images(df, base_path, class_mapping):
    images = []
    labels = []
    missing_files = 0
    for index, row in df.iterrows():
        img_folder = class_mapping[row['diagnosis']]
        img_path = os.path.join(base_path, img_folder, f"{row['id_code']}.png")
        
        # Check if the file exists
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            missing_files += 1
            continue
        
        image = load_img(img_path, target_size=(224, 224))  # Resize images to 224x224
        image = img_to_array(image) / 255.0  # Normalize image
        images.append(image)
        labels.append(row['diagnosis'])
    
    if missing_files > 0:
        print(f"Total missing files: {missing_files}")
    
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images(df, image_base_path, class_folder_mapping)

print(images.shape)
print(labels.shape)


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Convert labels to categorical
num_classes = df['diagnosis'].nunique()
labels = to_categorical(labels, num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add dropout layer
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

model.save('diabetic_retinopathy_model.h5')