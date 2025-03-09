# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Define class labels
class_labels = {0: "Cat", 1: "Dog"}

# Path to new test images
test_folder = "afterTrainTestImages"

# Function to preprocess an image (same as training data processing)
def preprocess_image(img_path):
    img = cv2.imread(img_path)  # Read image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (consistent with training)
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Dictionary to track results
results = []

# Loop through subfolders (cats/dogs)
for category in ["cats", "dogs"]:
    category_path = os.path.join(test_folder, category)
    # Iterate over each image file in the category folder
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        # Ensure the file is an image (check file extension)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)
            predicted_label = class_labels[int(prediction[0] > 0.5)]  # Binary classification
            actual_label = "Cat" if category == "cats" else "Dog"

            # Store result
            results.append((img_name, actual_label, predicted_label))

            # Print result for each image
            print(f"Image: {img_name} | Actual: {actual_label} | Predicted: {predicted_label}")

# Print summary of model predictions
print("\n**Summary of Model Predictions:**")
correct = sum(1 for img, actual, predicted in results if actual == predicted)
total = len(results)
accuracy = (correct / total) * 100 if total > 0 else 0
# Print accuracy summary
print(f"Model Accuracy on Test Images: {accuracy:.2f}% ({correct}/{total} correct)")
# Print detailed results
print("\n🔍 **Detailed Results:**")
for img_name, actual, predicted in results:
    print(f"- {img_name}: Actual: {actual}, Predicted: {predicted}")
#All done now
print("Testing complete.")