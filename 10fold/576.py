import joblib
import cv2
import os
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Resize the image to a smaller size
    resized_image = cv2.resize(image, (128, 128))
    
    win_size = (128, 128)
    block_size = (16, 16)
    block_stride = (16, 16)
    cell_size = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized_image)
    return hog_features.ravel()

# Create lists to store HOG features and labels
hog_features = []
labels = []

# Define the paths to your image folders
covid19_folder = "C:/Users/heman/Desktop/SRM/urop/selection/ap,pa,aps"
pneumonia_folder = "C:/Users/heman/Desktop/SRM/urop/chest_xray/test/PNEUMONIA"
normal_folder = "C:/Users/heman/Desktop/SRM/urop/chest_xray/test/NORMAL"

# Load HOG features and labels for "covid19" images
for filename in os.listdir(covid19_folder):
    image = cv2.imread(os.path.join(covid19_folder, filename))
    hog_feature = extract_hog_features(image)
    hog_features.append(hog_feature)
    labels.append(1)  # Label 1 for "covid19"
    
# Load HOG features and labels for "pneumonia" images
for filename in os.listdir(pneumonia_folder):
    image = cv2.imread(os.path.join(pneumonia_folder, filename))
    hog_feature = extract_hog_features(image)
    hog_features.append(hog_feature)
    labels.append(0)  # Label 0 for "pneumonia"
    
# Load HOG features and labels for "normal" images
for filename in os.listdir(normal_folder): 
    image = cv2.imread(os.path.join(normal_folder, filename))
    hog_feature = extract_hog_features(image)
    hog_features.append(hog_feature)
    labels.append(2)  # Label 2 for "normal"

# Split the data into training and testing sets (cross-validation will be performed)
X = hog_features
y = labels

# Initialize the SVM model
model = SVC()

# Initialize 10-fold cross-validation
cv = StratifiedKFold(n_splits=10)

# Perform 10-fold cross-validation and get predictions
y_pred = cross_val_predict(model, X, y, cv=cv)

# Generate the confusion matrix
cm = confusion_matrix(y, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate overall accuracy
accuracy = sum(np.diag(cm)) / np.sum(cm)
print("Overall Accuracy:", accuracy)

# Train the model on the entire dataset
model.fit(X, y)

# Save the trained model using joblib
filename = "C:/Users/heman/Desktop/SRM/urop/10fold/2025.sav"
joblib.dump(model, filename)

print("Model trained, saved, and confusion matrix generated successfully.")
