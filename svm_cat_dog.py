import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# Dataset Path
# =====================================

dataset_path = r"archive\training_set\training_set"

categories = ["cats", "dogs"]

print("Dataset exists :", os.path.exists(dataset_path))
print("Cats folder    :", os.path.exists(os.path.join(dataset_path, "cats")))
print("Dogs folder    :", os.path.exists(os.path.join(dataset_path, "dogs")))

# =====================================
# Parameters
# =====================================

IMG_SIZE = 32
MAX_IMAGES = 1000

data = []
labels = []

print("\nLoading images...\n")

# =====================================
# Load Images
# =====================================

for label, category in enumerate(categories):

    folder = os.path.join(dataset_path, category)

    print(f"Reading images from: {folder}")

    count = 0

    for image_name in os.listdir(folder):

        if count >= MAX_IMAGES:
            break

        image_path = os.path.join(folder, image_name)

        try:
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            data.append(image.flatten())
            labels.append(label)

            count += 1

        except Exception:
            continue

print("\nImages Loaded Successfully!")
print("Total Images :", len(data))

# =====================================
# Convert to NumPy Arrays
# =====================================

X = np.array(data)
y = np.array(labels)

# =====================================
# Split Dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Train Model
# =====================================

print("\nTraining SVM...\n")

model = LinearSVC(max_iter=5000)

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================
# Prediction
# =====================================

predictions = model.predict(X_test)

# =====================================
# Accuracy
# =====================================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions,
    target_names=categories
))