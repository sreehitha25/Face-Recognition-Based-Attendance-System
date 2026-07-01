"""
train_model.py
---------------
Reads all captured face images from dataset/ and trains an OpenCV LBPH
(Local Binary Patterns Histograms) face recognizer. Saves the trained
model to trainer/trainer.yml for use during attendance marking.

Run standalone:
    python train_model.py
"""

import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
TRAINER_PATH = os.path.join("trainer", "trainer.yml")


def train_recognizer(progress_callback=None):
    """
    Trains the LBPH recognizer on all images in dataset/.
    Folder naming convention: dataset/<person_id>_<name>/*.jpg

    Returns (num_people, num_images) trained on.
    """
    face_samples = []
    ids = []

    person_folders = [
        f for f in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, f))
    ]

    for i, folder in enumerate(person_folders):
        person_id = int(folder.split("_")[0])
        folder_path = os.path.join(DATASET_DIR, folder)

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue
            face_samples.append(gray_img)
            ids.append(person_id)

        if progress_callback:
            progress_callback(i + 1, len(person_folders))

    if not face_samples:
        raise ValueError("No training images found. Register at least one person first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))

    os.makedirs("trainer", exist_ok=True)
    recognizer.write(TRAINER_PATH)

    return len(person_folders), len(face_samples)


if __name__ == "__main__":
    n_people, n_images = train_recognizer()
    print(f"Training complete. Trained on {n_people} people, {n_images} images.")
    print(f"Model saved to {TRAINER_PATH}")
