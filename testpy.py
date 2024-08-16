import tensorflow as tf
from tensorflow.keras import layers, models
import pytesseract
from PIL import Image
import spacy
import cv2
import numpy as np

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Step 1: Computer Vision (CV) - Classify a medical image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))  # Resize image for CNN input
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dims for batch input
    return img

def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification for simplicity
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def classify_image(model, img_path):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    return "Positive" if prediction[0] > 0.5 else "Negative"

# Step 2: Optical Character Recognition (OCR) - Extract text from classified image
def extract_text_from_image(img_path):
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    return text

# Step 3: Natural Language Processing (NLP) - Extract medical entities from the text
def extract_medical_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Main Execution Flow
if __name__ == "__main__":
    # Example image file paths (replace with actual image paths)
    cv_image_path = "chest_xray.png"  # Image for classification
    ocr_image_path = "prescription.png"  # Image for OCR

    # Load CNN model (you would usually load pre-trained weights here)
    cnn_model = build_cnn_model()

    # Step 1: Run Computer Vision model to classify chest X-ray image
    cv_result = classify_image(cnn_model, cv_image_path)
    print(f"Computer Vision Result: {cv_result}")

    # Step 2: Run OCR to extract text from the prescription image
    extracted_text = extract_text_from_image(ocr_image_path)
    print(f"Extracted Text via OCR:\n{extracted_text}")

    # Step 3: Run NLP on the extracted text to identify medical entities
    medical_entities = extract_medical_entities(extracted_text)
    print("\nMedical Entities Extracted via NLP:")
    for entity in medical_entities:
        print(f"{entity[0]} -> {entity[1]}")
