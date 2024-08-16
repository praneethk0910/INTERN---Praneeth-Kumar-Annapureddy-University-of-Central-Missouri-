import tensorflow as tf
from tensorflow.keras import layers, models
import pytesseract
from PIL import Image
import spacy
import cv2
import numpy as np
import random


nlp = spacy.load("en_core_web_sm")


def load_and_preprocess_image(img_path):
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error: Unable to load image at {img_path}. Please check the file path.")
    
    img = cv2.resize(img, (150, 150))  
    img = img.astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)  
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
        layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def classify_image(model, img_path):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    return "Positive" if prediction[0] > 0.5 else "Negative"


def extract_text_from_image(img_path):
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    return text


def extract_medical_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def get_medications(entities):
    
    medication_labels = ['PRODUCT', 'DRUG', 'MED', 'GPE', 'PERSON']  
    medications = [ent[0] for ent in entities if ent[1] in medication_labels]  
    return medications



if __name__ == "__main__":
    
    cv_image_paths = ["chest_xray1.png", "chest_xray2.png", "chest_xray3.png"]  
    ocr_image_path = "prescription.png"  

    
    cnn_model = build_cnn_model()

    
    random_image = random.choice(cv_image_paths)
    print(f"Selected Image for Classification: {random_image}")

    
    cv_result = classify_image(cnn_model, random_image)
    print(f"Computer Vision Result: {cv_result}")

    
    if cv_result == "Positive":
        print("Detected Positive. Extracting medication from the prescription...")

        
        extracted_text = extract_text_from_image(ocr_image_path)
        print(f"Extracted Text via OCR:\n{extracted_text}")

        
        medical_entities = extract_medical_entities(extracted_text)

        
        medications = get_medications(medical_entities)

        if medications:
            print("\nMedications Extracted:")
            for med in medications:
                print(med)
        else:
            print("No medications found in the prescription.")
    else:
        print("Result: Negative")
