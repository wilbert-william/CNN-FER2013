import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Activation

# Define a custom object scope to include 'Activation' as a custom object
custom_objects = {'Activation': Activation}

# Load pre-trained emotion recognition model with custom objects
emotion_model_path = 'C:\Skripsi\Enhancing-FER2013-Imbalance-main\Enhancing-FER2013-Imbalance-main\checkpoint/best_model_experiment1_2024-07-01.h5'
emotion_model = load_model(emotion_model_path, custom_objects=custom_objects)

# Load pre-trained face detection classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion map and color map
emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
color_map = {
    'anger': (0, 0, 255),       # Red
    'disgust': (255, 0, 0),     # Blue
    'fear': (0, 255, 255),      # Cyan
    'happiness': (0, 255, 0),   # Green
    'sadness': (255, 255, 0),   # Yellow
    'surprise': (255, 0, 255),  # Magenta
    'neutral': (128, 128, 128)  # Gray
}

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess the face ROI for emotion recognition model
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        face_roi_resized = face_roi_resized.astype('float32') / 255.0
        face_roi_resized = np.expand_dims(face_roi_resized, axis=0)
        face_roi_resized = np.expand_dims(face_roi_resized, axis=-1)
        
        # Predict emotion using the emotion recognition model
        predictions = emotion_model.predict(face_roi_resized)
        emotion_label = emotion_map[np.argmax(predictions)]
        color = color_map[emotion_label]
        
        # Draw rectangle around the face and display emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()