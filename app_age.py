import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import streamlit as st
from PIL import Image

# Load MTCNN face detector
detector = MTCNN()

# Load FaceNet model
facenet_model = load_model('models/facenet_model.h5')

# Load face embeddings and labels from npz file
with np.load('models/Indian-celeb-embeddings.npz') as data:
    embeddings = data['arr_0']
    labels = data['arr_1']

# Load Age and Gender models
age_model = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('models/gender_deploy.prototxt', 'models/gender_net.caffemodel')

# Define age and gender categories
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Define function to extract face embeddings
def extract_face_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    embedding = facenet_model.predict(face)
    return embedding

# Define function to recognize faces
def recognize_faces(face_embeddings):
    distances = np.linalg.norm(face_embeddings - embeddings, axis=1)
    indices = np.argmin(distances)
    return labels[indices]

# Define function to predict age and gender
def predict_age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    # Predict gender
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = gender_list[gender_preds[0].argmax()]
    
    # Predict age
    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = age_list[age_preds[0].argmax()]
    
    return age, gender

# Streamlit app
st.title("Face Recognition with Age and Gender Detection")
st.write("Upload an image to detect, recognize faces, and predict age and gender.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_file))
    
    # Detect faces in the image
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        st.write("No faces detected.")
    else:
        for face in faces:
            x, y, width, height = face['box']
            face_image = image[y:y+height, x:x+width]
            try:
                # Recognize face
                embedding = extract_face_embeddings(face_image)
                recognized_label = recognize_faces(embedding)
                
                # Predict age and gender
                age, gender = predict_age_gender(face_image)
                label = f"{recognized_label}, {gender}, {age}"
            except Exception as e:
                label = "Error"
            
            # Draw rectangles and labels on detected faces
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the processed image
        st.image(image, caption="Processed Image", use_column_width=True)
