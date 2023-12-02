import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model_path = "mask_model.h5"
model = load_model(model_path)

# Load the face detection model (you may need to download the appropriate Haarcascades file)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Function for live detection with smoothing
def live_detection(model, face_cascade, confidence_threshold=0.6, smoothing_factor=0.8):
    # Open the camera and store probabilities from each frame.
    cap = cv2.VideoCapture(0)

    mask_prob_history = []

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            # Extract the face region
            face_roi = frame[y : y + h, x : x + w]

            # Preprocess the face for the mask detection model
            face_array = cv2.resize(face_roi, (224, 224))
            face_array = img_to_array(face_array)
            face_array = preprocess_input(face_array)
            face_array = np.expand_dims(face_array, axis=0)

            # Make predictions using the mask detection model
            mask_prob = model.predict(face_array)[0][0]

            # Apply smoothing using a simple moving average
            mask_prob_history.append(mask_prob)
            if len(mask_prob_history) > 5:
                mask_prob_history.pop(0)

            smoothed_mask_prob = np.mean(mask_prob_history)

            # Correct label assignment based on smoothed probability
            label = "Mask" if smoothed_mask_prob > confidence_threshold else "No Mask"

            # Display the result on the frame
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Show the frame
        cv2.imshow("Mask Detection", frame)

        # Break the loop if 'c' is pressed
        if cv2.waitKey(1) & 0xFF == ord("c"):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Call the live_detection function with smoothing
live_detection(model, face_cascade)
