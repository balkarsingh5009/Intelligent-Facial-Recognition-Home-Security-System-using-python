import face_recognition
import cv2
import numpy as np
import os
import uuid
import time

video_capture = cv2.VideoCapture(0)

image1 = face_recognition.load_image_file(os.path.abspath("C:/Users/DELL/Desktop/source_facerecog/codfacerecog/recognize/images/loveleen.jpeg"))
image1_face_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file(os.path.abspath("C:/Users/DELL/Desktop/source_facerecog/codfacerecog/recognize/images/sidhu.jpg"))
image2_face_encoding = face_recognition.face_encodings(image2)[0]

known_face_encodings = [
    image1_face_encoding,   
    image2_face_encoding
]
known_face_names = [
    "Loveleen",
    "Sidhu"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Initialize variables for capturing photo duration
start_time = None
capture_duration = 5  # in seconds

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model='small')

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    cv2.imshow('frame', frame)

    # Check if a person is present
    if len(face_locations) > 0:
        for i, (top, right, bottom, left) in enumerate(face_locations):
            if not start_time:
                # Start capturing photo duration
                start_time = time.time()
            elif time.time() - start_time >= capture_duration:
                # Capture and save the image
                top = max(0, top - 20)  # Adjust top coordinate
                right = min(frame.shape[1], right + 20)  # Adjust right coordinate
                bottom = min(frame.shape[0], bottom + 20)  # Adjust bottom coordinate
                left = max(0, left - 20)  # Adjust left coordinate
                face_image = frame[top*4:(bottom+1)*4, left*4:(right+1)*4]
                filename = f'captured_image_{uuid.uuid4()}.jpg'
                cv2.imwrite(filename, face_image)
                print(f"Image {i+1} captured and saved as {filename}")
                start_time = None

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
video_capture.release()
cv2.destroyAllWindows()

