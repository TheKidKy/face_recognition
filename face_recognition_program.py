import face_recognition as fr # to recognise faces
import numpy as np # to handle all lists/arrays
import cv2 # to capture footage
import os # to handle all matters relating to folders, paths, image/file names, etc.

faces_path = "~\\Desktop\\faces\\known" # replace your own folder path


# Function to get face names, as well as face encodings
def get_face_encoding():
    face_names = os.listdir(f"{faces_path}\\known")
    face_encoding = []

    # retrieves all face encodings and store them in a list
    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_path}\\known\\{name}")
        face_encoding.append(fr.face_encodings(face)[0])

        face_names[i] = name.split(".")[0] # to remove ".jpg" or any other image extension

    return face_encoding, face_names


# retrieving face encodings and storing them in the face_encodings variable, along with the names
face_encodings, face_names = get_face_encoding()

video = cv2.VideoCapture(0)

scl = 2 # scales down the webcam so the program runs faster

# continuously capturing webcam footage
while True:
    success, image = video.read()

    # making current frame smaller so program runs faster
    resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))

    # converting current frame to RGB, since that's what face recognition uses
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # retrieving face location coordinates and unknown encodings
    face_locations = fr.face_locations(rgb_image)
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    # Iterating through each encoding, as well as the face's location
    for face_encodings, face_location in zip(unknown_encodings, face_locations):

        # comparing known faces with unknown faces
        result = fr.compare_faces(face_encodings, face_encodings, 0.4)

        # getting correct name if a match was found
        if True in result:
            name = face_names[result.index(True)]

            # setting coordinates for face location
            top, right, bottom, left = face_location

            # drawing rectangle around face
            cv2.rectangle(image, (left*scl, top*scl), (right*scl, bottom*scl), (0, 0, 255), 2)

            # setting font, as well as displaying text of name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left*scl, bottom*scl * 20), font, 0.8, (255, 255, 255), 1)
