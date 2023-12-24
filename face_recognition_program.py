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