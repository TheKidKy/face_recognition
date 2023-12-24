import face_recognition as fr # to recognise faces
import numpy as np # to handle all lists/arrays
import cv2 # to capture footage
import os # to handle all matters relating to folders, paths, image/file names, etc.

faces_path = "~/Desktop/faces/known" # replace your own folder path


# Function to get face names, as well as face encodings
def get_face_encoding():
    face_names = os.listdir(f"{faces_path}\\known")
    face_encoding = []

