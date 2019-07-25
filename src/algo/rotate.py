# import cv2
import face_recognition


def rotate(image, model="hog"):
    faceLocations = face_recognition.face_locations(
        image, model=model)
    return faceLocations
