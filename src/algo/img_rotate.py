import face_recognition
import numpy as np


def rotate_image(image, rotate_time=0):
    for i in range(rotate_time % 4):
        image = np.rot90(image)
    return image


def rotate_locations(locations, shape, rotate_time=0):
    return_location = locations
    for i in range(rotate_time):
        new_locations = []
        for location in locations:
            new_location = [location[-1]] + list(location[:-1])
            new_location[1] = shape[1] - new_location[1]
            new_location[3] = shape[1] - new_location[3]
            new_locations.append(tuple(new_location))
        return_location = new_locations
    return return_location


def rotate(image, model="hog", rotate_time=0):
    img_shape = image.shape
    image = rotate_image(image, rotate_time)
    faceLocations = face_recognition.face_locations(
        image, model=model)
    faceLocations = rotate_locations(faceLocations, img_shape, rotate_time)
    return faceLocations
