import os
import cv2
from datetime import datetime


class Face:
    max_recog_time = 3

    def __init__(self, face_name):
        self.name = face_name
        self.recog_time = 0
        self.last_recog = -1
        self.is_recognized = False
        self.last_locat = ()
        self.last_image = None

    def recognize(self, last_reconition, last_location):
        self.last_recog = last_reconition
        self.last_locat = last_location
        self.is_recognized = True
        return self

    def update(self):
        if self.is_recognized:
            self.recog_time = min(Face.max_recog_time, self.recog_time + 1)
        else:
            self.recog_time = max(0, self.recog_time - 1)
        self.is_recognized = False
        return self

    def should_save(self):
        return (
            (not self.is_recognized) and self.recog_time == Face.max_recog_time
        )

    def save(self, image, path, location=None):
        self.last_image = image
        if location:
            image = Face.get_face(image, location)
        time_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # cv2.imwrite(
        #     os.path.join(path, time_info) + ".png",
        #     image
        # )

    def get_face(image, location):
        return image[location[0]:location[2],
                     location[3]:location[1]]


if __name__ == "__main__":
    face1 = Face()
    face1.recognize(32).update()
    face1.recognize(32).update()
    face1.recognize(32).update()
    print(face1.recog_time)
    face1.update()
    face1.update()
    print(face1.recog_time)
