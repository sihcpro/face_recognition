import os
import os.path
from face_recognition.face_recognition_cli import image_files_in_folder
from knn import predict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate(evaluateDir, modelPath):
    right = 0
    wrong = 0
    # Loop through each person in the training set
    for classDir in os.listdir(evaluateDir):
        if not os.path.isdir(os.path.join(evaluateDir, classDir)):
            continue
        # Loop through each training image for the current person
        for imgPath in image_files_in_folder(
            os.path.join(evaluateDir, classDir)
        ):
            faces = predict(imgPath, model_path=modelPath)
            if faces:
                # There is only 1 face
                face = faces[0]
                face_name = face[0]
                if face_name == classDir:
                    right = right + 1
                else:
                    wrong = wrong + 1
    return right / (right + wrong) * 100


if __name__ == '__main__':
    modelPath = "data/model/knn.clf"
    logger.warning("Accuracy: %s" % evaluate('data/test', modelPath))
