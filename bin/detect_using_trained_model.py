from face_reg import logger
import cv2
import face_recognition
import pickle
import time


# Get a reference to webcam #0 (the default one)
videoPath = 0
modelPath = "data/model/knn.clf"
numJitters = 10
faceDetectionModel = "hog"
threshold = 0.3
numFrame = 0

# load video
videoCapture = cv2.VideoCapture(videoPath)

# load model KNN
logger.info("Loading model ...")
with open(modelPath, "rb") as f:
    knnClf = pickle.load(f)
logger.info("Model is loaded")
logger.info("Using model %s to detect face" % faceDetectionModel)

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break
    image = frame.copy()
    numFrame += 1
    prev = time.time()
    if numFrame > 1:
        frame = frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        faceLocations = face_recognition.face_locations(
            frame, model=faceDetectionModel)
        if len(faceLocations) == 0:
            logger.info("No face is detected")
        else:
            faceEncodings = face_recognition.face_encodings(
                frame, faceLocations, num_jitters=numJitters)
            closestDistances = knnClf.kneighbors(
                faceEncodings, n_neighbors=1)
            matches = [closestDistances[0][i][0] <=
                       threshold for i in range(len(faceLocations))]
            predictions = [
                (pred, loc) if rec
                else ("unknown", loc) for pred, loc, rec in zip(
                    knnClf.predict(
                        faceEncodings
                    ),
                    faceLocations,
                    matches
                )
            ]

            # Display the results
            for name, (top, right, bottom, left) in predictions:
                # Draw a box around the face
                cv2.rectangle(image, (left, top),
                              (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(image, (left, bottom - 35),
                              (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6),
                            font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Face Recognition', image)
        logger.debug(time.time() - prev)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
