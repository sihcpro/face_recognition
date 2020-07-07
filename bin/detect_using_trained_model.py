from algo import img_rotate, nearest_face, img_quality
from face_reg import logger
from face_reg.face import Face
from config.default import MODEL_PATH, DATA_TRAIN_PATH, DATA_TMP_PATH
import os
import cv2
import face_recognition
import pickle
import time


# Get a reference to webcam #0 (the default one)
videoPath = 0
modelPath = MODEL_PATH
numJitters = 1
faceDetectionModel = ["hog", "cnn"]
threshold = 0.35
numFrame = 0

# load video
videoCapture = cv2.VideoCapture(videoPath)

# load model KNN
logger.info("Loading model ...")
with open(modelPath, "rb") as f:
    knnClf = pickle.load(f)
logger.info("Model is loaded")
logger.info("Using model %s to detect face" % str(faceDetectionModel))

# Coefficient reduce size make algo faster
resize_coef = 4

ret, frame = videoCapture.read()
if not ret:
    "Not found camera!"
else:
    old_frame_shape = frame.shape[:2][::-1]
    new_frame_shape = tuple([i // resize_coef for i in old_frame_shape])
    logger.debug("Resize : %s -> %s" % (old_frame_shape, new_frame_shape))

recog_faces = {}
should_save = True
for face_name in os.listdir(DATA_TRAIN_PATH):
    recog_faces[face_name] = Face(face_name)
    tmp_each_face = os.path.join(DATA_TMP_PATH, face_name)
    if not os.path.exists(tmp_each_face):
        os.makedirs(tmp_each_face)

cnt = 0
max_duration = 0
min_duration = 100
avg_duration = 0
frequency = 100
while True:
    cnt += 1
    ret, frame = videoCapture.read()
    if not ret:
        break
    image = frame.copy()

    # Reduce frame size
    frame = cv2.resize(frame, new_frame_shape)

    numFrame += 1
    prev = time.time()
    if numFrame > 1:
        faceLocations = img_rotate.rotate(
            frame, model=faceDetectionModel[0], rotate_time=0)
        # if len(faceLocations) == 0:
        #     faceLocations = rotate(frame, model=faceDetectionModel[1])
        #     if len(faceLocations) == 0:
        #         logger.info("No face is detected")

        # Reverse face location because of resize
        faceLocations = [
            tuple([location * resize_coef for location in faceLocation]) for faceLocation in faceLocations
        ]

        if len(faceLocations) > 0:
            faceEncodings = face_recognition.face_encodings(
                image, faceLocations, num_jitters=numJitters)
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

            # Consider all faces that recognized
            for name, (top, right, bottom, left) in predictions:
                if name != 'unknown':
                    recog_faces[name].recognize(
                        cnt,
                        (top, right, bottom, left)
                    )
                    face = image[top:bottom, left:right]

            # Look around list faces and consider should save a new face
            for name in recog_faces:
                face = recog_faces[name]
                if should_save and face.should_save():
                    logger.debug("should save: %s" % name)
                    guest_face = nearest_face.find(
                        face.last_locat, predictions)
                    if guest_face:
                        get_face = Face.get_face(image, guest_face[0][1])
                        if img_quality.blur_detection(get_face) > 130:
                            cv2.imshow("face %s" % name, get_face)
                            logger.info("%s face blur: %s" % (
                                face.name,
                                str(img_quality.blur_detection(get_face)))
                            )
                            # Face.save(
                            #     image=get_face,
                            #     path=os.path.join(DATA_TMP_PATH, face.name)
                            # )
                            # logger.info("save %s" % face.name)
                            # new_face = face_recognition.face_encodings(
                            #     get_face, num_jitters=1)
                            # if new_face:
                            #     X = [new_face[0]]
                            #     y = [face.name]
                            #     logger.debug(X)
                            #     logger.debug(y)
                            #     knnClf.fit(X, y)
                face.update()

            # Display the results
            for name, (top, right, bottom, left) in predictions:
                # Draw a box around the face
                cv2.rectangle(
                    image,
                    (left, top),
                    (right, bottom),
                    (0, 0, 255),
                    2
                )

                # Draw a label with a name below the face
                cv2.rectangle(
                    image,
                    (left, bottom - 35),
                    (right, bottom),
                    (0, 0, 255),
                    cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image,
                    name,
                    (left + 6, bottom - 6),
                    font,
                    1.0,
                    (255, 255, 255),
                    1
                )

        # Display the resulting image
        cv2.imshow('Face Recognition', image)
    duration = time.time() - prev
    max_duration = max(max_duration, duration)
    min_duration = min(min_duration, duration)
    avg_duration += duration
    if cnt % frequency == 0:
        logger.info("max: %fs | min: %fs | avg: %fs" % (
            round(max_duration, 4),
            round(min_duration, 4),
            round(avg_duration / frequency, 4)))
        min_duration = 100
        max_duration = 0
        avg_duration = 0

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
