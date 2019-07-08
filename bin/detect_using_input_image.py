import face_recognition
import cv2
import time
import logging

# Get a reference to webcam #0 (the default one)
videoPath = 0
numJitters = 1
threshold = 0.4
count = 0
faceNames = []
isProcessFrame = True
prev = time.time()
numberFrame = 0
# load video
videoCapture = cv2.VideoCapture(videoPath)

# Load a sample picture and learn how to recognize it.
sourceImg = cv2.imread(
    "data/train/ngu/ngu_1.png")

sourceImg_face_encoding = face_recognition.face_encodings(
    sourceImg, num_jitters=numJitters
)[0]

# Create arrays of known face encodings and their names
knownFaceEncodings = [
    sourceImg_face_encoding,
]

known_faceNames = [
    "ngu"
]
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break
    numberFrame += 1
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses)
    # to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if isProcessFrame:
        # Find all the faces and face encodings in the current frame of video
        faceLocations = face_recognition.face_locations(
            rgb_small_frame)
        faceEncodings = face_recognition.face_encodings(
            rgb_small_frame,
            faceLocations,
            num_jitters=numJitters)

        faceNames = []
        for faceEncoding in faceEncodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                knownFaceEncodings, faceEncoding, threshold)
            name = "Unknown"
            logging.info("matches: ", matches)

            # If a match was found in knownFaceEncodings,
            # just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_faceNames[first_match_index]

            faceNames.append(name)

    # isProcessFrame = not isProcessFrame

    # Display the results
    for (top, right, bottom, left), name in zip(
        faceLocations, faceNames
    ):
        # Scale back up face locations since the frame
        # Because we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
    cv2.imshow('Video', frame)
    # cv2.imwrite("output/%s")
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    next_time = time.time() - prev
logging.info("count %s", count)
logging.info("numberFrame %s", numberFrame)
cv2.destroyAllWindows()
