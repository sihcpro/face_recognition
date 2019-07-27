from face_reg import logger
# import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(
        train_dir,
        model_save_path=None,
        n_neighbors=None,
        knn_algo='ball_tree',
        verbose=False,
        min_image_each=5):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory
        for each known person, with its name

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh
        in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure
        to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    logger.info("Detect and Encoding image....")
    # Loop through each person in the training set
    accept_faces = []
    for class_dir in os.listdir(train_dir):
        user_dir = os.path.join(train_dir, class_dir)
        if not os.path.isdir(user_dir):
            continue
        image_faces = image_files_in_folder(user_dir)

        if len(image_faces) < min_image_each:
            continue
        else:
            logger.info("Train face for: %s" % class_dir)
            accept_faces.append(class_dir)
        # Loop through each training image for the current person
        for img_path in image_faces:
            image = face_recognition.load_image_file(img_path)
            # Add face encoding for current image to the training set
            face_encodings = face_recognition.face_encodings(
                image, num_jitters=10)
            if len(face_encodings) < 1:
                continue
            else:
                X.append(face_encodings[0])
                y.append(class_dir)
    logger.info("Finished encoding ...")
    # Determine how many neighbors to use for weighting in the KNN classifier
    if not n_neighbors:
        n_neighbors = len(accept_faces)
        if verbose:
            logger.info("Chose n_neighbors automatically: %d" % n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.3):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object
        if not specified, model_save_path must be specified
    :param model_path: (optional) path to a pickled knn classifier
        if not specified, model_save_path must be knn_clf
    :param distance_threshold: (optional) distance threshold
        for face classification the larger it is, the more chance
        of mis-classifying an unknown person as a known one
    :return: a list of names and face locations for the recognized faces
        in the image: [(name, bounding box), ...]
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or \
            os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=10)
    # print(closest_distances)
    are_matches = [closest_distances[0][i][0] <=
                   distance_threshold for i in range(len(X_face_locations))]
    # print('are_matches %s' % are_matches)
    # print('closest_distances %s' % closest_distances[0][0][0])
    # Predict classes and remove classifications
    # that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(
        knn_clf.predict(faces_encodings), X_face_locations, are_matches
    )]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")
        print(name)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()
