from knn import train

if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train(
        "data/train",
        model_save_path="data/model/knn.clf",
        verbose=True)
    # TODO: set n_neighbors dynamicaly to the number of people
    print("Training complete!")

    # PREDICTION
    # for image_file in os.listdir("data/train/biden"):
    #     full_file_path = os.path.join("data/train/biden", image_file)
    #     if 'DS_Store' in full_file_path:
    #         continue
    #     print("Looking for faces in {}".format(image_file))

    #     # Find all people in the image using a trained classifier model
    #     predictions = predict(
    #         full_file_path, model_path="more_class.clf")

    #     # Print results on the console
    #     for name, (top, right, bottom, left) in predictions:
    #         print("- Found {} at ({}, {})".format(name, left, top))

    #     Display results overlaid on an image
    #     show_prediction_labels_on_image(os.path.join(
    #         "data/train/phu", image_file), predictions)
