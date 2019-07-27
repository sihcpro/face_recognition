# from face_reg import logger
import math


def get_image_center(location):
    return (
        (location[0] + location[2]) / 2.0,
        (location[1] + location[3]) / 2.0,
    )


def distance_between_center(first_center, secon_center):
    x = first_center[0] - secon_center[0]
    y = first_center[1] - secon_center[1]
    return math.sqrt(
        x * x + y * y
    )


def max_allow_distance(first_location, threshold):
    return (
        abs(first_location[2] - first_location[0]) +
        abs(first_location[1] - first_location[3])
    ) * threshold / 2


def find(last_location, new_face_locations, threshold=0.3):
    if not last_location:
        return None
    face_center = get_image_center(last_location)
    max_distance = max_allow_distance(last_location, threshold)
    for other_face in new_face_locations:
        other_face_center = get_image_center(other_face[1])
        # logger.debug("%s: %s -> %s" % (
        #     other_face[0],
        #     str(other_face[1]),
        #     str(other_face_center))
        # )
        distance = distance_between_center(face_center, other_face_center)
        if distance < max_distance:
            return (other_face, distance, max_distance)
