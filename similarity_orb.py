import cv2


orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _orb_sim(descriptor_1, descriptor_2):
    matches = bf.match(descriptor_1, descriptor_2)

    # Look for similar regions with distance < 50.
    # Goes from 0 to 100 so pick a number between
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


def _get_image_keypoints_and_descriptors(image):
    kp, desc = orb.detectAndCompute(image, None)
    return kp, desc


def compare_images_with_orb(image_1, image_2):
    try:
        _, descriptor_1 = _get_image_keypoints_and_descriptors(image_1)
        _, descriptor_2 = _get_image_keypoints_and_descriptors(image_2)

        orb_similarity = _orb_sim(descriptor_1, descriptor_2) * 100

    except Exception as ex:
        print("ERROR in app: While comparing images with orb, error occurred!")
        print(str(ex))
        orb_similarity = 0

    return orb_similarity


