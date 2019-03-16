import connexion
import six
import cv2 as cv
import face_recognition
import numpy as np
import pickle

from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.match_result import MatchResult  # noqa: E501
from swagger_server import util

def index():
    with open('index.html') as file:
        return file.read()

def last_face_picture_get():  # noqa: E501
    with open('face.png', 'rb') as file:
        return file.read()


def last_id_picture_get():
    with open('id_card.png', 'rb') as file:
        return file.read()


(kp_array, old_card_des1) = pickle.load(open("template_old_camera.id_data", "rb"))
old_card_kp1 = []
for point in kp_array:
    temp = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    old_card_kp1.append(temp)

(kp_array, new_card_des1) = pickle.load(open("template_new_camera.id_data", "rb"))
new_card_kp1 = []
for point in kp_array:
    temp = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    new_card_kp1.append(temp)


def read_id_post(image):  # noqa: E501
    result = "NOT_FOUND"

    try:
        nparr = np.fromstring(image.stream.read(), np.uint8)

        MIN_MATCH_COUNT = 10
        input_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        cv.imwrite('input_img.png', input_img)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(input_img, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(new_card_des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([new_card_kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = (388,622)
            input_id_card_img = cv.warpPerspective(input_img, M, (w, h))
            cv.imwrite("id_card.png", input_id_card_img)
            input_id_face_img = cv.cvtColor(input_id_card_img[90:365, 30:220], cv.COLOR_BGR2RGB)

            face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            input_gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(input_gray_img, 1.3, 5)
            selected_face = faces[0]
            for (x, y, w, h) in faces:
                if (w > selected_face[2]):
                    selected_face = (x, y, w, h)

            x_zoom = 1.1
            y_zoom = 1.3
            (x, y, w, h) = selected_face
            selected_face = (int(x - (x_zoom - 1) / 2 * w), int(y - (y_zoom - 1) / 2 * h), int(x_zoom * w), int(y_zoom * h))
            (x, y, w, h) = selected_face
            cv.imwrite("face.png", input_img[y:y + h, x:x + w])
            selected_face_img = cv.cvtColor(input_img[y:y + h, x:x + w], cv.COLOR_BGR2RGB)

            known_face_encoding = face_recognition.face_encodings(input_id_face_img)[0]
            unknown_face_encoding = face_recognition.face_encodings(selected_face_img)[0]

            results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)

            if results[0] == True:
                result = "CORRECT"
            else:
                result = "INCORRECT"
    except:
        print("Error caught!")

    if result != "CORRECT":
        try:
            # print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matches = flann.knnMatch(old_card_des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([old_card_kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w = (428, 721)
                input_id_card_img = cv.warpPerspective(input_img, M, (w, h))
                cv.imwrite("id_card.png", input_id_card_img)
                input_id_face_img = cv.cvtColor(input_id_card_img[162:380, 54:230], cv.COLOR_BGR2RGB)

                face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
                input_gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(input_gray_img, 1.3, 5)
                selected_face = faces[0]
                for (x, y, w, h) in faces:
                    if (w > selected_face[2]):
                        selected_face = (x, y, w, h)

                x_zoom = 1.1
                y_zoom = 1.3
                (x, y, w, h) = selected_face
                selected_face = (
                    int(x - (x_zoom - 1) / 2 * w), int(y - (y_zoom - 1) / 2 * h), int(x_zoom * w), int(y_zoom * h))
                (x, y, w, h) = selected_face
                cv.imwrite("face.png", input_img[y:y + h, x:x + w])
                selected_face_img = cv.cvtColor(input_img[y:y + h, x:x + w], cv.COLOR_BGR2RGB)

                known_face_encoding = face_recognition.face_encodings(input_id_face_img)[0]
                unknown_face_encoding = face_recognition.face_encodings(selected_face_img)[0]

                results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)

                if results[0] == True:
                    result = "CORRECT"
                else:
                    result = "INCORRECT"
        except:
            print("Error caught!")

    return MatchResult(result)
