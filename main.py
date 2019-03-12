import numpy as np
import cv2 as cv
import face_recognition
import operator

if __name__ == '__main__':
    MIN_MATCH_COUNT = 10
    id_template_img = cv.imread('szemelyi.png')          # queryImage
    input_img = cv.imread('kamera.jpg') # trainImage
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(id_template_img, None)
    kp2, des2 = sift.detectAndCompute(input_img, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = id_template_img.shape
        input_id_card_img = cv.warpPerspective(input_img, M, (w, h))
        cv.imwrite("id_card.jpg", input_id_card_img)
        input_id_face_img = cv.cvtColor(input_id_card_img[90:365, 30:220], cv.COLOR_BGR2RGB)

        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        input_gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(input_gray_img, 1.3, 5)
        selected_face = faces[0]
        for (x,y,w,h) in faces:
            if (w > selected_face[2]):
                selected_face = (x,y,w,h)

        x_zoom = 1.1
        y_zoom = 1.3
        (x,y,w,h) = selected_face
        selected_face = (int(x - (x_zoom - 1) / 2 * w), int(y - (y_zoom - 1) / 2 * h), int(x_zoom * w), int(y_zoom * h))
        (x,y,w,h) = selected_face
        cv.imwrite("face.jpg", input_img[y:y + h, x:x + w])
        selected_face_img = cv.cvtColor(input_img[y:y + h, x:x + w], cv.COLOR_BGR2RGB)

        known_face_encoding = face_recognition.face_encodings(input_id_face_img)[0]
        unknown_face_encoding = face_recognition.face_encodings(selected_face_img)[0]

        results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)

        if results[0] == True:
            print("JÓ")
        else:
            print("NEM")

        #lbph = cv.face_LBPHFaceRecognizer.create()
        #lbph.train(face_image)

        #cv.imshow("TEST", selected_face_img)
        #cv.waitKey(0)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None