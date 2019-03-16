import cv2
import numpy as np
import pickle

def __get_template_old_image_mask(width, height):
    ret = np.ones((height, width), np.uint8) * 255
    #ret[:6, :] = 0
    #ret[:40, 900:] = 0
    #ret[:, 934:] = 0
    #ret[:, :12] = 0
    #ret[580:, :40] = 0
    #ret[590:, :] = 0
    #ret[560:, 920:] = 0
    ret[149:401, 42:246] = 0 #kép
    ret[17:44, 570:692] = 0 #azonosító
    ret[192:216,254:490] = 0 # név
    ret[271:325,258:632] = 0 # aláírás
    ret[362:391,254:444] = 0 # dátum
    #ret[465:537,310:736] = 0
    return ret


def __get_template_new_image_mask(width, height):
    ret = np.ones((height, width), np.uint8) * 255
    #ret[:6, :] = 0
    #ret[:36, :34] = 0
    #ret[672:, :31] = 0
    #ret[699:, :] = 0
    #ret[658:, 908:] = 0
    #ret[:, 951:] = 0
    #ret[:39, 917:] = 0
    #ret[:6, :] = 0
    ret[84:373,26:255] = 0 #kép
    ret[338:377,270:471] = 0 #aláírás
    ret[112:133,255:421] = 0 #név
    ret[172:194,310:384] = 0 #nem
    ret[262:288,284:395] = 0 #CAN
    ret[194:257,464:602] = 0 # dátumok
    #ret[220:262,264:655] = 0
    #ret[220:307,264:363] = 0
    #ret[169:209,458:652] = 0
    #ret[189:380,906:942] = 0
    return ret


if __name__ == "__main__":
    img1 = cv2.imread('szemelyi.png') # queryImage
    sift = cv2.xfeatures2d.SIFT_create()
    mask = __get_template_new_image_mask(img1.shape[1], img1.shape[0])
    #cv2.imshow("Masked", cv2.bitwise_and(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), mask))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    kp, des = sift.detectAndCompute(img1, mask)

    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)

    pickle.dump((index, des), open("template_new_camera.id_data", "wb"))

    (kp_array, des_copy) = pickle.load(open("template_new_camera.id_data", "rb"))

    kp_copy = []

    for point in index:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        kp_copy.append(temp)

    print("Finished")
