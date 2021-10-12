import cv2
import numpy as np
import math


# def scale_to_dst_size(img, dst_szie):
#     h, w = img.shape[:2]
#     # rate = dst_szie[0] / max(h, w)
#
#     rate = dst_szie[1] / h
#
#     img_nd = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_AREA)
#     # h, w = img_nd.shape[:2]
#     # pad_x = dst_szie[0] - w
#     # pad_y = dst_szie[1] - h
#     #
#     # img_nd = np.pad(img_nd, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
#
#     # cv2.imwrite('scled_img.jpg', img_nd)
#
#     return img_nd


def scale_to_dst_size(img, dst_szie):
    img_nd = cv2.resize(img, dst_szie, interpolation=cv2.INTER_AREA)

    # cv2.imwrite('scled_img.jpg', img_nd)

    return img_nd