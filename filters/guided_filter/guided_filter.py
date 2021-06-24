import numpy as np
import cv2 as cv
import argparse

def guided_Fitler(I, p, r, eps):
    """
    This is a guided filter implementation depends on opencv
    For more info about guided filter, please refer to
    https://en.wikipedia.org/wiki/Guided_filter
    :parms
    I: guided image
    p: filtering input image
    r: window radius
    eps: regularization
    :return
    filtering output q
    """
    ksize = 2 * r + 1
    I = I.astype(np.float32)
    P = p.astype(np.float32)

    mean_i = cv.blur(I, (ksize, ksize), borderType=cv.BORDER_REFLECT_101)
    mean_p = cv.blur(P, (ksize, ksize), borderType=cv.BORDER_REFLECT_101)
    corr_i = cv.blur(I * I, (ksize, ksize), borderType=cv.BORDER_REFLECT_101)
    corr_ip = cv.blur(I * P, (ksize, ksize), borderType=cv.BORDER_REFLECT_101)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    a = cv.blur(a, (ksize, ksize), borderType=cv.BORDER_REFLECT_101)
    b = cv.blur(b, (ksize, ksize), borderType=cv.BORDER_REFLECT_101)

    q = a * I + b
    return q.astype(np.uint8)


if __name__ == '__main__':
    txt = 'This is an implementation fo guided filter using opencv'
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help = 'path to the image file',required=True)
    parser.add_argument('kernel',help = 'kernel radius',type=int, default=3)
    parser.add_argument('epsilon',help = 'epsilon value',type=int, default=100)

    try:
        args = parser.parse_args()
        img_path = args.file
        r = args.kernel
        eps = args.epsilon
        img = cv.imread(args.file)
        res = guided_Fitler(img, img, r=3, eps=100)
        cv.imshow('input_img', img)
        cv.imshow('res_img', res)
        cv.waitKey(0)
    except Exception as e:
        print(e)
