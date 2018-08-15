import cv2
import numpy as np
import os, fnmatch

from pathlib import Path
import sys
sys.path.append('/home/undead/reps/tf_models/object_detection/datasets/')

import navmii_dataset_utils as navmii_utils
import uuid
import multiprocessing as mp
import time
from itertools import product

def shift(x1, x2, cx, k):
    x3 = x1 + (x2 - x1) * 0.5
    res1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)))
    res3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)))

    return x3, res1, res3

def calc_shift(x1, x2, cx, k):
    thresh = 1.0

    x3, res1, res3 = shift(x1, x2, cx, k)

    while not -thresh < res1 < thresh:
        x3, res1, res3 = shift(x1, x2, cx, k)
        if res3 < 0:
            x1 = x3
        else:
            x2 = x3

    return x1


def getRadialYX(x, y, cx, cy, k, k1, k2, scale, props):
    if scale:
        xshift, yshift, xscale, yscale = props

        x = (x * xscale + xshift)
        y = (y * yscale + yshift)
        ty = y - cy
        tx = x - cx
        r = tx * tx + ty * ty
        r2 = r * r
        r3 = r2 * r
        resulty = (y + k * (ty * r) + k1 * (ty * r2) + k2 * (ty * r3))
        resultx = (x + k * (tx * r) + k1 * (tx * r2) + k2 * (tx * r3))
    else:
        r = (x - cx) * (x - cx) + (y - cy) * (x - cx)
        resulty = (y + k * ((y - cy) * r) + k1 * ((y - cy) * (r * r)) + k2 * ((y - cy) * (r * r * r)))
        resultx = (x + k * ((x - cx) * r) + k1 * ((x - cx) * (r * r)) + k2 * ((x - cx) * (r * r * r)))
    return resulty, resultx


def crop_image(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop


def fisheye(img, Cx, Cy, k, k1, k2, scale, adjust = True, method = cv2.INTER_NEAREST):
    
    props = [0] * 4
    w = img.shape[0]
    h = img.shape[1]
    mapx = np.zeros((w, h), dtype = np.float32)
    mapy = np.zeros((w, h), dtype = np.float32)

    xshift = calc_shift(0, Cx - 1, Cx, k)
    props[0] = xshift
    newcenterx = w - Cx

    xshift2 = calc_shift(0, newcenterx - 1, newcenterx, k)

    yshift = calc_shift(0, Cy - 1, Cy, k)
    props[1] = yshift
    newcentery = w - Cy
    yshift2 = calc_shift(0, newcentery - 1, newcentery, k)

    xscale = (w - xshift - xshift2) / w
    props[2] = xscale
    yscale = (h - yshift - yshift2) / h
    props[3] = yscale

    radial_func =  lambda x, y : getRadialYX(x, y, Cx, Cy, k, k1, k2, scale, props)
    for y, x in product(range(h), range(w)):
        mapy[y][x], mapx[y][x] = radial_func(x, y)
    
    rmpd = cv2.remap(img, mapx, mapy, interpolation = cv2.INTER_NEAREST)

    if adjust:
        rmpd = crop_image(rmpd)

    return rmpd


def get_fisheye(img, f, method = cv2.INTER_NEAREST ):
    # magic constants
    k = 0.5 * 10 ** -2 / f
    k1 = k2 = k * 10 ** -9

    sz = img.shape[:2]
    mx = max(sz[0], sz[1])

    Cx = mx / 2
    Cy = mx / 2

    fisheyed = fisheye(img, Cx, Cy, k, k1, k2, scale = True, adjust = True, method = method)

    return fisheyed


def augment_img(fimg):
    
    random_focal = np.random.uniform(3, 30000)
    img = cv2.imread(fimg)
    shape = img.shape[:2]
    mask = cv2.imread(fimg.replace('.png', '_L.png').replace('.jpg', '_L.png'))
    img = get_fisheye(img, random_focal, method = cv2.INTER_LINEAR)
    mask = get_fisheye(mask, random_focal)

    img = cv2.resize(img, shape, interpolation = cv2.INTER_LINEAR)
    mask = cv2.resize(mask, shape, interpolation = cv2.INTER_NEAREST)

    new_name = str(uuid.uuid1())
    cv2.imwrite(outdir + '/' + new_name + '.jpg', img)
    cv2.imwrite(outdir + '/' + new_name + '_L.png', mask)


if __name__ == '__main__':
    
    #path = '/mnt/Data/Datasets/Segmentation/Cityscapes/remapped'
    outdir = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped_zoomed'
    for path in ['/mnt/Data/Datasets/Segmentation/Cityscapes/remapped', '/mnt/Data/Datasets/Segmentation/Cityscapes/remapped',
                 '/mnt/Data/Datasets/Segmentation/Cityscapes/remapped', '/mnt/Data/Datasets/Segmentation/Cityscapes/remapped'
                 '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped',
                 '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped']
                 #'/mnt/Data/Datasets/Segmentation/Apollo/remapped']:
        print(path, '\n')
        imgs = navmii_utils.GetAllFilesListRecusive(path, ['.png', '.jpg'])
        i = 0
        batch_size = 24
        batch = []
        times = []
        for fimg in imgs:
            if '_L' not in fimg:
                sys.stdout.flush()
                if i > 0 and len(times) > 0:
                    et = ((sum(times) / i) * (len(imgs) - i))
                    eta = '{:.5}'.format(et / 60)
                    oex = '{:.5}'.format(sum(times) / i)
                else:
                    eta = 'unknown'
                    oex = 'unknown'
                sys.stdout.write('\r>> Converting image %d/%d, ETA %s minutes, one example takes %s seconds' % (i, len(imgs), eta, oex))
                sys.stdout.flush()
                batch.append(fimg)
                
                if len(batch) >= batch_size:
                    t = time.time()
                    with mp.Pool(batch_size) as pool:
                        pool.map(augment_img, batch)
                        batch = []
                        i = i + batch_size
                    times.append(time.time() - t)