import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import _utils


def DelFon(name_image, path_image, new_path):
    # global path_image, new_path
    smallestSideSize = 500
    # real would be thicker because of masking process
    mainRectSize = .04
    fgSize = .15

    # path_image = 'dataset-resized/cardboard/'
    # new_path = 'dataset-resized_back/cardboard/'
    img = cv2.imread(path_image + name_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    new_w, new_h = _utils.new_image_size(width, height, smallestSideSize)

    # resize image to lower resources usage
    # if you need masked image in original size, do not resize it
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # quantify colors
    # img_small = _utils.quantify_colors(img_small, 32, 10)

    # create mask tpl
    mask = np.zeros(img_small.shape[:2], np.uint8)

    # create BG rect
    bg_w = round(new_w * mainRectSize)
    bg_h = round(new_h * mainRectSize)
    bg_rect = (bg_w, bg_h, new_w - bg_w, new_h - bg_h)

    # create FG rect
    fg_w = round(new_w * (1 - fgSize) / 2)
    fg_h = round(new_h * (1 - fgSize) / 2)
    fg_rect = (fg_w, fg_h, new_w - fg_w, new_h - fg_h)

    # color: 0 - bg, 1 - fg, 2 - probable bg, 3 - probable fg
    cv2.rectangle(mask, fg_rect[:2], fg_rect[2:4], color=cv2.GC_FGD, thickness=-1)

    mask_preset = mask.copy()

    bgdModel1 = np.zeros((1, 65), np.float64)
    fgdModel1 = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_small, mask, bg_rect, bgdModel1, fgdModel1, 3, cv2.GC_INIT_WITH_RECT)
    mask_rect = mask.copy()

    cv2.rectangle(mask, bg_rect[:2], bg_rect[2:4], color=cv2.GC_PR_BGD, thickness=bg_w * 3)
    cv2.grabCut(img_small, mask, bg_rect, bgdModel1, fgdModel1, 10, cv2.GC_INIT_WITH_MASK)
    mask_mask = mask.copy()

    # mask to remove background
    mask_result = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    # apply mask to image
    masked = cv2.bitwise_and(img_small, img_small, mask=mask_result)
    masked[mask_result < 2] = [255, 255, 255]  # change black bg to blue

    # draw rect on original image
    cv2.rectangle(img_small, bg_rect[:2], bg_rect[2:4], (255, 0, 0), 2)
    cv2.rectangle(img_small, fg_rect[:2], fg_rect[2:4], (0, 255, 0), 2, cv2.FILLED)

    fig, ax = plt.subplots()
    plt.imshow(masked)
    fig.savefig(new_path + name_image, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

'''
path_image = 'dataset-resized/'
new_path = 'dataset-resized_back/'
Num = ['cardboard/', 'glass/', 'metal/', 'paper/', 'plastic/', 'trash/']

for j in Num:
    tree = os.walk(path_image + j)
    for i in tree:
        S = i[2]
        break
    for i in S:
        DelFon(str(i), path_image + j, new_path + j)
        print(str(i), path_image + j, new_path + j)
'''
