import numpy as np
import os
import copy
import cv2
import math
from scipy.optimize import minimize
import glob
import numpy as np

vec_length = {
    'rectangle': 4,
    'cross': 12,
    'lshape': 6,
    'ushape': 8,
    'hole' : 8
}


directP_dict = {'north': 0, 'south': 1, 'east': 2,'west': 3}


def get_Block_azumith(loaded_road):
    x_rd, y_rd = loaded_road.exterior.xy
    x = x_rd[:-1]
    y = y_rd[:-1]
    length = len(x)
    miny_id = np.argmin(y)
    pre_miny_id = miny_id - 1
    if pre_miny_id < 0:
        pre_miny_id = length - 1

    dx = np.fabs(x[pre_miny_id] - x[miny_id])
    dy = y[pre_miny_id] - y[miny_id]

    azumith = np.arctan(np.float(dy) / np.float(dx))

    return azumith

def Ecu_dis(x1, y1, x2, y2):
    return np.sqrt( (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2) )




thres_mean_size = 5

def generate_RoadAccess_EdgeType(line_list):
    dict = {}

    line_vol = len(line_list)
    tmp = []
    tmp_idx = -1


    for i in range(line_vol):
        meany = line_list[i].coords[0][1] + line_list[i].coords[1][1]
        dy = line_list[i].coords[0][1] - line_list[i].coords[1][1]
        if np.fabs(dy) < thres_mean_size:
            if meany >= 0:
                dict['north'] = i
            else:
                dict['south'] = i
            continue

    for i in range(line_vol):
        dy = line_list[i].coords[0][1] - line_list[i].coords[1][1]
        if np.fabs(dy) > thres_mean_size:
            if not tmp:
                meanx = line_list[i].coords[0][0] + line_list[i].coords[1][0]
                tmp.append(meanx)
                tmp_idx = i
            else:
                meanx = line_list[i].coords[0][0] + line_list[i].coords[1][0]
                if meanx > tmp[0]:
                    dict['west'] = tmp_idx
                    dict['east'] = i
                else:
                    dict['west'] = i
                    dict['east'] = tmp_idx

    return dict



# def get_RoadAccess_EdgeType(tmp_line, tmp_bldg_centroid):
#     if tmp_line.coords[0][0] >= tmp_bldg_centroid.x and tmp_line.coords[1][0] >= tmp_bldg_centroid.x:
#         return 0.0, 2
#     if tmp_line.coords[0][0] <= tmp_bldg_centroid.x and tmp_line.coords[1][0] <= tmp_bldg_centroid.x:
#         return np.pi, 3
#     if tmp_line.coords[0][1] >= tmp_bldg_centroid.y and tmp_line.coords[1][1] >= tmp_bldg_centroid.y:
#         return 0.5 * np.pi, 0
#     if tmp_line.coords[0][1] <= tmp_bldg_centroid.y and tmp_line.coords[1][1] <= tmp_bldg_centroid.y:
#         return 1.5 * np.pi, 1
#     print("none_type")

def get_RoadAccess_EdgeType(direction):
    if direction == 'north':
        return 0.5 * np.pi, directP_dict['north']
    elif direction == 'south':
        return 1.5 * np.pi, directP_dict['south']
    elif direction == 'east':
        return 0.0, directP_dict['east']
    else:
        return np.pi, directP_dict['west']


def get_BldgRela_EdgeType(strat_bldg_centroid, target_bldg_centroid):
    azumith = included_angle(strat_bldg_centroid.x - 10.0, strat_bldg_centroid.y, strat_bldg_centroid.x, strat_bldg_centroid.y, target_bldg_centroid.x, target_bldg_centroid.y)

    if (azumith >= (7.0 / 4.0) * np.pi and azumith < 2 * np.pi) or (azumith >= 0 and azumith < np.pi / 4.0):
        return azumith, 3
    elif azumith >= (1.0 / 4.0) * np.pi and azumith < (3.0 / 4.0) * np.pi:
        return azumith, 0
    elif azumith >= (3.0 / 4.0) * np.pi and azumith < (5.0 / 4.0) * np.pi:
        return azumith, 2
    elif azumith >= (5.0 / 4.0) * np.pi and azumith < (7.0 / 4.0) * np.pi:
        return azumith, 1
    else:
        print("Nonvalid get_BldgRela_EdgeType")
        return azumith, None


def included_angle(x1, y1, x2, y2, x3, y3):
    dx1 = x1-x2
    dx2 = x3-x2
    dy1 = y1-y2
    dy2 = y3-y2

    cos1 = dx1 * dx2 + dy1 * dy2
    distance = np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2)
    angle = np.arccos(cos1 / (distance + 1e-6))

    return angle



def make_obj(id):
    if id == 0 or id == 'rectangle':
        return make_rectangle
    elif id == 1 or id == 'cross' :
        return make_cross
    elif id == 2 or id == 'lshape':
        return make_lshape
    elif id == 3 or id == 'ushape':
        return make_ushape
    elif id == 4 or id == 'hole':
        return make_hole
    else:
        print('Non existed make shape function.')
        return



def rotate(x0, y0, centerx, centery, theta): #clockwise
    x = (x0 - centerx) * math.cos(theta) - (y0 - centery) * math.sin(theta) + centerx
    y = (x0 - centerx) * math.sin(theta) + (y0 - centery) * math.cos(theta) + centery
    return x, y

def make_rectangle(x):
    centery, centerx, height, width, theta = x
    ulx = centerx + float(width) / 2.0
    uly = centery + float(height) / 2.0

    blx = centerx + float(width) / 2.0
    bly = centery - float(height) / 2.0

    urx = centerx - float(width) / 2.0
    ury = centery + float(height) / 2.0

    brx = centerx - float(width) / 2.0
    bry = centery - float(height) / 2.0

    ulx1, uly1 = rotate(ulx, uly, centerx, centery, theta)
    blx1, bly1 = rotate(blx, bly, centerx, centery, theta)
    urx1, ury1 = rotate(urx, ury, centerx, centery, theta)
    brx1, bry1 = rotate(brx, bry, centerx, centery, theta)

    return [ulx1, uly1, blx1, bly1, brx1, bry1, urx1, ury1]





def make_cross(x):
    centery, centerx, height, width, theta, a, b, c, d, e, f, g, h = x

    if a + c >= width:
        if a > c:
            a = 0
        else:
            c = 0
    if e + g >= width:
        if e > g:
            e = 0
        else:
            g = 0
    if d + f >= height:
        if d > f:
            d = 0
        else:
            f = 0
    if b + h >= height:
        if b > h:
            b = 0
        else:
            h = 0

    if a < 0:
        a = 0
    if b < 0:
        b = 0
    if c < 0:
        c = 0
    if d < 0:
        d = 0
    if e < 0:
        e = 0
    if f < 0:
        f = 0
    if g < 0:
        g = 0
    if h < 0:
        h = 0



    urx = centerx + float(width) / 2.0
    ury = centery + float(height) / 2.0

    brx = centerx + float(width) / 2.0
    bry = centery - float(height) / 2.0

    ulx = centerx - float(width) / 2.0
    uly = centery + float(height) / 2.0

    blx = centerx - float(width) / 2.0
    bly = centery - float(height) / 2.0

    result = [urx - a, ury, urx - a, ury - b, urx, ury - b,
              brx, bry + h, brx - g, bry + h, brx - g, bry,
              blx + e, bly, blx + e, bly + f, blx, bly + f,
              ulx, uly - d, ulx + c, uly - d, ulx + c, uly]

    for i in range(int(len(result)/2)):
        result[2*i], result[2*i+1] = rotate(result[2*i], result[2*i+1], centerx, centery, theta)

    return result



def make_ushape(x):
    centery, centerx, height, width, theta, a, b, c = x

    urx = centerx + float(width) / 2.0
    ury = centery + float(height) / 2.0

    brx = centerx + float(width) / 2.0
    bry = centery - float(height) / 2.0

    ulx = centerx - float(width) / 2.0
    uly = centery + float(height) / 2.0

    blx = centerx - float(width) / 2.0
    bly = centery - float(height) / 2.0


    if a + b >= width:
        a = width / 4
        b = width / 4

    if a + b > width * 4 / 5:
        a = width * 2 / 5
        b = width * 2 / 5

    if c >= height:
        c = height / 2

    if c <= height / 5:
        c = height / 5

    if c < 0:
        c = 0
    if a < 0:
        a = 0
    if b < 0:
        b = 0

    result = [urx, ury, urx - b, ury, urx - b, ury - c,
          ulx + a, uly - c, ulx + a, uly, ulx, uly,
          blx, bly, brx, bry]

    for i in range(8):
        result[2*i], result[2*i+1] = rotate(result[2*i], result[2*i+1], centerx, centery, theta)

    return result



def make_lshape(x):
    centery, centerx, height, width, theta, a, b = x # a: width , b: height of missing part

    urx = centerx + float(width) / 2.0
    ury = centery + float(height) / 2.0

    brx = centerx + float(width) / 2.0
    bry = centery - float(height) / 2.0

    ulx = centerx - float(width) / 2.0
    uly = centery + float(height) / 2.0

    blx = centerx - float(width) / 2.0
    bly = centery - float(height) / 2.0


    if a >= width:
        a = width / 2

    if b >= height:
        b = height / 2

    if a < 0:
        a = 0
    if b < 0:
        b = 0

    result = [urx, ury,
              urx - a, ury,
              urx - a, ury - b,
              ulx, ury - b,
              blx, bly,
              brx, bry]

    for i in range(6):
        result[2*i], result[2*i+1] = rotate(result[2*i], result[2*i+1], centerx, centery, theta)

    return result





def make_hole(x):
    centery, centerx, height, width, theta, a, b, c, d = x
    if a + b >= width:
        a = width / 4
        b = width / 4

    if c + d >= height:
        c = height / 4
        d = height / 4

    if a + b > width * 4 / 5:
        a = width * 2 / 5
        b = width * 2 / 5

    if c + d > width * 4 / 5:
        c = width * 2 / 5
        d = width * 2 / 5


    if d < 0:
        d = 0
    if c < 0:
        c = 0
    if a < 0:
        a = 0
    if b < 0:
        b = 0

    urx = centerx + float(width) / 2.0
    ury = centery + float(height) / 2.0

    brx = centerx + float(width) / 2.0
    bry = centery - float(height) / 2.0

    ulx = centerx - float(width) / 2.0
    uly = centery + float(height) / 2.0

    blx = centerx - float(width) / 2.0
    bly = centery - float(height) / 2.0

    result = [urx, ury, ulx, uly, blx, bly, brx, bry,
              urx - b, ury - d, ulx + a, uly - d,
              blx + a, bly + c, brx - b, bry + c]

    for i in range(int(len(result)/2)):
        result[2*i], result[2*i+1] = rotate(result[2*i], result[2*i+1], centerx, centery, theta)

    return result




def IoU(x0, args):
    img, mode = args
    fit_img = np.zeros(img.shape, np.uint8)

    curr_shape_func = make_obj(mode)
    shape = curr_shape_func(x0)

    fit_contours = np.array(shape).reshape((vec_length[mode],  # number should change for more variant shapes
                                            1, 2)).astype(np.int
                                                          )  # the type should be int, otherwise return "npoints > 0" error
    if mode == 'hole':
        inner_contours = [fit_contours[4:8]]
        out_contours = [fit_contours[0:4]]
        cv2.drawContours(fit_img, out_contours, -1, (255), thickness=-1)
        cv2.drawContours(fit_img, inner_contours, -1, (0), thickness=-1)
    else:
        fit_contours = [fit_contours]
        cv2.drawContours(fit_img, fit_contours, -1, (255), thickness=-1)


    Inter_img = cv2.bitwise_and(fit_img,img)
    Union_img = cv2.bitwise_or(fit_img,img)

    Inter_count = (Inter_img > 127).sum()
    Union_count = (Union_img > 127).sum()

    iou = float(Inter_count)/float(Union_count)

    return (1-iou)

