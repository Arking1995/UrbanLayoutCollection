import os
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shapely
from os import listdir
from os.path import isfile, join
import json
import shapely.geometry as sg
import shapely.affinity as sa
# from utils import included_angle, get_Block_azumith, get_RoadAccess_EdgeType, get_BldgRela_EdgeType, Ecu_dis, generate_RoadAccess_EdgeType
import shutil
import matplotlib.pyplot as plt
# from Bldg_fit_func import fit_bldg_features
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from shapely.ops import clip_by_rect
import copy

# plt.subplots(figsize=(20, 4))

def norm_coords(coords, size):
    x_min = np.amin(coords[0, :])
    x_max = np.amax(coords[0, :])
    y_min = np.amin(coords[1, :])
    y_max = np.amax(coords[1, :])
    x_mean = (x_max + x_min) / 2.0
    y_mean = (y_max + y_min) / 2.0

    # print(x_min, x_max, y_min, y_max)
    max_dim = max((x_max - x_min), (y_max - y_min))

    # print(max_dim, (x_max - x_min), (y_max - y_min) )
    coords[0, :] = (size * (coords[0, :] - x_mean) / max_dim) + size / 2.0
    coords[1, :] = size / 2.0 - size * (coords[1, :] - y_mean) / max_dim

    # x_min = np.amin(coords[0, :])
    # x_max = np.amax(coords[0, :])
    # y_min = np.amin(coords[1, :])
    # y_max = np.amax(coords[1, :])
    # print(x_min, x_max, y_min, y_max)

    coords = np.transpose(np.array(coords, dtype=np.int32))
    coords = np.expand_dims(coords, axis=1)
    
    # print(coords.shape, coords[0].shape)
    return coords, max_dim



def k_core(G, k):
    H = nx.Graph(G, as_view=True)
    H.remove_edges_from(nx.selfloop_edges(H))
    core_nodes = nx.k_core(H, k)
    H = H.subgraph(core_nodes)
    return G.subgraph(core_nodes)


def plot2img(fig):
    # remove margins
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # convert to image
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    as_rgba = np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))
    return as_rgba[:,:,:3]


def get_node_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri


def find_nearest_bigger(array, value):
    if value <= array[0]:
        return 0
    if value >= array[array.size-1]:
        return array.size-1

    for idx in range(array.size):
        if value <= array[idx]:
            return idx
        

def find_nearest_smaller(array, value):
    if value <= array[0]:
        return 0

    if value >= array[array.size-1]:
        return array.size-1

    for idx in range(1, array.size):
        if value >= array[idx-1] and value <= array[idx]:
            return idx-1




def get_cross_geometry(offset_x_list, offset_y_list, rot_rect):

    if offset_x_list[0] + offset_x_list[1] > 0.95:
        offset_x_list[1] = 0.95 - offset_x_list[0]
    if offset_x_list[2] + offset_x_list[3] > 0.95:
        offset_x_list[2] = 0.95 - offset_x_list[3]

    if offset_y_list[1] + offset_y_list[2] > 0.95:
        offset_y_list[2] = 0.95 - offset_y_list[1]

    if offset_y_list[3] + offset_y_list[0] > 0.95:
        offset_y_list[3] = 0.95 - offset_y_list[0]

    if rot_rect.geom_type != 'Polygon':
        return rot_rect

    o_pt1 = Point(rot_rect.exterior.coords[0])
    o_pt2 = Point(rot_rect.exterior.coords[1])
    o_pt3 = Point(rot_rect.exterior.coords[2])
    o_pt4 = Point(rot_rect.exterior.coords[3])

    line1 = LineString([o_pt1, o_pt2])  
    line2 = LineString([o_pt1, o_pt4])
    pt1 = line2.interpolate(offset_y_list[0], normalized = True)
    pt3 = line1.interpolate(offset_x_list[0], normalized = True)
    vec_01_3 = np.array([pt3.x - o_pt1.x, pt3.y - o_pt1.y])
    vec_01_1 = np.array([pt1.x - o_pt1.x, pt1.y - o_pt1.y])
    pt2 = Point([o_pt1.x + vec_01_1[0] + vec_01_3[0], o_pt1.y + vec_01_1[1] + vec_01_3[1]])


    line3 = LineString([o_pt2, o_pt1])  
    line4 = LineString([o_pt2, o_pt3])
    pt4 = line3.interpolate(offset_x_list[1], normalized = True)
    pt6 = line4.interpolate(offset_y_list[1], normalized = True)
    vec_02_4 = np.array([pt4.x - o_pt2.x, pt4.y - o_pt2.y])
    vec_02_6 = np.array([pt6.x - o_pt2.x, pt6.y - o_pt2.y])
    pt5 = Point([o_pt2.x + vec_02_4[0] + vec_02_6[0], o_pt2.y + vec_02_4[1] + vec_02_6[1]])


    line5 = LineString([o_pt3, o_pt2])  
    line6 = LineString([o_pt3, o_pt4])
    pt7 = line5.interpolate(offset_y_list[2], normalized = True)
    pt9 = line6.interpolate(offset_x_list[2], normalized = True)
    vec_03_7 = np.array([pt7.x - o_pt3.x, pt7.y - o_pt3.y])
    vec_03_9 = np.array([pt9.x - o_pt3.x, pt9.y - o_pt3.y])
    pt8 = Point([o_pt3.x + vec_03_7[0] + vec_03_9[0], o_pt3.y + vec_03_7[1] + vec_03_9[1]])


    line7 = LineString([o_pt4, o_pt3])  
    line8 = LineString([o_pt4, o_pt1])
    pt10 = line7.interpolate(offset_x_list[3], normalized = True)
    pt12 = line8.interpolate(offset_y_list[3], normalized = True)
    vec_04_10 = np.array([pt10.x - o_pt4.x, pt10.y - o_pt4.y])
    vec_04_12 = np.array([pt12.x - o_pt4.x, pt12.y - o_pt4.y])
    pt11 = Point([o_pt4.x + vec_04_10[0] + vec_04_12[0], o_pt4.y + vec_04_10[1] + vec_04_12[1]])

    output_cross = Polygon([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12])

    return output_cross



def get_L_geometry(offset_x, offset_y, rot_rect):
    rd_id = int(np.random.randint(0, high=4, size=1)[0])

    if rot_rect.geom_type != 'Polygon':
        return rot_rect

    o_pt1 = Point(rot_rect.exterior.coords[0])
    o_pt2 = Point(rot_rect.exterior.coords[1])
    o_pt3 = Point(rot_rect.exterior.coords[2])
    o_pt4 = Point(rot_rect.exterior.coords[3])

    line1 = LineString([o_pt1, o_pt2])  
    line2 = LineString([o_pt1, o_pt4])

    line3 = LineString([o_pt2, o_pt1])  
    line4 = LineString([o_pt2, o_pt3])

    line5 = LineString([o_pt3, o_pt2])  
    line6 = LineString([o_pt3, o_pt4])

    line7 = LineString([o_pt4, o_pt3])  
    line8 = LineString([o_pt4, o_pt1])

    if rd_id == 0:
        pt1 = line2.interpolate(offset_y, normalized = True)
        pt3 = line1.interpolate(offset_x, normalized = True)
        vec_01_3 = np.array([pt3.x - o_pt1.x, pt3.y - o_pt1.y])
        vec_01_1 = np.array([pt1.x - o_pt1.x, pt1.y - o_pt1.y])
        pt2 = Point([o_pt1.x + vec_01_1[0] + vec_01_3[0], o_pt1.y + vec_01_1[1] + vec_01_3[1]])
        output_L = Polygon([pt1, pt2, pt3, o_pt2, o_pt3, o_pt4])

    if rd_id == 1:
        line3 = LineString([o_pt2, o_pt1])  
        line4 = LineString([o_pt2, o_pt3])
        pt4 = line3.interpolate(offset_x, normalized = True)
        pt6 = line4.interpolate(offset_y, normalized = True)
        vec_02_4 = np.array([pt4.x - o_pt2.x, pt4.y - o_pt2.y])
        vec_02_6 = np.array([pt6.x - o_pt2.x, pt6.y - o_pt2.y])
        pt5 = Point([o_pt2.x + vec_02_4[0] + vec_02_6[0], o_pt2.y + vec_02_4[1] + vec_02_6[1]])
        output_L = Polygon([o_pt1, pt4, pt5, pt6, o_pt3, o_pt4])

    if rd_id == 2:
        line5 = LineString([o_pt3, o_pt2])  
        line6 = LineString([o_pt3, o_pt4])
        pt7 = line5.interpolate(offset_y, normalized = True)
        pt9 = line6.interpolate(offset_x, normalized = True)
        vec_03_7 = np.array([pt7.x - o_pt3.x, pt7.y - o_pt3.y])
        vec_03_9 = np.array([pt9.x - o_pt3.x, pt9.y - o_pt3.y])
        pt8 = Point([o_pt3.x + vec_03_7[0] + vec_03_9[0], o_pt3.y + vec_03_7[1] + vec_03_9[1]])
        output_L = Polygon([o_pt1, o_pt2, pt7, pt8, pt9, o_pt4])
    
    if rd_id == 3:
        line7 = LineString([o_pt4, o_pt3])  
        line8 = LineString([o_pt4, o_pt1])
        pt10 = line7.interpolate(offset_x, normalized = True)
        pt12 = line8.interpolate(offset_y, normalized = True)
        vec_04_10 = np.array([pt10.x - o_pt4.x, pt10.y - o_pt4.y])
        vec_04_12 = np.array([pt12.x - o_pt4.x, pt12.y - o_pt4.y])
        pt11 = Point([o_pt4.x + vec_04_10[0] + vec_04_12[0], o_pt4.y + vec_04_10[1] + vec_04_12[1]])
        output_L = Polygon([o_pt1, o_pt2, o_pt3, pt10, pt11, pt12])

    return output_L




def get_U_geometry(offset_x, offset_y, rot_rect):
    if rot_rect.geom_type != 'Polygon':
        return rot_rect
        
    rd_id = int(np.random.randint(0, high=4, size=1)[0])

    o_pt1 = Point(rot_rect.exterior.coords[0])
    o_pt2 = Point(rot_rect.exterior.coords[1])
    o_pt3 = Point(rot_rect.exterior.coords[2])
    o_pt4 = Point(rot_rect.exterior.coords[3])

    line1 = LineString([o_pt1, o_pt2])  
    line2 = LineString([o_pt1, o_pt4])

    line3 = LineString([o_pt2, o_pt1])  
    line4 = LineString([o_pt2, o_pt3])

    line5 = LineString([o_pt3, o_pt2])  
    line6 = LineString([o_pt3, o_pt4])

    line7 = LineString([o_pt4, o_pt3])  
    line8 = LineString([o_pt4, o_pt1])

    if rd_id == 0:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)[0]
        pt1 = Point(line1.interpolate(interp, normalized=True))
        dummy_x = Point(line1.interpolate(offset_x, normalized=True))
        dummy_y = Point(line2.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt1.x, dummy_x.y - o_pt1.y])
        vec_y = np.array([dummy_y.x - o_pt1.x, dummy_y.y - o_pt1.y])
        pt2 = Point((pt1.x + vec_y[0], pt1.y + vec_y[1]))
        pt3 = Point((pt2.x + vec_x[0], pt2.y + vec_x[1]))
        pt4 = Point((pt3.x - vec_y[0], pt3.y - vec_y[1]))
        output_U = Polygon([o_pt1, pt1, pt2, pt3, pt4, o_pt2, o_pt3, o_pt4])


    if rd_id == 1:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)[0]
        pt1 = Point(line4.interpolate(interp, normalized=True))
        dummy_x = Point(line3.interpolate(offset_x, normalized=True))
        dummy_y = Point(line4.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt2.x, dummy_x.y - o_pt2.y])
        vec_y = np.array([dummy_y.x - o_pt2.x, dummy_y.y - o_pt2.y])
        pt2 = Point((pt1.x + vec_x[0], pt1.y + vec_x[1]))
        pt3 = Point((pt2.x + vec_y[0], pt2.y + vec_y[1]))
        pt4 = Point((pt3.x - vec_x[0], pt3.y - vec_x[1]))
        output_U = Polygon([o_pt1, o_pt2, pt1, pt2, pt3, pt4, o_pt3, o_pt4])


    if rd_id == 2:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)[0]
        pt1 = Point(line6.interpolate(interp, normalized=True))
        dummy_x = Point(line6.interpolate(offset_x, normalized=True))
        dummy_y = Point(line5.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt3.x, dummy_x.y - o_pt3.y])
        vec_y = np.array([dummy_y.x - o_pt3.x, dummy_y.y - o_pt3.y])
        pt2 = Point((pt1.x + vec_y[0], pt1.y + vec_y[1]))
        pt3 = Point((pt2.x + vec_x[0], pt2.y + vec_x[1]))
        pt4 = Point((pt3.x - vec_y[0], pt3.y - vec_y[1]))
        output_U = Polygon([o_pt1, o_pt2, o_pt3, pt1, pt2, pt3, pt4, o_pt4])

    if rd_id == 3:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)[0]
        pt1 = Point(line8.interpolate(interp, normalized=True))
        dummy_x = Point(line7.interpolate(offset_x, normalized=True))
        dummy_y = Point(line8.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt4.x, dummy_x.y - o_pt4.y])
        vec_y = np.array([dummy_y.x - o_pt4.x, dummy_y.y - o_pt4.y])
        pt2 = Point((pt1.x + vec_x[0], pt1.y + vec_x[1]))
        pt3 = Point((pt2.x + vec_y[0], pt2.y + vec_y[1]))
        pt4 = Point((pt3.x - vec_x[0], pt3.y - vec_x[1]))
        output_U = Polygon([o_pt1, o_pt2, o_pt3, o_pt4, pt1, pt2, pt3, pt4])

    return output_U






def get_geometry_by_shape_iou(shape_type, iou, bldg):
    rot_rect = bldg.minimum_rotated_rectangle
    if iou <= 0.0:
        iou = 0.1

    if shape_type == 0 or shape_type == 1 or shape_type == 5:  # rectangle
        scale_x = np.random.normal(np.sqrt(iou), 0.02, 1)
        scale_y = iou / scale_x
        out_bldg = sa.scale(rot_rect, xfact = scale_x[0], yfact = scale_y[0])
        if out_bldg.geom_type != 'Polygon':
            return bldg
    

    if shape_type == 2:   # cross shape
        offset_area_list = np.random.random(4)
        offset_area_list = (1.0-iou) * offset_area_list / np.sum(offset_area_list)
        offset_x_list = []
        offset_y_list = []

        for i in range(4):
            rd_asp_rto = np.random.normal(0.5, 0.1, 1)[0] + 1e-3 #np.random.random() + 1e-3
            offset_area = offset_area_list[i]
            offset_x1 = np.sqrt(offset_area / rd_asp_rto)
            offset_y1 = offset_x1 * rd_asp_rto
            offset_x_list.append(offset_x1)
            offset_y_list.append(offset_y1)

        # rd_asp_rto = np.random.normal(0.5, 0.1, 1)[0] + 1e-3
        # offset_area = offset_area_list[0]
        # offset_x1 = np.sqrt(offset_area / rd_asp_rto)
        # offset_y1 = offset_x1 * rd_asp_rto
        # offset_x_list.append(offset_x1)
        # offset_y_list.append(offset_y1)

        # offset_x2 = 1.0
        # while  offset_x2 + offset_x1 > 1.0:        
        #     rd_asp_rto = np.random.random() + 1e-3
        #     offset_area = offset_area_list[1]
        #     offset_x2 = np.sqrt(offset_area / rd_asp_rto)
        #     offset_y2 = offset_x2 * rd_asp_rto
        # offset_x_list.append(offset_x2)
        # offset_y_list.append(offset_y2)

        # offset_y3 = 1.0
        # while offset_y3 + offset_y2 > 1.0: 
        #     rd_asp_rto = np.random.random() + 1e-3
        #     offset_area = offset_area_list[0]
        #     offset_x3 = np.sqrt(offset_area / rd_asp_rto)
        #     offset_y3 = offset_x3 * rd_asp_rto
        # offset_x_list.append(offset_x3)
        # offset_y_list.append(offset_y3)


        # offset_x4 = 1.0
        # offset_y4 = 1.0
        # while offset_x3 + offset_x4 or offset_y4 + offset_y1 > 1.0: 
        #     rd_asp_rto = np.random.random() + 1e-3
        #     offset_area = offset_area_list[0]
        #     offset_x4 = np.sqrt(offset_area / rd_asp_rto)
        #     offset_y4 = offset_x4 * rd_asp_rto
        # offset_x_list.append(offset_x4)
        # offset_y_list.append(offset_y4)
       
        out_bldg = get_cross_geometry(offset_x_list, offset_y_list, rot_rect)

    
    if shape_type == 3:  # L-shape
        offset_area = 1.0 - iou
        rd_asp_rto = np.random.random() + 1e-3
        offset_x = np.sqrt(offset_area / rd_asp_rto)
        offset_y = offset_x * rd_asp_rto
        out_bldg = get_L_geometry(offset_x, offset_y, rot_rect)
    


    if shape_type == 4:  # U-shape
        offset_area = 1.0 - iou
        rd_asp_rto = np.random.normal(0.5, 0.1, 1)[0]
        offset_x = np.sqrt(offset_area / rd_asp_rto)
        offset_y = offset_x * rd_asp_rto
        out_bldg = get_U_geometry(offset_x, offset_y, rot_rect)
    
    return out_bldg  





def bldg_shape_iou_generator(shape_freq, iou_stats, length):
    shape = np.random.choice(np.arange(1, 6), length, p=shape_freq)
    iou = np.random.normal(iou_stats[0], iou_stats[1], length)
    iou[iou>1.0] = 1.0
    return shape, iou



def simpthres_select(block):
    minx, miny, maxx, maxy = block.bounds
    block_shape_portion = np.double(block.area) / np.double(block.minimum_rotated_rectangle.area)

    width = maxx - minx

    if width > 250:
        return 10.0
    if width > 150:
        return (width - 150) / 100.0 * 5.0 + 5.0
    elif block_shape_portion > 0.8:
        return 5.0
    elif width > 50:            
        return (width - 50) / 100.0 * 2.0 + 3.0
    else:
        return 3.0





def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])

def get_azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])
    return az



def get_size_with_vector(mrr):
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        return axis2, axis1, np.array(bbox[0])-np.array(bbox[1]), np.array(bbox[0])-np.array(bbox[3]), [0, 1], [0, 3]
    else:
        return axis1, axis2, np.array(bbox[0])-np.array(bbox[3]), np.array(bbox[0])-np.array(bbox[1]), [0, 3], [0, 1] # longside, shortside    



def get_aspect_ratio(mrr):
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        aspect = np.double(axis1) / np.double(axis2)
    else:
        aspect = np.double(axis2) / np.double(axis1)

    return aspect


def get_size(mrr):
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        return axis2, axis1
    else:
        return axis1, axis2   # longside, shortside

def get_bldggroup_maxarea(bldg):
    maxarea = -9999999999.9
    for i in range(len(bldg)):
        maxarea = bldg[i].area if maxarea < bldg[i].area else maxarea    
    return maxarea


def get_block_parameters(block):
    bbx = block.minimum_rotated_rectangle
    azimuth = get_azimuth(bbx)
    return azimuth, bbx 


def get_bldggroup_parameters(bldg):
    multi_poly = MultiPolygon(bldg)
    bbx = multi_poly.minimum_rotated_rectangle
    azimuth = get_azimuth(bbx)
    return azimuth, bbx 



############ get the distance matrix from 'target' point to 'anchor' x-y matrix. asp_rto is the ratio of y unit compared to x, will be [0, 1]
def dist(anchor, target, asp_rto = 1.0):
    dist_x = np.abs(anchor[:, 0] - target[0])
    dist_y = np.abs(anchor[:, 1] - target[1]) * asp_rto
    dist = np.multiply(dist_x, dist_x) + np.multiply(dist_y, dist_y)
    return dist



############ get the index of smallest element in 'dist' and append it into 'seq' list, if it is not in 'seq' yet. If in, find the second smallest index.
############ input dist matrix is the distance from all possible anchor point to the target point.
def get_anchor_idx(dist, seq): 
    if np.argmin(dist) not in seq:
        return np.argmin(dist)
    else:
        dist[np.argmin(dist)] = np.finfo(dist.dtype).max
        return get_anchor_idx(dist, seq)



def norm_block_to_horizonal(bldg, azimuth, bbx):
    blk_offset_x = np.double(bbx.centroid.x)
    blk_offset_y = np.double(bbx.centroid.y)

    for i in range(len(bldg)):
        curr = sa.translate(bldg[i], -blk_offset_x, -blk_offset_y)
        bldg[i] = sa.rotate(curr, azimuth - 90, origin = (0.0, 0.0))

    return bldg


# def norm_singleblock_to_horizonal(block, azimuth, bbx):
#     blk_offset_x = np.double(bbx.centroid.x)
#     blk_offset_y = np.double(bbx.centroid.y)
#     block = sa.translate(block, -blk_offset_x, -blk_offset_y)
#     block = sa.rotate(block, azimuth - 90, origin = (0.0, 0.0))
#     return block



def norm_geometry_to_array(geometries, coord_scale):
    bldgnum = len(geometries)
    bounds = []
    size = []
    pos = []

    for i in range(bldgnum):
        bounds.append(geometries[i].bounds)
    
    bounds = np.array(bounds, dtype = np.double) # (minx, miny, maxx, maxy)
    minx = np.amin(bounds, axis = 0)[0]
    miny = np.amin(bounds, axis = 0)[1]
    maxx = np.amax(bounds, axis = 0)[2]
    maxy = np.amax(bounds, axis = 0)[3]

    lenx = maxx - minx
    leny = maxy - miny
        
    bounds[:, 0] = ( bounds[:, 0] - minx ) * 2.0 * coord_scale / lenx - coord_scale
    bounds[:, 2] = ( bounds[:, 2] - minx ) * 2.0 * coord_scale / lenx - coord_scale

    bounds[:, 1] = ( bounds[:, 1] - miny ) * 2.0 * coord_scale / leny - coord_scale
    bounds[:, 3] = ( bounds[:, 3] - miny ) * 2.0 * coord_scale / leny - coord_scale

    mx = np.mean( (bounds[:, 0], bounds[:, 2]) , axis = 0)
    my = np.mean( (bounds[:, 1], bounds[:, 3]) , axis = 0)

    
    size = np.stack((bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1]), axis = 1)
    pos = np.stack( (mx, my), axis = 1 )

    if lenx <= leny:   # swap x-y if x-length is shorter than y-length
        size[:, [0, 1]] = size[:, [1, 0]]
        pos[:, [0, 1]] = pos[:, [1, 0]]


    pos_sort = np.lexsort((pos[:,1],pos[:,0])) # The last column is the primary sort key.
    pos_sorted = pos[pos_sort]
    size_sorted = size[pos_sort]
    
    return pos_sorted, size_sorted, pos_sort, [minx, miny, maxx, maxy, lenx, leny]




def combine_small_rows(all_row, pos_x_sort):
    rownum = len(all_row)
    y_thres = 0.2 # 0.6

    if rownum == 1:
        return all_row
    
    if rownum == 2:
        y_thres = 0.1 #0.2

    for i in range(rownum-1):
        # print(i, i, len(all_row[i]) / np.double(len(all_row[i+1])))
        if len(all_row[i]) / np.double(len(all_row[i+1])) < 0.333 or len(all_row[i]) / np.double(len(all_row[i+1])) > 3.333:
            # print(i, pos_x_sort[all_row[i], 1], pos_x_sort[all_row[i+1], 1])
            if np.fabs(np.mean(pos_x_sort[all_row[i], 1]) - np.mean(pos_x_sort[all_row[i+1], 1])) < y_thres:
                all_row[i].extend(all_row[i+1])
                all_row.pop(i+1)
                break

    if len(all_row) == rownum:
        return all_row

    return combine_small_rows(all_row, pos_x_sort)





#########################   recursively remove to return only 2-rows    ##########################
def remove_outlier_row(all_row):
    row_num = []
    for i in range(len(all_row)):
        row_num.append(len(all_row[i]))
    row_num = np.array(row_num)
    min_idx = np.argmin(row_num)

    if len(all_row) == 2:
        if row_num[min_idx] > np.sum(row_num) * 0.1:
            return all_row
        else:
            # all_row[0].extend(all_row[1])
            all_row.pop(min_idx)
            return all_row    

    all_row.pop(min_idx)
    return remove_outlier_row(all_row)







def geoarray_to_dense_2row_cycle(row_assign, pos_sorted, size_sorted, template_width, template_height, bldggroup_asp_rto, bldggroup_longside):
    
    rownum = len(row_assign)
    
    xsort = []
    each_rownum = []
    for i in range(rownum):
        xsort.append(np.sort(row_assign[i]))
        each_rownum.append(len(row_assign[i]))
    
    idx_map = {}

    if rownum == 1:

        
        for i in range(each_rownum[0]):
            if np.mean(pos_sorted[xsort[0], 1]) < 0:      ############ if mean of y-coordinate is lower than 0, use the lower part of grid, otherwise use upper part of the grid.
                idx_map[xsort[0][i]] = i
            else:
                idx_map[xsort[0][i]] = i + template_width
            
                
    elif rownum == 2:
        max_len = max(each_rownum)
        min_len = min(each_rownum)
        maxrowid = np.argmax(each_rownum)
        minrowid = np.argmin(each_rownum)

        if max_len == min_len:
            maxrowid = 0
            minrowid = 1 

        
        is_upsidedown = True if np.mean(pos_sorted[xsort[maxrowid], 1]) >= np.mean(pos_sorted[xsort[minrowid], 1]) else False

        for i in range(max_len):
            if is_upsidedown:
                idx_map[xsort[maxrowid][i]] = template_width + i
            else:
                idx_map[xsort[maxrowid][i]] = i

        anchor_row = pos_sorted[xsort[maxrowid]]
        nearest_pos_seq = []
        for i in range(min_len):
            dist_to_anchori = dist(anchor_row, pos_sorted[xsort[minrowid][i], :], 0.0)
            anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
            nearest_pos_seq.append(anchor_idx)   

        nearest_pos_seq = [xsort[maxrowid][i] for i in nearest_pos_seq]
         
        for i in range(min_len):
            if is_upsidedown:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] - template_width
            else:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] + template_width               

    else:
        print('Error row number inside block, number is: {}'.format(rownum))


    #############  remove left and right Reverse node, which means the node x_sort order is reverse to graph topo row left-right order.
    for i in range(rownum):
        cur_max_graph_idx = 0
        for j in range(len(xsort[i])):
            if cur_max_graph_idx > idx_map[xsort[i][j]]:
                cur_max_graph_idx += 1
                idx_map[xsort[i][j]] = cur_max_graph_idx
            else:
                cur_max_graph_idx = idx_map[xsort[i][j]]



    pos_out = np.zeros((template_width * template_height, 2))
    size_out = np.zeros_like(pos_out)
    exist_out = np.zeros(template_width * template_height)

    for xsort_idx, graph_idx in idx_map.items():
        pos_out[graph_idx] = pos_sorted[xsort_idx, :]
        size_out[graph_idx] = size_sorted[xsort_idx, :]
        exist_out[graph_idx] = 1

    
    g = nx.grid_2d_graph(template_height, template_width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')

    G.graph['aspect_ratio'] = bldggroup_asp_rto
    G.graph['long_side'] = bldggroup_longside
    for i in range(template_height):
        for j in range(template_width):
            idx = i * template_width + j
            G.nodes[idx]['posx'] = pos_out[idx, 0]
            G.nodes[idx]['posy'] = pos_out[idx, 1]
            G.nodes[idx]['exist'] = exist_out[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = size_out[idx, 0]
            G.nodes[idx]['size_y'] = size_out[idx, 1]
    return G








def geoarray_to_dense_grid(row_assign, pos_sorted, size_sorted, template_width, template_height, bldggroup_asp_rto, bldggroup_longside, shape_xsorted, iou_xsorted):
    
    rownum = len(row_assign)
    
    xsort = []
    each_rownum = []
    for i in range(rownum):
        xsort.append(np.sort(row_assign[i]))
        each_rownum.append(len(row_assign[i]))
    
    idx_map = {}

    if rownum == 1:

        for i in range(each_rownum[0]):
            if np.mean(pos_sorted[xsort[0], 1]) < 0:      ############ if mean of y-coordinate is lower than 0, use the lower part of grid, otherwise use upper part of the grid.
                idx_map[xsort[0][i]] = i
            else:
                idx_map[xsort[0][i]] = i + template_width
            
                
    elif rownum == 2:
        max_len = max(each_rownum)
        min_len = min(each_rownum)
        maxrowid = np.argmax(each_rownum)
        minrowid = np.argmin(each_rownum)

        if max_len == min_len:
            maxrowid = 0
            minrowid = 1 

        
        is_upsidedown = True if np.mean(pos_sorted[xsort[maxrowid], 1]) >= np.mean(pos_sorted[xsort[minrowid], 1]) else False

        for i in range(max_len):
            if is_upsidedown:
                idx_map[xsort[maxrowid][i]] = template_width + i
            else:
                idx_map[xsort[maxrowid][i]] = i

        anchor_row = pos_sorted[xsort[maxrowid]]
        nearest_pos_seq = []
        for i in range(min_len):
            dist_to_anchori = dist(anchor_row, pos_sorted[xsort[minrowid][i], :], 0.0)
            anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
            nearest_pos_seq.append(anchor_idx)   

        nearest_pos_seq = [xsort[maxrowid][i] for i in nearest_pos_seq]
         
        for i in range(min_len):
            if is_upsidedown:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] - template_width
            else:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] + template_width               

    else:
        print('Error row number inside block, number is: {}'.format(rownum))


    #############  remove left and right Reverse node, which means the node x_sort order is reverse to graph topo row left-right order.
    for i in range(rownum):
        cur_max_graph_idx = 0
        for j in range(len(xsort[i])):
            if cur_max_graph_idx > idx_map[xsort[i][j]]:
                cur_max_graph_idx += 1
                if cur_max_graph_idx < template_width * template_height: 
                    idx_map[xsort[i][j]] = cur_max_graph_idx
            else:
                cur_max_graph_idx = idx_map[xsort[i][j]]



    pos_out = np.zeros((template_width * template_height, 2))
    size_out = np.zeros_like(pos_out)
    exist_out = np.zeros(template_width * template_height)
    shape_out = np.zeros_like(exist_out)
    iou_out = np.zeros_like(exist_out)

    for xsort_idx, graph_idx in idx_map.items():
        pos_out[graph_idx] = pos_sorted[xsort_idx, :]
        size_out[graph_idx] = size_sorted[xsort_idx, :]
        exist_out[graph_idx] = 1
        shape_out[graph_idx] = shape_xsorted[xsort_idx]
        iou_out[graph_idx] = iou_xsorted[xsort_idx]


    
    g = nx.grid_2d_graph(template_height, template_width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')

    G.graph['aspect_ratio'] = bldggroup_asp_rto
    G.graph['long_side'] = bldggroup_longside
    for i in range(template_height):
        for j in range(template_width):
            idx = i * template_width + j
            G.nodes[idx]['posx'] = pos_out[idx, 0]
            G.nodes[idx]['posy'] = pos_out[idx, 1]
            G.nodes[idx]['exist'] = exist_out[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = size_out[idx, 0]
            G.nodes[idx]['size_y'] = size_out[idx, 1]
            G.nodes[idx]['shape'] = shape_out[idx]
            G.nodes[idx]['iou'] = iou_out[idx]
    return G




def geoarray_to_dense_4row_grid(row_assign, pos_sorted, size_sorted, template_width, template_height, bldggroup_asp_rto, bldggroup_longside, shape_xsorted, iou_xsorted):
    rownum = len(row_assign)    
    xsort = []
    each_rownum = []
    for i in range(rownum):
        xsort.append(np.sort(row_assign[i]))
        each_rownum.append(len(row_assign[i]))
    
    idx_map = {}

    if rownum == 1:

        for i in range(each_rownum[0]):
            if np.mean(pos_sorted[xsort[0], 1]) < 0:      ############ if mean of y-coordinate is lower than 0, use the lower part of grid, otherwise use upper part of the grid.
                idx_map[xsort[0][i]] = i
            else:
                idx_map[xsort[0][i]] = i + template_width
            
                
    elif rownum == 2:
        max_len = max(each_rownum)
        min_len = min(each_rownum)
        maxrowid = np.argmax(each_rownum)
        minrowid = np.argmin(each_rownum)

        if max_len == min_len:
            maxrowid = 0
            minrowid = 1 

        
        is_upsidedown = True if np.mean(pos_sorted[xsort[maxrowid], 1]) >= np.mean(pos_sorted[xsort[minrowid], 1]) else False

        for i in range(max_len):
            if is_upsidedown:
                idx_map[xsort[maxrowid][i]] = template_width + i
            else:
                idx_map[xsort[maxrowid][i]] = i

        anchor_row = pos_sorted[xsort[maxrowid]]
        nearest_pos_seq = []
        for i in range(min_len):
            dist_to_anchori = dist(anchor_row, pos_sorted[xsort[minrowid][i], :], 0.0)
            anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
            nearest_pos_seq.append(anchor_idx)   

        nearest_pos_seq = [xsort[maxrowid][i] for i in nearest_pos_seq]
         
        for i in range(min_len):
            if is_upsidedown:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] - template_width
            else:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] + template_width               

    elif rownum <= 4:
        max_len = max(each_rownum)
        maxrowid = np.argmax(each_rownum)
        # print('maxrow_id: ', maxrowid, rownum)
        
        for i in range(max_len):
            idx_map[xsort[maxrowid][i]] = template_width * maxrowid + i

        anchor_row = pos_sorted[xsort[maxrowid]]
        # print(anchor_row)

        for i in range(rownum):
            if i == maxrowid:
                continue
            nearest_pos_seq = []
            for j in range(each_rownum[i]):
                dist_to_anchori = dist(anchor_row, pos_sorted[xsort[i][j], :], 0.0)
                anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
                nearest_pos_seq.append(anchor_idx)   
            nearest_pos_seq = [xsort[maxrowid][i] for i in nearest_pos_seq]
         
            for j in range(each_rownum[i]):
                idx_map[xsort[i][j]] = idx_map[nearest_pos_seq[j]] + template_width * (i - maxrowid)
    else:
        print('Error row number inside block, number is: {}'.format(rownum))

    #############  remove left and right Reverse node, which means the node x_sort order is reverse to graph topo row left-right order.
    for i in range(rownum):
        cur_max_graph_idx = 0
        for j in range(len(xsort[i])):
            if cur_max_graph_idx > idx_map[xsort[i][j]]:
                cur_max_graph_idx += 1
                if cur_max_graph_idx < template_width * template_height: 
                    idx_map[xsort[i][j]] = cur_max_graph_idx
            else:
                cur_max_graph_idx = idx_map[xsort[i][j]]



    pos_out = np.zeros((template_width * template_height, 2))
    size_out = np.zeros_like(pos_out)
    exist_out = np.zeros(template_width * template_height)
    shape_out = np.zeros_like(exist_out)
    iou_out = np.zeros_like(exist_out)


    for xsort_idx, graph_idx in idx_map.items():
        pos_out[graph_idx] = pos_sorted[xsort_idx, :]
        size_out[graph_idx] = size_sorted[xsort_idx, :]
        exist_out[graph_idx] = 1
        shape_out[graph_idx] = shape_xsorted[xsort_idx]
        iou_out[graph_idx] = iou_xsorted[xsort_idx]


    g = nx.grid_2d_graph(template_height, template_width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')

    G.graph['aspect_ratio'] = bldggroup_asp_rto
    G.graph['long_side'] = bldggroup_longside
    for i in range(template_height):
        for j in range(template_width):
            idx = i * template_width + j
            G.nodes[idx]['posx'] = pos_out[idx, 0]
            G.nodes[idx]['posy'] = pos_out[idx, 1]
            G.nodes[idx]['exist'] = exist_out[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = size_out[idx, 0]
            G.nodes[idx]['size_y'] = size_out[idx, 1]
            G.nodes[idx]['shape'] = shape_out[idx]
            G.nodes[idx]['iou'] = iou_out[idx]
    return G




################################################    add height to 4-row assign               #############################################
def geoarray_to_dense_4row_grid_addheight(row_assign, pos_sorted, size_sorted, template_width, template_height, bldggroup_asp_rto, bldggroup_longside, shape_xsorted, iou_xsorted, height_xsorted):
    rownum = len(row_assign)    
    xsort = []
    each_rownum = []
    for i in range(rownum):
        xsort.append(np.sort(row_assign[i]))
        each_rownum.append(len(row_assign[i]))
    
    idx_map = {}

    if rownum == 1:

        for i in range(each_rownum[0]):
            if np.mean(pos_sorted[xsort[0], 1]) < 0:      ############ if mean of y-coordinate is lower than 0, use the lower part of grid, otherwise use upper part of the grid.
                idx_map[xsort[0][i]] = i
            else:
                idx_map[xsort[0][i]] = i + template_width
            
                
    elif rownum == 2:
        max_len = max(each_rownum)
        min_len = min(each_rownum)
        maxrowid = np.argmax(each_rownum)
        minrowid = np.argmin(each_rownum)

        if max_len == min_len:
            maxrowid = 0
            minrowid = 1 

        
        is_upsidedown = True if np.mean(pos_sorted[xsort[maxrowid], 1]) >= np.mean(pos_sorted[xsort[minrowid], 1]) else False

        for i in range(max_len):
            if is_upsidedown:
                idx_map[xsort[maxrowid][i]] = template_width + i
            else:
                idx_map[xsort[maxrowid][i]] = i

        anchor_row = pos_sorted[xsort[maxrowid]]
        nearest_pos_seq = []
        for i in range(min_len):
            dist_to_anchori = dist(anchor_row, pos_sorted[xsort[minrowid][i], :], 0.0)
            anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
            nearest_pos_seq.append(anchor_idx)   

        nearest_pos_seq = [xsort[maxrowid][i] for i in nearest_pos_seq]
         
        for i in range(min_len):
            if is_upsidedown:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] - template_width
            else:
                idx_map[xsort[minrowid][i]] = idx_map[nearest_pos_seq[i]] + template_width               

    elif rownum ==3 or rownum == 4:
        max_len = max(each_rownum)
        maxrowid = np.argmax(each_rownum)
        # print('maxrow_id: ', maxrowid, rownum)
        
        for i in range(max_len):
            idx_map[xsort[maxrowid][i]] = template_width * maxrowid + i

        anchor_row = pos_sorted[xsort[maxrowid]]
        # print(anchor_row)

        for i in range(rownum):
            if i == maxrowid:
                continue
            nearest_pos_seq = []
            for j in range(each_rownum[i]):
                dist_to_anchori = dist(anchor_row, pos_sorted[xsort[i][j], :], 0.0)
                anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
                nearest_pos_seq.append(anchor_idx)   
            nearest_pos_seq = [xsort[maxrowid][i] for i in nearest_pos_seq]
         
            for j in range(each_rownum[i]):
                idx_map[xsort[i][j]] = idx_map[nearest_pos_seq[j]] + template_width * (i - maxrowid)
    else:
        print('Error row number inside block, number is: {}'.format(rownum))

    #############  remove left and right Reverse node, which means the node x_sort order is reverse to graph topo row left-right order.
    for i in range(rownum):
        cur_max_graph_idx = 0
        for j in range(len(xsort[i])):
            if cur_max_graph_idx > idx_map[xsort[i][j]]:
                cur_max_graph_idx += 1
                if cur_max_graph_idx < template_width * template_height: 
                    idx_map[xsort[i][j]] = cur_max_graph_idx
            else:
                cur_max_graph_idx = idx_map[xsort[i][j]]



    pos_out = np.zeros((template_width * template_height, 2))
    size_out = np.zeros_like(pos_out)
    exist_out = np.zeros(template_width * template_height)
    shape_out = np.zeros_like(exist_out)
    iou_out = np.zeros_like(exist_out)
    height_out = np.zeros_like(exist_out)


    for xsort_idx, graph_idx in idx_map.items():
        pos_out[graph_idx] = pos_sorted[xsort_idx, :]
        size_out[graph_idx] = size_sorted[xsort_idx, :]
        exist_out[graph_idx] = 1
        shape_out[graph_idx] = shape_xsorted[xsort_idx]
        iou_out[graph_idx] = iou_xsorted[xsort_idx]
        height_out[graph_idx] = height_xsorted[xsort_idx]



    g = nx.grid_2d_graph(template_height, template_width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')

    G.graph['aspect_ratio'] = bldggroup_asp_rto
    G.graph['long_side'] = bldggroup_longside
    for i in range(template_height):
        for j in range(template_width):
            idx = i * template_width + j
            G.nodes[idx]['posx'] = pos_out[idx, 0]
            G.nodes[idx]['posy'] = pos_out[idx, 1]
            G.nodes[idx]['exist'] = exist_out[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = size_out[idx, 0]
            G.nodes[idx]['size_y'] = size_out[idx, 1]
            G.nodes[idx]['shape'] = shape_out[idx]
            G.nodes[idx]['iou'] = iou_out[idx]
            G.nodes[idx]['height'] = height_out[idx]

    return G







################################################    handle the strip case               #############################################
def generate_strip_row_assign(pos_sorted, size_sorted):
    pos_y_sort = np.argsort(pos_sorted[:,1])
    bldgnum = pos_sorted.shape[0]
    lx = ly = -10.1
    lw = lh = 0
    cur_row = 0
    cur_max_maxy = -10.1
    cur_min_maxy = 10.1
    ################    all_row: each element stores the index of buildings in i-th row   
    all_row = []
    ################    row: the index of buildings in current row   
    row = []

    ################    initially separate row assigning into small groups  
    for i in range(bldgnum):
        curx, cury = pos_sorted[pos_y_sort[i]]
        curw, curh = size_sorted[pos_y_sort[i]]

        curmaxy = cury + curh / 2.0
        curminy = cury - curh / 2.0

        if len(row) == 0:
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh
            row.append(pos_y_sort[i])
            continue



        if ( (ly + lh / 2.0) <= curminy )  or (cury >= cur_mean_maxy / np.double(len(row)) ):  ## removed condition:   or ( curmaxy - cur_max_maxy > 1e-5  and  curminy - cur_min_maxy > 1e-5 )
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh

            all_row.append(row)
            row = []
            row.append(pos_y_sort[i])            
            continue
        
        cur_mean_maxy += curmaxy

        if cur_max_maxy <= curmaxy:
            cur_max_maxy = curmaxy

        if cur_min_maxy >= curmaxy:
            cur_min_maxy = curmaxy

        row.append(pos_y_sort[i])
        lx, ly, lw, lh = curx, cury, curw, curh

    ################    push the last row
    if len(row) > 0:
        all_row.append(row)

    # print('Original assign: ', all_row)

    ################    combine small row groups into larger and nearby group on y-axis  
    all_row = recursively_combine_rows(all_row, pos_sorted)

    # ################    remove the smallest third row 
    # if len(all_row) > 1:
    #     all_row = remove_outlier_row(all_row)

    # print('Processed assign: ', all_row)            

    ################    rownum: store number of bldgs in each row
    rownum = []
    for i in range(len(all_row)):
        rownum.append(len(all_row[i]))

    if len(all_row) == 0:
        return all_row, rownum, [] 

    ################    all_rowidx: store all idx number that is still existed inside the input pos_sorted array.
    if len(all_row) > 1:
        all_rowidx = all_row[0] + all_row[1]
    else:
        all_rowidx = all_row[0]
    all_rowidx = [int(x) for x in all_rowidx]


    ################    all_row: each element stores the index of buildings in i-th row 
    ################    rownum: store number of bldgs in each row
    ################    store all idx number that is still existed inside the input pos_sorted array.
    return all_row, rownum, all_rowidx   



def recursively_combine_rows(all_row, pos_x_sort, y_thres = 0.6, group_thres = 0.333):
    rownum = len(all_row)

    if rownum == 1:
        return all_row
    
    if rownum == 2:
        y_thres = 0.2

    for i in range(rownum-1):
        # print(i, i, len(all_row[i]) / np.double(len(all_row[i+1])))
        if len(all_row[i]) / np.double(len(all_row[i+1])) <= group_thres or len(all_row[i]) / np.double(len(all_row[i+1])) >= 1.0 / group_thres:
            # print(i, pos_x_sort[all_row[i], 1], pos_x_sort[all_row[i+1], 1])
            if np.fabs(np.mean(pos_x_sort[all_row[i], 1]) - np.mean(pos_x_sort[all_row[i+1], 1])) < y_thres:
                all_row[i].extend(all_row[i+1])
                all_row.pop(i+1)
                break

    if len(all_row) == rownum:
        if len(all_row) > 2:
            return recursively_combine_rows(all_row, pos_x_sort, y_thres = np.inf, group_thres=1.0)
        else:
            return all_row

    return recursively_combine_rows(all_row, pos_x_sort, y_thres)




################################################    handle the default 1-2 row cases               #############################################
def generate_row_assign(pos_sorted, size_sorted):
    pos_y_sort = np.argsort(pos_sorted[:,1])
    bldgnum = pos_sorted.shape[0]
    lx = ly = -100.1
    lw = lh = 0
    cur_row = 0
    cur_max_maxy = -100.1
    cur_min_maxy = 100.1
    ################    all_row: each element stores the index of buildings in i-th row   
    all_row = []
    ################    row: the index of buildings in current row   
    row = []

    ################    initially separate row assigning into small groups  
    for i in range(bldgnum):
        curx, cury = pos_sorted[pos_y_sort[i]]
        curw, curh = size_sorted[pos_y_sort[i]]

        curmaxy = cury + curh / 2.0
        curminy = cury - curh / 2.0

        if len(row) == 0:
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh
            row.append(pos_y_sort[i])
            continue



        if ( (ly + lh / 2.0) <= curminy )  or (cury >= cur_mean_maxy / np.double(len(row)) ):  ## removed condition:   or ( curmaxy - cur_max_maxy > 1e-5  and  curminy - cur_min_maxy > 1e-5 )
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh

            all_row.append(row)
            row = []
            row.append(pos_y_sort[i])            
            continue
        
        cur_mean_maxy += curmaxy

        if cur_max_maxy <= curmaxy:
            cur_max_maxy = curmaxy

        if cur_min_maxy >= curmaxy:
            cur_min_maxy = curmaxy

        row.append(pos_y_sort[i])
        lx, ly, lw, lh = curx, cury, curw, curh

    ################    push the last row
    if len(row) > 0:
        all_row.append(row)

    # print('Original assign: ', all_row)

    ################    combine small row groups into larger and nearby group on y-axis  
    all_row = combine_small_rows(all_row, pos_sorted)

    ################    remove the smallest third row 
    if len(all_row) > 2:
        all_row = remove_outlier_row(all_row)

    # print('Processed assign: ', all_row)            

    ################    rownum: store number of bldgs in each row
    rownum = []
    for i in range(len(all_row)):
        rownum.append(len(all_row[i]))

    ################    all_rowidx: store all idx number that is still existed inside the input pos_sorted array.
    if len(all_row) > 1:
        all_rowidx = all_row[0] + all_row[1]
    else:
        all_rowidx = all_row[0]
    all_rowidx = [int(x) for x in all_rowidx]


    ################    all_row: each element stores the index of buildings in i-th row 
    ################    rownum: store number of bldgs in each row
    ################    store all idx number that is still existed inside the input pos_sorted array.
    return all_row, rownum, all_rowidx   




################################################    handle the default 1-2 row cases               #############################################
def combine_small_4rows(all_row, pos_x_sort):
    rownum = len(all_row)
    y_thres = 0.6

    if rownum == 1:
        return all_row
    
    if rownum == 4:
        y_thres = 0.2

    for i in range(rownum-1):
        # print(i, i, len(all_row[i]) / np.double(len(all_row[i+1])))
        if len(all_row[i]) / np.double(len(all_row[i+1])) < 0.333 or len(all_row[i]) / np.double(len(all_row[i+1])) > 3.333:
            # print(i, pos_x_sort[all_row[i], 1], pos_x_sort[all_row[i+1], 1])
            if np.fabs(np.mean(pos_x_sort[all_row[i], 1]) - np.mean(pos_x_sort[all_row[i+1], 1])) < y_thres:
                all_row[i].extend(all_row[i+1])
                all_row.pop(i+1)
                break

    if len(all_row) == rownum:
        return all_row

    return combine_small_4rows(all_row, pos_x_sort)


#########################   recursively remove to return only 4-rows    ##########################
def remove_outlier_4row(all_row):
    row_num = []
    for i in range(len(all_row)):
        row_num.append(len(all_row[i]))
    row_num = np.array(row_num)
    min_idx = np.argmin(row_num)

    if len(all_row) == 4:
        if row_num[min_idx] > np.sum(row_num) * 0.1:
            return all_row
        else:
            # all_row[0].extend(all_row[1])
            all_row.pop(min_idx)
            return all_row    

    all_row.pop(min_idx)
    return remove_outlier_4row(all_row)



################################################    handle the default 1-2 row cases               #############################################
def generate_4row_assign(pos_sorted, size_sorted, maxlen = None):
    pos_y_sort = np.argsort(pos_sorted[:,1])
    bldgnum = pos_sorted.shape[0]
    lx = ly = -100.1
    lw = lh = 0
    cur_row = 0
    cur_max_maxy = -100.1
    cur_min_maxy = 100.1
    ################    all_row: each element stores the index of buildings in i-th row   
    all_row = []
    ################    row: the index of buildings in current row   
    row = []

    ################    initially separate row assigning into small groups  
    for i in range(bldgnum):
        curx, cury = pos_sorted[pos_y_sort[i]]
        curw, curh = size_sorted[pos_y_sort[i]]

        curmaxy = cury + curh / 2.0
        curminy = cury - curh / 2.0

        if len(row) == 0:
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh
            row.append(pos_y_sort[i])
            continue



        if ( (ly + lh / 2.0) <= curminy )  or (cury >= cur_mean_maxy / np.double(len(row)) ):  ## removed condition:   or ( curmaxy - cur_max_maxy > 1e-5  and  curminy - cur_min_maxy > 1e-5 )
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh

            all_row.append(row)
            row = []
            row.append(pos_y_sort[i])            
            continue
        
        cur_mean_maxy += curmaxy

        if cur_max_maxy <= curmaxy:
            cur_max_maxy = curmaxy

        if cur_min_maxy >= curmaxy:
            cur_min_maxy = curmaxy

        row.append(pos_y_sort[i])
        lx, ly, lw, lh = curx, cury, curw, curh

    ################    push the last row
    if len(row) > 0:
        all_row.append(row)
    else:
        return [], [], []

    # print('Original assign: ', all_row)

    ################    combine small row groups into larger and nearby group on y-axis  
    all_row = combine_small_4rows(all_row, pos_sorted)

    ################    remove the smallest third row 
    if len(all_row) > 4:
        all_row = remove_outlier_4row(all_row)

    # print('Processed assign: ', all_row)            

    ################    rownum: store number of bldgs in each row
    rownum = []
    for i in range(len(all_row)):
        rownum.append(len(all_row[i]))

    ################    all_rowidx: store all idx number that is still existed inside the input pos_sorted array.
    all_rowidx = copy.deepcopy(all_row[0])


    if maxlen is not None:
        for kk in range(len(rownum)):
            cur_maxlen = rownum[kk]
            if cur_maxlen > maxlen:
                all_row[kk] = all_row[kk][:maxlen]
                rownum[kk] = maxlen


    for i in range(1, len(all_row)):
        all_rowidx = all_rowidx + all_row[i]

    all_rowidx = [int(x) for x in all_rowidx]


    ################    all_row: each element stores the index of buildings in i-th row 
    ################    rownum: store number of bldgs in each row
    ################    store all idx number that is still existed inside the input pos_sorted array.
    return all_row, rownum, all_rowidx   









#########################   recursively remove to return no small groups   ##########################
# def remove_small_row(all_row):
#     row_num = []
#     for i in range(len(all_row)):
#         row_num.append(len(all_row[i]))
#     row_num = np.array(row_num)
#     min_idx = np.argmin(row_num)

#     if len(all_row) <= 4:
#         return all_row
#     else:
#         all_row.pop(min_idx)
#         return remove_small_row(all_row) 


#########################   recursively remove to return no small groups   ##########################
def combine_small_multirows(all_row, pos_x_sort):
    rownum = len(all_row)
    y_thres = 0.6

    if rownum == 1:
        return all_row
    
    if rownum == 4:
        y_thres = 0.2

    for i in range(rownum-1):
        # print(i, i, len(all_row[i]) / np.double(len(all_row[i+1])))
        if len(all_row[i]) / np.double(len(all_row[i+1])) < 0.333 or len(all_row[i]) / np.double(len(all_row[i+1])) > 3.333:
            # print(i, pos_x_sort[all_row[i], 1], pos_x_sort[all_row[i+1], 1])
            if np.fabs(np.mean(pos_x_sort[all_row[i], 1]) - np.mean(pos_x_sort[all_row[i+1], 1])) < y_thres:
                all_row[i].extend(all_row[i+1])
                all_row.pop(i+1)
                break

    if len(all_row) == rownum:
        return all_row

    return combine_small_4rows(all_row, pos_x_sort)

################################################    handle the multiple row cases               #############################################
def generate_multi_row_assign(pos_sorted, size_sorted):
    pos_y_sort = np.argsort(pos_sorted[:,1])
    bldgnum = pos_sorted.shape[0]
    lx = ly = -100.1
    lw = lh = 0
    cur_row = 0
    cur_max_maxy = -100.1
    cur_min_maxy = 100.1
    ################    all_row: each element stores the index of buildings in i-th row   
    all_row = []
    ################    row: the index of buildings in current row   
    row = []

    ################    initially separate row assigning into small groups  
    for i in range(bldgnum):
        curx, cury = pos_sorted[pos_y_sort[i]]
        curw, curh = size_sorted[pos_y_sort[i]]

        curmaxy = cury + curh / 2.0
        curminy = cury - curh / 2.0

        if len(row) == 0:
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh
            row.append(pos_y_sort[i])
            continue



        if ( (ly + lh / 2.0) <= curminy )  or (cury >= cur_mean_maxy / np.double(len(row)) ):  ## removed condition:   or ( curmaxy - cur_max_maxy > 1e-5  and  curminy - cur_min_maxy > 1e-5 )
            cur_row = cur_row + 1
            cur_max_maxy = curmaxy
            cur_min_maxy = curmaxy
            cur_mean_maxy = curmaxy
            lx, ly, lw, lh = curx, cury, curw, curh

            all_row.append(row)
            row = []
            row.append(pos_y_sort[i])            
            continue
        
        cur_mean_maxy += curmaxy

        if cur_max_maxy <= curmaxy:
            cur_max_maxy = curmaxy

        if cur_min_maxy >= curmaxy:
            cur_min_maxy = curmaxy

        row.append(pos_y_sort[i])
        lx, ly, lw, lh = curx, cury, curw, curh

    ################    push the last row
    if len(row) > 0:
        all_row.append(row)
    else:
        return [], [], []

    # print('Original assign: ', all_row)

    ################    combine small row groups into larger and nearby group on y-axis  
    all_row = combine_small_rows(all_row, pos_sorted)

    ################    remove the smallest third row 
    if len(all_row) > 4:
        all_row = remove_outlier_4row(all_row)

    # print('Processed assign: ', all_row)            

    ################    rownum: store number of bldgs in each row
    rownum = []
    for i in range(len(all_row)):
        rownum.append(len(all_row[i]))

    ################    all_rowidx: store all idx number that is still existed inside the input pos_sorted array.
    all_rowidx = copy.deepcopy(all_row[0])
    for i in range(1, len(all_row)):
        all_rowidx = all_rowidx + all_row[i]

    all_rowidx = [int(x) for x in all_rowidx]


    ################    all_row: each element stores the index of buildings in i-th row 
    ################    rownum: store number of bldgs in each row
    ################    store all idx number that is still existed inside the input pos_sorted array.
    return all_row, rownum, all_rowidx   







def geoarray_to_anchor_grid(pos_sorted, size_sorted, bldggroup_aspect_ratio, bldggroup_longside, template_width, template_height, coord_scale):

    unit_w = np.double(2 * coord_scale) / np.double(template_width)
    unit_h = np.double(2 * coord_scale) / np.double(template_height)

    w_anchor = np.arange(-coord_scale + unit_w / 2.0, coord_scale + 1e-6, unit_w)
    h_anchor = np.arange(-coord_scale + unit_h / 2.0, coord_scale + 1e-6, unit_h)
    
    anchorw = np.tile(w_anchor, len(h_anchor))
    anchorh = np.repeat(h_anchor, len(w_anchor))
    anchor = np.stack( (anchorw, anchorh), axis = 1)


    bldgnum = pos_sorted.shape[0]
    nearest_pos_seq = []
    for i in range(bldgnum):
        dist_to_anchori = dist(anchor, pos_sorted[i, :])
        anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
        nearest_pos_seq.append(anchor_idx)

    # for i in range(bldgnum):  # check the matching quality between nodes and anchor
    #     print(anchor[nearest_pos_seq[i]], pos_sorted[i])

    pos_out = np.zeros((template_width * template_height, 2))
    size_out = np.zeros_like(pos_out)
    exist_out = np.zeros(template_width * template_height)

    for i in range(bldgnum):
        idx = nearest_pos_seq[i]
        pos_out[idx] = pos_sorted[i, :]
        size_out[idx] = size_sorted[i, :]
        exist_out[idx] = 1
    
    max_node = template_width * template_height
    g = nx.grid_2d_graph(template_height, template_width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    G.graph['aspect_ratio'] = bldggroup_aspect_ratio
    G.graph['long_side'] = bldggroup_longside
    for i in range(template_height):
        for j in range(template_width):
            idx = i * template_width + j
            G.nodes[idx]['posx'] = pos_out[idx, 0]
            G.nodes[idx]['posy'] = pos_out[idx, 1]
            G.nodes[idx]['exist'] = exist_out[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = size_out[idx, 0]
            G.nodes[idx]['size_y'] = size_out[idx, 1]

    return G


def read_filter_polygon(openfile):
    bldg = pickle.load(openfile)
    out = []
    for i in bldg:
        if i.geom_type == 'Polygon' and i.area > 4.0:
            out.append(i)
    return out



def remove_mutual_overlap(pos, size, intersect):
    int_bbx = intersect.bounds
    x_d = int_bbx[2] - int_bbx[0]
    y_d = int_bbx[3] - int_bbx[1]
    cx_d = intersect.centroid.x
    cy_d = intersect.centroid.y

    s_x = size[0]
    s_y = size[1]

    if np.double(x_d) / np.double(s_x) >= np.double(y_d) / (np.double(s_y) + 1e-8):
        if pos[1] >= cy_d:
            pos[1] = pos[1] + y_d / 2.0
        else:
            pos[1] = pos[1] - y_d / 2.0
        size[1] = size[1] - y_d
    else:
        if pos[0] >= cx_d:
            pos[0] = pos[0] + x_d / 2.0
        else:
            pos[0] = pos[0] - x_d / 2.0
        size[0] = size[0] - x_d

    return pos, size



def modify_geometry_overlap(bldg, iou_threshold = 0.5):
    bldgnum = len(bldg)
    rm_list = []
    pos = []
    size = []

    for i in range(bldgnum):
        pos.append([(bldg[i].bounds[0] + bldg[i].bounds[2]) / 2.0, (bldg[i].bounds[1] + bldg[i].bounds[3]) / 2.0])
        size.append([bldg[i].bounds[2] - bldg[i].bounds[0], bldg[i].bounds[3] - bldg[i].bounds[1] ])
    pos = np.array(pos)
    size = np.array(size)


    for i in range(bldgnum):
        for j in range(i+1, bldgnum):
            is_mod = False
            p1 = bldg[i]
            p2 = bldg[j]
            if p1.contains(p2):
                rm_list.append(i)
                continue
            if p2.contains(p1):
                rm_list.append(j)
                continue
            if p1.intersects(p2):
                intersect = p1.intersection(p2)
                int_area = intersect.area
                iou1 = int_area / p1.area
                iou2 = int_area / p2.area

                if iou1 > iou_threshold:
                    rm_list.append(i)
                    continue
                else:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)
                    is_mod = True

                if iou2 > iou_threshold:
                    rm_list.append(j)
                    continue
                elif not is_mod:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)
                
    pos = np.delete(pos, rm_list, axis=0)
    size = np.delete(size, rm_list, axis=0)

    bldg_list = []
    for i in range(pos.shape[0]):
        bldg_list.append(box(pos[i,0] - size[i,0] / 2.0, pos[i,1] - size[i,1] / 2.0, pos[i,0] + size[i,0] / 2.0, pos[i,1] + size[i,1] / 2.0))
    
    return bldg_list




def modify_pos_size_arr_overlap(pos, size, iou_threshold = 0.5, height = None):    # input pos and size np.array, the same as shapely geometry version "modify_geometry_overlap".
    bldgnum = pos.shape[0]
    pos = np.array(pos)
    size = np.array(size)
    rm_list = []
    bldg = []

    for i in range(pos.shape[0]):
        bldg.append(box(pos[i,0] - size[i,0] / 2.0, pos[i,1] - size[i,1] / 2.0, pos[i,0] + size[i,0] / 2.0, pos[i,1] + size[i,1] / 2.0))

    for i in range(bldgnum):
        for j in range(i+1, bldgnum):
            is_mod = False
            p1 = bldg[i]
            p2 = bldg[j]
            if p1.contains(p2):
                rm_list.append(i)
                continue
            if p2.contains(p1):
                rm_list.append(j)
                continue
            if p1.intersects(p2):
                intersect = p1.intersection(p2)
                int_area = intersect.area
                iou1 = int_area / (p1.area + 1e-6)
                iou2 = int_area / (p2.area + 1e-6)

                if iou1 > iou_threshold:
                    rm_list.append(i)
                    continue
                else:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)
                    is_mod = True

                if iou2 > iou_threshold:
                    rm_list.append(j)
                    continue
                elif not is_mod:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)

    for i in range(bldgnum):
       if np.fabs(size[i, 0]) < 1e-2 or np.fabs(size[i, 1]) < 1e-2:
            rm_list.append(i)
                
    pos = np.delete(pos, rm_list, axis=0)
    size = np.delete(size, rm_list, axis=0)

    if height is None:
        return pos, size
    else:
        height = np.delete(height, rm_list, axis=0)
        return pos, size, height




def geometry_envelope(geometries):
    for i in range(len(geometries)):
        geometries[i] = geometries[i].envelope
    return geometries



def get_bldggroup_size_and_asp_rto(bldg_list):
    multi_poly = MultiPolygon(bldg_list)
    bbx = multi_poly.minimum_rotated_rectangle
    aspect_ratio = get_aspect_ratio(bbx)
    longside, shortside = get_size(bbx)
    return aspect_ratio, longside, shortside



def geometry_augment(bldg, cat_len = 3, flip_len = 3, rd_len = 1):   # concatenate will add "cat_len" base situtaion, and flip add "3*cat_len" situtations, random sampling added "rd_len*3*cat_len' situation
    bldg_list = [bldg]
 
    #######################  partial concatenate  ##############################
    if cat_len > 0:
        multi_poly = MultiPolygon(bldg)
        xmin = multi_poly.bounds[0]
        xmax = multi_poly.bounds[2]
        width = xmax - xmin
        int_len = cat_len + 2
        interval = np.linspace(xmin, xmax, int_len)  # (int_len-2) intervals, only on x-axis

        for i in range(1, int_len-1):
            cur_int = interval[i]
            cur_bldg = []
            for i in range(len(bldg)):
                if bldg[i].centroid.x < cur_int:
                    cur_bldg.append(sa.translate(bldg[i], width, 0))
                else:
                    cur_bldg.append(bldg[i])
            bldg_list.append(cur_bldg)
    #################################################################
    ############ Flip  ##############################################
    if flip_len > 0:
        mat_flip1 = [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0]
        mat_flip2 = [-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        for i in range(len(bldg_list)):
            num = len(bldg_list[i])
            cur_a = []
            cur_b = []
            cur_c = []

            if flip_len > 0:
                for j in range(num):
                    cur_a.append(sa.affine_transform(bldg_list[i][j], mat_flip1))
                bldg_list.append(cur_a)

            if flip_len > 1:
                for j in range(num):
                    cur_b.append(sa.affine_transform(bldg_list[i][j], mat_flip2))
                bldg_list.append(cur_b)

            if flip_len > 2:
                for j in range(num):
                    cur_c.append(sa.affine_transform(cur_a[j], mat_flip2))
                bldg_list.append(cur_c)      
    ##########################################################################
    ############ random sample  ##############################################
    if rd_len > 0:
        for i in range(len(bldg_list)):
            for k in range(rd_len):
                curr = []
                for j in range(len(bldg_list[i])):
                    cur_bldg = bldg_list[i][j]
                    b_w = cur_bldg.bounds[2] - cur_bldg.bounds[0]
                    b_h = cur_bldg.bounds[3] - cur_bldg.bounds[1]

                    mat_rd = [( 0.95 + np.random.random_sample() * 0.16), 0, 0, 
                    0, ( 0.95 + np.random.random_sample() * 0.16), 0, 
                    0, 0, 1, 
                    b_w * (-0.1 + np.random.random_sample() * 0.2), b_h * (-0.1 + np.random.random_sample() * 0.2), 0]
                    
                    curr.append(sa.affine_transform(bldg_list[i][j], mat_rd))
                bldg_list.append(curr)
    return bldg_list



def filter_little_intersected_bldglist(bldg, block, thres, idx = None):
    inside = []
    filtered_id = []
    for bi in range(len(bldg)):
        if block.intersects(bldg[bi]):
            try: 
                i_area = block.intersection(bldg[bi]).area
            except:
                i_area = block.buffer(0).intersection(bldg[bi].buffer(0)).area

            portion = i_area / np.double(bldg[bi].area)
            if portion >= thres:
                inside.append(bldg[bi])
                if idx is not None:
                    filtered_id.append(bi)                    
    if idx is not None:
        return inside, idx[filtered_id]
    else:
        return inside


def save_visual_block_bldg(fp, bldg_list, block, c_idx):
    plt.plot(*block.exterior.xy)            
    for kk in range(len(bldg)):
        plt.plot(*bldg[kk].exterior.xy)
    plt.savefig(os.path.join(fp, str(c_idx) + '.png'))
    plt.clf()


def save_visual_bldggroup(fp, bldg_list, c_idx):
    for kk in range(len(bldg_list)):
        plt.plot(*bldg_list[kk].exterior.xy)
    plt.savefig(os.path.join(fp, str(c_idx) + '.png'))
    plt.clf()




def remove_toosmall_bldg(bldg):
    out = []
    for i in bldg:
        if i.area < 10.0:
            continue
        else:
            out.append(i)
    return out



def visual_block_graph(G, filepath, filename, draw_edge = False, draw_nonexist = False):
    plt.clf()
    plt.close()
    pos = []
    size = []
    edge = []
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    if not draw_nonexist:
        for i in range(G.number_of_nodes()):
            if G.nodes[i]['exist'] == 0: # or abs(G.nodes[i]['size_x']) < 1e-2 or abs(G.nodes[i]['size_y']) < 1e-2 s
                G.remove_node(i)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    for i in range(G.number_of_nodes()):
        pos.append([G.nodes[i]['posx'], G.nodes[i]['posy']])
        size.append([G.nodes[i]['size_x'], G.nodes[i]['size_y']])

    for e in G.edges:
        edge.append(e)

    pos = np.array(pos, dtype = np.double)    
    size = np.array(size, dtype = np.double)
    edge = np.array(edge, dtype = np.int16)

    if len(pos) > 0:
        plt.scatter(pos[:, 0], pos[:, 1], c = 'red', s=50)
    ax = plt.gca()
    for i in range(size.shape[0]):
        ax.add_patch(Rectangle((pos[i, 0] - size[i, 0] / 2.0, pos[i, 1] - size[i, 1] / 2.0), size[i, 0], size[i, 1], linewidth=2, edgecolor='r', facecolor='b', alpha=0.3)) 

    if draw_edge:
        for i in range(edge.shape[0]):
            l = mlines.Line2D([pos[edge[i, 0], 0], pos[edge[i, 1], 0]], [pos[edge[i, 0], 1], pos[edge[i, 1], 1]])
            ax.add_line(l)

    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.savefig(os.path.join(filepath,filename + '.png'))
    plt.clf()
    plt.close()



def get_node_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri


def graph_to_vector(g):
    output = {}
    num_nodes = g.number_of_nodes()

    asp_rto = g.graph['aspect_ratio']
    longside = g.graph['long_side']

    posx = get_node_attribute(g, 'posx', np.double)
    posy = get_node_attribute(g, 'posy', np.double)

    size_x = get_node_attribute(g, 'size_x', np.double)
    size_y = get_node_attribute(g, 'size_y', np.double)

    exist = get_node_attribute(g, 'exist', np.int_)

    node_pos = np.stack((posx, posy), 1)
    node_size = np.stack((size_x, size_y), 1)

    shape = get_node_attribute(g, 'shape', np.int16)
    iou = get_node_attribute(g, 'iou', np.double)

    if len(nx.get_node_attributes(g, 'height')) > 0:
        bheight = get_node_attribute(g, 'height', np.double)
        output['n_height'] = bheight

    output['n_size'] = node_size
    output['n_pos'] = node_pos
    output['n_exist'] = exist
    output['g_asp_rto'] = asp_rto
    output['g_longside'] = longside
    output['n_shape'] = shape
    output['n_iou'] = iou

    return output


def get_bbx(pos, size):
    bl = ( pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0)
    br = ( pos[0] + size[0] / 2.0, pos[1] - size[1] / 2.0)
    ur = ( pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0)
    ul = ( pos[0] - size[0] / 2.0, pos[1] + size[1] / 2.0)
    return bl, br, ur, ul




def recover_original_pos_size(pos, size, shortside, longside, coord_scale):
    minx = pos[:, 0] - size[:, 0] / 2.0
    minx = (minx + coord_scale) * longside / (2.0 * coord_scale)

    maxx = pos[:, 0] + size[:, 0] / 2.0
    maxx = (maxx + coord_scale) * longside / (2.0 * coord_scale)

    miny = pos[:, 1] - size[:, 1] / 2.0
    miny = (miny + coord_scale) * shortside / (2.0 * coord_scale)

    maxy = pos[:, 1] + size[:, 1] / 2.0
    maxy = (maxy + coord_scale) * shortside / (2.0 * coord_scale)

    mx = np.mean( (minx, maxx) , axis = 0)
    my = np.mean( (miny, maxy) , axis = 0)

    
    size = np.stack((maxx - minx, maxy - miny), axis = 1)
    pos = np.stack( (mx, my), axis = 1 )

    return pos, size