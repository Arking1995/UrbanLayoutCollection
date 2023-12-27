import os, re, itertools
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, MultiLineString
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import shapely
from os import listdir
from os.path import isfile, join
import json
import shapely.geometry as sg
import shapely.affinity as sa
from utils import included_angle, get_Block_azumith, get_RoadAccess_EdgeType, get_BldgRela_EdgeType, Ecu_dis, generate_RoadAccess_EdgeType
import shutil
import matplotlib.pyplot as plt
from Bldg_fit_func import fit_bldg_features
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from graph_util import *
from shapely.ops import nearest_points, linemerge
from skimage.morphology import medial_axis
import collections
import multiprocessing
import skgeom
import warnings
warnings.filterwarnings("ignore")
import random


def get_bldg_features(bldg):

    resolution = 0.3
    x, y = bldg.exterior.xy
    minx = np.amin(x)
    miny = np.amin(y)
    maxx = np.amax(x)
    maxy = np.amax(y)
    
    x = x - minx
    y = y - miny
    
    width = np.double(maxx - minx) / resolution # Width of pixel in 0.3m resolution
    height = np.double(maxy - miny) / resolution # Height of pixel in 0.3m resolution
    
    dpi = 400
    w_inch = width / np.double(dpi)
    h_inch = height / np.double(dpi)
    
    fig = plt.figure(figsize=(w_inch, h_inch), dpi=dpi)
    plt.fill(x, y)
    
    ax = fig.gca()
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    # To remove the huge white borders
    ax.margins(0)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    as_rgba = np.frombuffer(img_as_string, dtype='uint8').reshape((height, width, 4))
    
    img = as_rgba[:, :, :3]
    plt.clf()
    plt.close(fig)

    if img.shape[0] < 2 or img.shape[1] < 2:
        return 0, 1.0, 1.0, 1.0, 0.0
    curr_shape, iou, curr_height, curr_width, theta = fit_bldg_features(img)
    plt.clf()
    plt.close()

    return curr_shape, iou, curr_height, curr_width, theta




def cv2_transfer_coords_to_binary_mask(block, size):
    img = np.zeros((size, size), np.uint8)
    coords = np.array(block.exterior.xy, np.float32)
    coords, max_dim = norm_coords(coords, size)
    cv2.drawContours(img, [coords], -1, (255), -1)
    return img, max_dim



def filter_little_intersected_bldglist(bldg, block):
    inside = []
    if block.is_valid:
        for bi in range(len(bldg)):
            if block.intersects(bldg[bi]):
                portion = np.double(block.intersection(bldg[bi]).area) / np.double(bldg[bi].area)
                if portion >= 0.5:
                    inside.append(bldg[bi])        
    return inside



### return turn or false to decide whether the input block is included in the dataset or not
def check_block_included(block, bldg):
    longside_th = 800
    bldg_th = 120
    longside, _ = get_size(block.minimum_rotated_rectangle)

    ###########    fitler block is too large > 300m
    if longside > longside_th: #longside_thres[h]:
        # print(str(i), ', block too large > {}m.'.format(longside_th))
        return False

    ############    fitler bldg that is not more than halved covered by the block contour 
    bldg = filter_little_intersected_bldglist(bldg, block)
    if len(bldg) == 0:
        # print(str(i), ct_f_idx, ', no building after filtering.')
        return False

    if len(bldg) > bldg_th: #bldgnum_thres[h]:
        # print(str(i), ct_f_idx, ', too much building inside ({} > {}).'.format(len(bldg), bldg_th))
        return False

    # ############    fitler block contour too irregular             
    portion = np.double(block.area) / np.double(block.minimum_rotated_rectangle.area)
    multi_poly = MultiPolygon(bldg)

    bldg_bbx_portion = np.double(multi_poly.area) / np.double(multi_poly.minimum_rotated_rectangle.area)
    bldg_block_portion = np.double(multi_poly.area) / np.double(block.area)
    bldg_convex_portion = np.double(multi_poly.convex_hull.area) / np.double(block.area)


    ############    if building groups in irregular block contour looks good, keep them
    if (bldg_bbx_portion < 0.45 and bldg_convex_portion < 0.35) or bldg_block_portion < 0.25:
        # print(str(i), ct_f_idx, ", bldg_bbx_portion: ", bldg_bbx_portion, ', bldg_block_portion: ', bldg_block_portion, ', bldg_convex_portion: ', bldg_convex_portion)
        return False
    
    if 5 <= len(bldg) < 10:
        if portion < 0.4:
            # print(str(i), ct_f_idx, ', < 10, block shape portion: ', portion)
            return False

        if (bldg_bbx_portion < 0.55 and bldg_convex_portion < 0.45) or bldg_block_portion < 0.35:
            # print(str(i), ct_f_idx, ", < 10, bldg_bbx_portion: ", bldg_bbx_portion, ', bldg_block_portion: ', bldg_block_portion, ', bldg_convex_portion: ', bldg_convex_portion)
            return False

    if len(bldg) < 5:
        if portion < 0.5:
            # print(str(i), ct_f_idx, ', < 10, block shape portion: ', portion)
            return False
    
        if (bldg_bbx_portion < 0.7 and bldg_convex_portion < 0.6) or bldg_block_portion < 0.5:
            # print(str(i), ct_f_idx, ", < 10, bldg_bbx_portion: ", bldg_bbx_portion, ', bldg_block_portion: ', bldg_block_portion, ', bldg_convex_portion: ', bldg_convex_portion)
            return False

    return True



class Vertex:
    def __init__(self, point, degree=0, edges=None):
        self.point = np.asarray(point)
        self.degree = degree
        self.edges = []
        self.visited = False
        if edges is not None:
            self.edges = edges

    def __str__(self):
        return str(self.point)


class Edge:

    def __init__(self, start, end=None, pixels=None):
        self.start = start
        self.end = end
        self.pixels = []
        if pixels is not None:
            self.pixels = pixels
        self.visited = False




def buildTree(img, start=None):
    # copy image since we set visited pixels to black
    img = img.copy()
    shape = img.shape
    nWhitePixels = np.sum(img)

    # neighbor offsets (8 nbors)
    nbPxOff = np.array([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1], [0, 1],
                        [1, -1], [1, 0], [1, 1]
                        ])

    queue = collections.deque()

    # a list of all graphs extracted from the skeleton
    graphs = []

    blackedPixels = 0
    # we build our graph as long as we have not blacked all white pixels!
    while nWhitePixels != blackedPixels:

        # if start not given: determine the first white pixel
        if start is None:
            it = np.nditer(img, flags=['multi_index'])
            while not it[0]:
                it.iternext()

            start = it.multi_index

        startV = Vertex(start)
        queue.append(startV)
        print("Start vertex: ", startV)

        # set start pixel to False (visited)
        img[startV.point[0], startV.point[1]] = False
        blackedPixels += 1

        # create a new graph
        G = nx.Graph()
        G.add_node(startV)

        # build graph in a breath-first manner by adding
        # new nodes to the right and popping handled nodes to the left in queue
        while len(queue):
            currV = queue[0];  # get current vertex
            # print("Current vertex: ", currV)

            # check all neigboor pixels
            for nbOff in nbPxOff:

                # pixel index
                pxIdx = currV.point + nbOff

                if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]):
                    continue;  # current neigbor pixel out of image

                if img[pxIdx[0], pxIdx[1]]:
                    # print( "nb: ", pxIdx, " white ")
                    # pixel is white
                    newV = Vertex([pxIdx[0], pxIdx[1]])

                    # add edge from currV <-> newV
                    G.add_edge(currV, newV, object=Edge(currV, newV))
                    # G.add_edge(newV,currV)

                    # add node newV
                    G.add_node(newV)

                    # push vertex to queue
                    queue.append(newV)

                    # set neighbor pixel to black
                    img[pxIdx[0], pxIdx[1]] = False
                    blackedPixels += 1

            # pop currV
            queue.popleft()
        # end while

        # empty queue
        # current graph is finished ->store it
        graphs.append(G)

        # reset start
        start = None

    # end while

    return graphs, img


def getEndNodes(g):
    # return [n for n in nx.nodes_iter(g) if nx.degree(g, n) == 1]
    return [n for n in list(g.nodes()) if nx.degree(g, n) == 1]



def mergeEdges(graph):
    # copy the graph
    g = graph.copy()

    # v0 -----edge 0--- v1 ----edge 1---- v2
    #        pxL0=[]       pxL1=[]           the pixel lists
    #
    # becomes:
    #
    # v0 -----edge 0--- v1 ----edge 1---- v2
    # |_________________________________|
    #               new edge
    #    pxL = pxL0 + [v.point]  + pxL1      the resulting pixel list on the edge
    #
    # an delete the middle one
    # result:
    #
    # v0 --------- new edge ------------ v2
    #
    # where new edge contains all pixels in between!

    # start not at degree 2 nodes
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]

    for v0 in startNodes:

        # start a line traversal from each neighbor
        startNNbs = nx.neighbors(g, v0)

        startNNbs = list(startNNbs)
        if not len(startNNbs):
            continue

        counter = 0
        v1 = startNNbs[counter]  # next nb of v0
        while True:

            if nx.degree(g, v1) == 2:
                # we have a node which has 2 edges = this is a line segement
                # make new edge from the two neighbors
                nbs = nx.neighbors(g, v1)
                nbs = list(nbs)
                # if the first neihbor is not n, make it so!
                if nbs[0] != v0:
                    nbs.reverse()

                pxL0 = g[v0][v1]["object"].pixels  # the pixel list of the edge 0
                pxL1 = g[v1][nbs[1]]["object"].pixels  # the pixel list of the edge 1

                # fuse the pixel list from right and left and add our pixel n.point
                g.add_edge(v0, nbs[1],
                           object=Edge(v0, nbs[1], pixels=pxL0 + [v1.point] + pxL1)
                           )

                # delete the node n
                g.remove_node(v1)

                # set v1 to new left node
                v1 = nbs[1]

            else:
                counter += 1
                if counter == len(startNNbs):
                    break
                v1 = startNNbs[counter]  # next nb of v0

    # weight the edges according to their number of pixels
    for u, v, o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)

    return g


def getLongestPath(graph, endNodes):
    """
        graph is a fully reachable graph = every node can be reached from every node
    """

    if len(endNodes) < 2:
        raise ValueError("endNodes need to contain at least 2 nodes!")

    # get all shortest paths from each endpoint to another endpoint
    allEndPointsComb = itertools.combinations(endNodes, 2)

    maxLength = 0
    maxPath = None

    for ePoints in allEndPointsComb:

        # get shortest path for these end points pairs
        try:
            sL = nx.dijkstra_path_length(graph,
                                        source=ePoints[0],
                                        target=ePoints[1])
        except:
            continue
        # dijkstra can throw if now path, but we are sure we have a path

        # store maximum
        if (sL > maxLength):
            maxPath = ePoints
            maxLength = sL

    if maxPath is None:
        raise ValueError("No path found!")

    return nx.dijkstra_path(graph,
                            source=maxPath[0],
                            target=maxPath[1]), maxLength



###################################################################################################

def plot2greyimg(fig):
    # remove margins
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)

    # convert to image
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    as_rgba = np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))
    grey = np.sum(as_rgba[:,:,:3], axis = 2) / 3.0
    return grey


def get_extend_line(a, b, block, isfront, is_extend_from_end = False):
    minx, miny, maxx, maxy = block.bounds
    if a.x == b.x:  # vertical line
        if a.y <= b.y:
            extended_line = LineString([a, (a.x, minx)])
        else:
            extended_line = LineString([a, (a.x, maxy)])
    elif a.y == b.y:  # horizonthal line
        if a.x <= b.x:
            extended_line = LineString([a, (minx, a.y)])
        else:
            extended_line = LineString([a, (maxx, a.y)])

    else:
        # linear equation: y = k*x + m
        k = (b.y - a.y) / (b.x - a.x)
        m = a.y - k * a.x
        if k >= 0:
            if b.x - a.x >= 0:
                y1 = k * minx + m
                x1 = (miny - m) / k
                points_on_boundary_lines = [Point(minx, y1), Point(x1, miny)]
            else:
                y1 = k * maxx + m
                x1 = (maxy - m) / k
                points_on_boundary_lines = [Point(maxx, y1), Point(x1, maxy)]        
        else:
            if b.x - a.x >= 0:
                y1 = k * minx + m
                x1 = (maxy - m) / k
                points_on_boundary_lines = [Point(minx, y1), Point(x1, maxy)]
            else:
                y1 = k * maxx + m
                x1 = (miny - m) / k
                points_on_boundary_lines = [Point(maxx, y1), Point(x1, miny)]

        # print('points on bound: ', points_on_boundary_lines[0].coords.xy, points_on_boundary_lines[1].coords.xy)
        
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=a.distance)
        extended_line = LineString([a, Point(points_sorted_by_distance[0])])
    

    min_dis = 9999999999.9
    intersect = block.boundary.intersection(extended_line)
    if intersect.geom_type == 'MultiPoint':
        for i in range(len(intersect.geoms)):
            if intersect.geoms[i].distance(a) <= min_dis:
                nearest_points_on_contour = intersect.geoms[i]
    elif intersect.geom_type == 'Point':
        nearest_points_on_contour = intersect
    # elif intersect.geom_type == 'LineString':
    #     if not is_extend_from_end:
    #         nearest_points_on_contour = a
    #     else:
    #         nearest_points_on_contour = b
    else:
        if not is_extend_from_end:
            nearest_points_on_contour = a
        else:
            nearest_points_on_contour = b
        print('intersect: ', intersect)
        print('unknow geom type on intersection: ', intersect.geom_type)

    if not is_extend_from_end:
        if isfront:
            line_to_contour = LineString([nearest_points_on_contour, a])
        else:
            line_to_contour = LineString([a, nearest_points_on_contour])
    else:
        if isfront:
            line_to_contour = LineString([nearest_points_on_contour, b])
        else:
            line_to_contour = LineString([b, nearest_points_on_contour])

    return line_to_contour


def get_k_angle(a, b):
    if a.x == b.x:
        return np.pi/2.0
    if a.y == b.y:
        return 0.0
    else:
        k = (b.y - a.y) / (b.x - a.x)
        return np.arctan(k)


def heuristic_get_candidate_extended_nodes(coords, front):
    if front:
        p2 = Point(coords[:,1])
        p3 = Point(coords[:,2])
        # for i in range(coords.shape[1] - 3):
        #     k1 = get_k_angle(Point(coords[:,i]), Point(coords[:,i+1]) )
        #     k2 = get_k_angle(Point(coords[:,i+1]), Point(coords[:,i+2]) )
        #     if np.abs(k1 - k2) > np.pi/36.0:  # 5 degree
        #         return Point(coords[:,i+1]), Point(coords[:,i+2])
    else:
        p2 = Point(coords[:,-2])
        p3 = Point(coords[:,-3])
        # for i in reversed(range(1, coords.shape[1] - 2)):
        #     k1 = get_k_angle(Point(coords[:,-i]), Point(coords[:,-(i+1)]) )
        #     k2 = get_k_angle(Point(coords[:,-(i+1)]), Point(coords[:,-(i+2)]) )
        #     if np.abs(k1 - k2) > np.pi/36.0:  # 5 degree
        #         return Point(coords[:,i+1]), Point(coords[:,i])
    return p2, p3



def get_modified_midaxis(midaxis, block):
    minx, miny, maxx, maxy = block.bounds

    coords = np.array(midaxis.coords.xy)
    if coords.shape[1] < 4:
        return midaxis

    p2, p3 = heuristic_get_candidate_extended_nodes(coords, True)
    p_2, p_3 = heuristic_get_candidate_extended_nodes(coords, False)

    # print('p2, p3: ', p2, p3)
    # print('p_2, p_3: ', p_2, p_3)

    extended_line_front = get_extend_line(p2, p3, block, True)
    extended_line_back = get_extend_line(p_2, p_3, block, False)
    # print('extended line: ', extended_line_front, extended_line_back)

    midaxis_miss_bothend = LineString(midaxis.coords[1:-1])
    # print('midaxis without end: ', midaxis_miss_bothend)

    modfied_midaxis = linemerge([extended_line_front, midaxis_miss_bothend, extended_line_back])

    return modfied_midaxis




###############################################################


def get_longest_path(imgSk, thres_point = 1.5):
    xy_idx = np.array(np.where(imgSk == True))
    # x = xy_idx[0][np.int16(xy_idx.shape[1]/ 2.0) ]
    # y = xy_idx[1][np.int16(xy_idx.shape[1]/ 2.0) ]

    x = xy_idx[0][0]
    y = xy_idx[1][0]

    # disconnect
    imgSkT = imgSk.copy()

    graphs, imgB = buildTree(imgSkT, np.array([x, y]))
    print("built %i graphs" % len(graphs))

    # set all connectivity
    for i, g in enumerate(graphs):
        endNodes = getEndNodes(g)
        print("graph %i: %i end nodes" % (i, len(endNodes)))
        graphs[i] = {"graph": g, "endNodes": endNodes}

    if len(endNodes) > 500:
        return False, None
    ###################################

    simpleGraphs = []
    for g in graphs:
        newG = mergeEdges(g["graph"])

        simpleGraphs.append(
            {
                "graph": newG,
                "endNodes": getEndNodes(newG)
            }
        )

    for i, g in enumerate(simpleGraphs):
        try:
            path = getLongestPath(g["graph"], g["endNodes"])
            simpleGraphs[i]["longestPath"] = path
        except:
            return False, None
    #########################################

    for i, g in enumerate(simpleGraphs):
        graph = g["graph"]

        longestPathNodes = g["longestPath"][0]
        longestPathEdges = [(longestPathNodes[i], longestPathNodes[i + 1]) for i in
                            range(0, len(longestPathNodes) - 1)]
        # loop over edges and plot pixels inbetween
        edge_pix = []

        for i in range(len(longestPathEdges)):
            e = longestPathEdges[i]
            px = np.array(graph[e[0]][e[1]]["object"].pixels)
            if px.shape[0] > 0:
                if np.abs(e[0].point[0] - px[0,0]) < thres_point and np.abs(e[0].point[1] - px[0,1]) < thres_point:
                    edge_pix.extend(px)
                else:
                    edge_pix.extend(np.flipud(px))

    edge_swap = np.array(edge_pix)
    edge_swap[:, [0, 1]] = edge_swap[:, [1, 0]]

    return True, edge_swap





###############################################################
def get_strip_coords(bldg, block, alpha_strip_map, exterior_vertices, exterior_vtx_order_by_key, mean_width):
    bldgnum = len(bldg)
    bldg_support = {}
    bldg_coords = {}
    exterior_edge_length = {}
    bldg_size = {}
    total_length = block.length
    x_edge_dict = {}
    bldg_intersect = {}


    for i in alpha_strip_map:
        start_edge_pt = exterior_vertices[i]
        end_edge_pt_id = exterior_vtx_order_by_key.index(i) + 1 if exterior_vtx_order_by_key.index(i) != len(exterior_vtx_order_by_key) - 1 else 0
        end_edge_pt = exterior_vertices[exterior_vtx_order_by_key[end_edge_pt_id]]
        x_edge = LineString([(start_edge_pt.x, start_edge_pt.y), (end_edge_pt.x, end_edge_pt.y)])
        x_edge_dict[i] = x_edge

        for j in range(bldgnum):
            if alpha_strip_map[i].intersects(bldg[j]):
            # if alpha_strip_map[i].contains(bldg[j].centroid):
                if i not in bldg_intersect:
                    bldg_intersect[i] = [j]
                else:
                    bldg_intersect[i].append(j)

    # bldg_support = bldg_intersect
    for i in alpha_strip_map:  # all strips

        if i in bldg_intersect:  # strip that intersects bldg
            x_edge = x_edge_dict[i]

            for j in bldg_intersect[i]: # traverse all intersected bldg
                cur_x = x_edge.project(bldg[j].centroid)

                if alpha_strip_map[i].contains(bldg[j].centroid):
                    if 0.001 < np.fabs(cur_x) < (x_edge.length - 0.001):  # perfect match
                        if i not in bldg_support:
                            bldg_support[i] = [j]
                        else:
                            bldg_support[i].append(j)
                    else:
                        is_find = False
                        # print('cur_x: ',j, cur_x, x_edge.length, bldg_support)
                        for k in bldg_intersect:   # not perfect match, first search all intersected strips
                            if j in bldg_intersect[k]:
                                k_x_edge = x_edge_dict[k]
                                k_cur_x = k_x_edge.project(bldg[j].centroid)
                                if 0.001 < np.fabs(k_cur_x) < (k_x_edge.length - 0.001):
                                    if k not in bldg_support:
                                        bldg_support[k] = [j]
                                    else:
                                        bldg_support[k].append(j)
                                    is_find = True
                                    break
                        if not is_find: # not found, search all possible strips
                            for kk in alpha_strip_map:
                                kk_x_edge = x_edge_dict[kk]
                                kk_cur_x = kk_x_edge.project(bldg[j].centroid)
                                if 0.001 < np.fabs(kk_cur_x) < (kk_x_edge.length - 0.001):
                                    if kk not in bldg_support:
                                        bldg_support[kk] = [j]
                                    else:
                                        bldg_support[kk].append(j)
                                    is_find = True
                                    break                               
    # print(bldg_intersect)
    # print(bldg_support)

    # print('mean_width: ', mean_width )
    # print(bldg_support)
    # print(alpha_strip_map)

    for i in alpha_strip_map:
        start_edge_pt = exterior_vertices[i]
        end_edge_pt_id = exterior_vtx_order_by_key.index(i) + 1 if exterior_vtx_order_by_key.index(i) != len(exterior_vtx_order_by_key) - 1 else 0
        end_edge_pt = exterior_vertices[exterior_vtx_order_by_key[end_edge_pt_id]]

        x_edge = LineString([(start_edge_pt.x, start_edge_pt.y), (end_edge_pt.x, end_edge_pt.y)])
        vector_x_edge = np.array([end_edge_pt.x - start_edge_pt.x, end_edge_pt.y- start_edge_pt.y])
        exterior_edge_length[i] = x_edge.length
        # print('edge: ', i, exterior_vtx_order_by_key[end_edge_pt_id] ,x_edge, x_edge.length, vector_x_edge)

        if i in bldg_support:
            for j in bldg_support[i]:
                # print(i, j, x_edge, bldg[j])
                cur_x = x_edge.project(bldg[j].centroid)
                # print('pred_cur_x: ', cur_x)
                cur_y = bldg[j].centroid.distance(x_edge)
                bldg_coords[j] = [cur_x, cur_y]

                longside, shortside, long_vec, short_vec, _, _ = get_size_with_vector(bldg[j].minimum_rotated_rectangle)
                long_vec = long_vec / np.linalg.norm(long_vec)
                unit_x_edge =  vector_x_edge / np.linalg.norm(vector_x_edge)

                long_angle = np.arccos(np.dot(long_vec, unit_x_edge ))
                long_angle = np.min([np.pi - long_angle, long_angle])

                short_vec = short_vec / np.linalg.norm(short_vec)
                short_angle = np.arccos(np.dot(short_vec, unit_x_edge ))
                short_angle = np.min([np.pi - short_angle, short_angle])

                # print('long_short_angle: ', long_vec, short_vec, unit_x_edge, long_angle, short_angle)

                if short_angle < long_angle:
                    currr = shortside
                    shortside = longside
                    longside = currr
                
                bldg_size[j] = [longside / total_length, shortside / mean_width]


    accum_length = 0.0
    # print(total_length)
    # print(bldg_coords, accum_length)
    
    for i in range(len(exterior_vtx_order_by_key)):
        if exterior_vtx_order_by_key[i] in bldg_support:
            if i == 0 :
                for j in bldg_support[exterior_vtx_order_by_key[i]]:
                    bldg_coords[j][0] = bldg_coords[j][0] / total_length
                    bldg_coords[j][1] = bldg_coords[j][1] / mean_width
            else:
                for j in bldg_support[exterior_vtx_order_by_key[i]]:
                    bldg_coords[j][0] = (bldg_coords[j][0] + accum_length) / total_length
                    bldg_coords[j][1] = bldg_coords[j][1] / mean_width
        accum_length += exterior_edge_length[exterior_vtx_order_by_key[i]]

    
    return bldg_coords, bldg_size


#####################################################################
###############################################################
def get_real_coords_from_strip(block, pos_sorted, size_sorted, exterior_vertices, exterior_vtx_order_by_key, mean_width):
    bldgnum = pos_sorted.shape[0] 
    
    org_size = np.zeros_like(pos_sorted)
    org_pos = np.zeros_like(pos_sorted)

    org_size[:, 0] = size_sorted[:, 0] * block.length
    org_size[:, 1] = size_sorted[:, 1] * mean_width

    polyline_list = []
    for i in exterior_vtx_order_by_key:
        pt = exterior_vertices[i]
        polyline_list.append([pt.x, pt.y])
    polyline_list.append([exterior_vertices[exterior_vtx_order_by_key[0]].x, exterior_vertices[exterior_vtx_order_by_key[0]].y])
    midaxis = LineString(polyline_list)
    # print('midaxis: ', midaxis)

    relative_cutoff = [0.0]
    vector_midaxis = []
    coords = np.array(midaxis.coords.xy)
    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])
    relative_cutoff = np.array(relative_cutoff)
    relative_cutoff[-1] = 1.0
    vector_midaxis = np.array(vector_midaxis)
    # print('relative_cutoff: ', relative_cutoff)
    # print('vector_midaxis: ', vector_midaxis)

    bldgnum = pos_sorted.shape[0]        
    corres_midaxis_vector_idx = []
    for i in range(bldgnum):
        cur_x = pos_sorted[i, 0]
        insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
        # print('cur_x: ', cur_x, insert_pos)
        if insert_pos > vector_midaxis.shape[0]-1:
            print('\n out of index in vector_midaxis. \n')
            corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
            continue
        corres_midaxis_vector_idx.append(insert_pos)
    corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)

    ###############################################################################   get correct position and size   ###################
    for i in range(bldgnum):
        vertical_point_on_midaxis = midaxis.interpolate(pos_sorted[i, 0], normalized=True)
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        if pos_sorted[i, 1] <= 0:
            vec_from_midaxis_to_bldg = np.array([cur_vector_midaxis[1], -cur_vector_midaxis[0]])
        else:
            vec_from_midaxis_to_bldg = np.array([-cur_vector_midaxis[1], cur_vector_midaxis[0]])
        
        vec_from_midaxis_to_bldg = vec_from_midaxis_to_bldg / np.linalg.norm(vec_from_midaxis_to_bldg)

        cur_pos_x = vertical_point_on_midaxis.x + vec_from_midaxis_to_bldg[0] * np.abs(pos_sorted[i, 1]) * (mean_width)
        cur_pos_y = vertical_point_on_midaxis.y + vec_from_midaxis_to_bldg[1] * np.abs(pos_sorted[i, 1]) * (mean_width)

        org_pos[i, 0], org_pos[i, 1] = cur_pos_x, cur_pos_y ##   changed from multiply by "line_from_midaxis_to_contour.length"
    ###############################################################################   get correct position and size   ###################
    # org_pos, org_size = modify_pos_size_arr_overlap(org_pos, org_size)

    ###############################################################################   get original rotation  ###################
    org_bldg = []
    for i in range(org_pos.shape[0]):
        curr_bldg = Polygon(get_bbx(org_pos[i,:], org_size[i,:]))
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        angle = np.arctan2(cur_vector_midaxis[1], cur_vector_midaxis[0]) * 180.0 / np.pi
        curr_bldg = sa.rotate(curr_bldg, angle, origin=(org_pos[i,0], org_pos[i,1]))
        org_bldg.append(curr_bldg)
    
    return org_bldg, org_pos, org_size

        






###############################################################
def warp_irregular_blk(block_in, scale = 10.0, simp_thres = 1.5):
    thres_point = 2.5
    recover_scale = 100.0 / scale
    block = block_in  # block_in.simplify(0.4)  #block_in 

    tile_width = block.bounds[2] - block.bounds[0]
    tile_height = block.bounds[3] - block.bounds[1]
    fig1, ax1 = plt.subplots(figsize=(tile_width / scale, tile_height/ scale))
    ax1.set_facecolor([0.0, 0.0, 0.0])

    plt.xlim(block.bounds[0], block.bounds[2])
    plt.ylim(block.bounds[1], block.bounds[3])
    ax1.fill(*block.exterior.xy, color = 'white') 

    img = plot2greyimg(fig1)    
    plt.clf()
    plt.close()
    
    blk, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    offset_cnt = blk[0]

    # cv2.imwrite(os.path.join(transvis_fp, ct_f_idx + '_q.png'), img)
    imgSk, distance = medial_axis(img, return_distance=True)
    # cv2.imwrite(os.path.join(transvis_fp, ct_f_idx + '_s.png'), imgSk.astype(np.uint8) * 255.0)
    

    is_success, edge_swap = get_longest_path(imgSk, thres_point)
    if not is_success:
        return is_success, None

    # im_skt1 = np.zeros(img.shape)
    # cv2.polylines(im_skt1, [np.int32(edge_swap)], False, (255), thickness=1, lineType=8, shift=0)
    # cv2.drawContours(im_skt1, [offset_cnt], 0, (255), 1)
    # cv2.imwrite(os.path.join(rawvis_fp, ct_f_idx + '_skt.png'), im_skt1)

    mid_contour = np.array(edge_swap.copy(), dtype = np.double)
    mid_contour[:, 1] = (tile_height * recover_scale - mid_contour[:, 1]) / recover_scale + block.bounds[1]
    mid_contour[:, 0] = mid_contour[:, 0] / recover_scale + block.bounds[0]

    midaxis = LineString(mid_contour)
    midaxis = midaxis.simplify(simp_thres)

    modfied_midaxis = get_modified_midaxis(midaxis, block)

    if np.array(modfied_midaxis.coords.xy)[0, 0] > np.array(modfied_midaxis.coords.xy)[0, -1]:
        modfied_midaxis = LineString(modfied_midaxis.coords[::-1])

    return True, modfied_midaxis

# ###############################################################

def is_on_contour(polygon, pt):
    for i in polygon.coords:
        if np.linalg.norm(i - np.array((pt.x(), pt.y()), dtype=np.float)) < 0.01:
            return True
    return False

def skgeom_dist(p1, p2):
    return np.sqrt((float(p1.x()) - float(p2.x())) **2 + (float(p1.y()) - float(p2.y())) **2)

def get_polyskeleton_longest_path(skeleton, polygon):
    # skgeom.draw.draw(polygon)
    interior_skel_vertices = {}
    interior_skel_time = {}
    exterior_vertices = {}
    connect_dict = {}

    G = nx.Graph() 
    end_nodes = []

    for v in skeleton.vertices:
        if v.time > 0.001:
            interior_skel_vertices[v.id] = Point(v.point.x(), v.point.y())
            interior_skel_time[v.id] = v.time
        else:
            exterior_vertices[v.id] = Point(v.point.x(), v.point.y())
            end_nodes.append(v.id)
        G.add_node(v.id, posx = float(v.point.x()), posy = float(v.point.y()) )
    
    # for i in exterior_vertices:
    #     print(exterior_vertices[i])

    # for i in interior_skel_vertices:
    #     print(interior_skel_vertices[i])


    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point
            # plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=2)
            if not (is_on_contour(polygon, p1) and is_on_contour(polygon, p2)):
                G.add_edge(h.vertex.id, h.opposite.vertex.id, weight = skgeom_dist(p1, p2))

    
    path, length = getLongestPath(G, end_nodes)
    longest_skel = []
    for i in path:
        longest_skel.append((G.nodes[i]['posx'], G.nodes[i]['posy']))
    longest_skel = LineString(longest_skel)


    return G, longest_skel


def get_modified_nodes(coords):
    default = 1
    for i in range(coords.shape[1] - 2):
        k1 = get_k_angle(Point(coords[:,i]), Point(coords[:,i+1]) )
        k2 = get_k_angle(Point(coords[:,i+1]), Point(coords[:,i+2]) )
        # print(coords[:,i], coords[:,i+1], coords[:,i+2], k1, k2 , 5 * np.pi/36.0)
        if np.abs(k1 - k2) > 2 * np.pi/36.0:  # 5 degree
            # if front:
            return i+1
            # else:
            #     return Point(coords[:,i+2]), Point(coords[:,i+1])
    # print('no found')
    return default


def modified_skel_to_medaxis(longest_skel, block):

    coords = np.array(longest_skel.coords.xy)
    if coords.shape[1] <=3:
        return longest_skel
    elif coords.shape[1] == 4:
        front_start = 1
        end_start = 1

    else:
        flip_coords = np.flip(coords,1)
        # print(coords, flip_coords)
        front_start = get_modified_nodes(coords)
        end_start = get_modified_nodes(flip_coords)
        # print(front_start, end_start)

    p2 = Point(coords[:,front_start])
    p3 = Point(coords[:,front_start + 1])
    p_2 = Point(coords[:,-end_start -1])
    p_3 = Point(coords[:,-end_start -2])

    # print(p2, p3, p_2, p_3)

    extended_line_front = get_extend_line(p2, p3, block, True)
    extended_line_back = get_extend_line(p_2, p_3, block, False)

    # print(coords.shape, front_start, end_start, longest_skel.coords[front_start:-end_start])
    if len(longest_skel.coords[front_start:-end_start]) <=1:
        midaxis_miss_bothend = LineString(longest_skel.coords[1:-1])
        print('no enough point on medaxis')
    else:
        midaxis_miss_bothend = LineString(longest_skel.coords[front_start:-end_start])
    # print('midaxis without end: ', midaxis_miss_bothend)

    modfied_midaxis = linemerge([extended_line_front, midaxis_miss_bothend, extended_line_back])

    return modfied_midaxis



###############################################################
def get_image_block_midaxis(img, scale = 10.0, simp_thres = 1.5):
    thres_point = 2.5
    # recover_scale = 100.0 / scale

    tile_width = img.shape[0]
    tile_height = img.shape[1]
    
    blk, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if blk[0].reshape(-1, 2).shape[0] <=1:
        org_block = Polygon(blk[1].reshape(-1, 2))
    else:  
        org_block = Polygon(blk[0].reshape(-1, 2))
    block = org_block.simplify(2.0)
    # offset_cnt = blk[0]

    exterior_polyline = list(block.exterior.coords)[:-1]
    exterior_polyline.reverse()
    poly_list = []
    for ix in range(len(exterior_polyline)):
        poly_list.append(exterior_polyline[ix])
    poly_block = skgeom.Polygon(poly_list)

    skel = skgeom.skeleton.create_interior_straight_skeleton(poly_block)
    G, longest_skel = get_polyskeleton_longest_path(skel, poly_block)

    medaxis = modified_skel_to_medaxis(longest_skel, block)


    # # cv2.imwrite(os.path.join(transvis_fp, ct_f_idx + '_q.png'), img)
    # imgSk, distance = medial_axis(img, return_distance=True)
    # # cv2.imwrite(os.path.join(transvis_fp, ct_f_idx + '_s.png'), imgSk.astype(np.uint8) * 255.0)
    

    # is_success, edge_swap = get_longest_path(imgSk, thres_point)
    # if not is_success:
    #     return is_success, None, None

    # # im_skt1 = np.zeros(img.shape)
    # # cv2.polylines(im_skt1, [np.int32(edge_swap)], False, (255), thickness=1, lineType=8, shift=0)
    # # cv2.drawContours(im_skt1, [offset_cnt], 0, (255), 1)
    # # cv2.imwrite(os.path.join(rawvis_fp, ct_f_idx + '_skt.png'), im_skt1)

    # mid_contour = np.array(edge_swap.copy(), dtype = np.double)
    # # print(mid_contour)
    # if len(mid_contour) < 2:
    #     return False, None, None
    # # mid_contour[:, 1] = (tile_height * recover_scale - mid_contour[:, 1]) / recover_scale + block.bounds[1]
    # # mid_contour[:, 0] = mid_contour[:, 0] / recover_scale + block.bounds[0]


    # midaxis = LineString(mid_contour)
    # midaxis = midaxis.simplify(simp_thres)

    # modfied_midaxis = get_modified_midaxis(midaxis, block)

    # if np.array(modfied_midaxis.coords.xy)[0, 0] > np.array(modfied_midaxis.coords.xy)[0, -1]:
    #     modfied_midaxis = LineString(modfied_midaxis.coords[::-1])

    return True, medaxis, block, org_block





###############################################################
def get_insert_position(arr, K):
    # Traverse the array
    for i in range(arr.shape[0]):         
        # If K is found
        if arr[i] == K:
            return np.int16(i)
        # If arr[i] exceeds K
        elif arr[i] >= K:
            return np.int16(i)
    return np.int16(arr.shape[0])


################################################################
def get_block_width_from_pt_on_midaxis(block, vector_midaxis, pt_on_midaxis):
    unit_v =  vector_midaxis / np.linalg.norm(vector_midaxis)
    left_v = np.array([-unit_v[1], unit_v[0]])
    right_v = np.array([unit_v[1], -unit_v[0]])
    
    dummy_left_pt = Point(pt_on_midaxis.x + left_v[0], pt_on_midaxis.y + left_v[1])
    dummy_right_pt = Point(pt_on_midaxis.x + right_v[0], pt_on_midaxis.y + right_v[1])

    left_line_to_contour = get_extend_line(dummy_left_pt, pt_on_midaxis, block, False, is_extend_from_end = True)
    right_line_to_contour = get_extend_line(dummy_right_pt, pt_on_midaxis, block, False, is_extend_from_end = True)

    # print(left_line_to_contour.length, right_line_to_contour.length)
    
    return left_line_to_contour.length + right_line_to_contour.length



###############################################################
def get_block_aspect_ratio(block, midaxis):
    if midaxis.geom_type == 'GeometryCollection':
        for jj in list(midaxis.geoms):
            if jj.geom_type == 'LineString':
                midaxis = jj
                break

    coords = np.array(midaxis.coords.xy)
    midaxis_length = midaxis.length

    ################################################################
    relative_cutoff = [0.0]
    vector_midaxis = []
    block_width_list = []

    coords = np.array(midaxis.coords.xy)

    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])

        if i < coords.shape[1] - 1:
            cur_width = get_block_width_from_pt_on_midaxis(block, coords[:, i] - coords[:, i-1], Point(coords[0, i], coords[1, i])) # each node on midaxis, except the last and the front.
            block_width_list.append(cur_width)

    if block_width_list == []:
        mean_block_width = block.bounds[3] - block.bounds[1]
    else:
        block_width_list = np.array(block_width_list)
        mean_block_width = np.mean(block_width_list)

    aspect_rto = np.double(mean_block_width) / np.double(midaxis_length)
    return aspect_rto



###############################################################
def warp_bldg_by_midaxis(bldg, block, midaxis):
    bldgnum = len(bldg)
    normalized_size = []
    midaxis_length = midaxis.length

    ################################################################
    relative_cutoff = [0.0]
    vector_midaxis = []
    block_width_list = []

    if midaxis.geom_type == 'GeometryCollection':
        for jj in list(midaxis.geoms):
            if jj.geom_type == 'LineString':
                midaxis = jj
                break
    coords = np.array(midaxis.coords.xy)

    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])

        if i < coords.shape[1] - 1:
            cur_width = get_block_width_from_pt_on_midaxis(block, coords[:, i] - coords[:, i-1], Point(coords[0, i], coords[1, i])) # each node on midaxis, except the last and the front.
            block_width_list.append(cur_width)


    if block_width_list == []:
        mean_block_width = block.bounds[3] - block.bounds[1]
    else:
        block_width_list = np.array(block_width_list)
        mean_block_width = np.mean(block_width_list)

    relative_cutoff = np.array(relative_cutoff)
    vector_midaxis = np.array(vector_midaxis)

    normalized_x = []
    corres_midaxis_vector_idx = []
    for i in range(bldgnum):
        cur_x = midaxis.project(bldg[i].centroid, normalized=True)
        normalized_x.append(cur_x)
        insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
        if insert_pos > vector_midaxis.shape[0]-1:
            print('\n out of index in vector_midaxis. \n')
            corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
            continue
        corres_midaxis_vector_idx.append(insert_pos)
    corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)

    normalized_y = []
    for i in range(bldgnum):
        vertical_point_on_midaxis = midaxis.interpolate(normalized_x[i], normalized=True)
        # line_to_contour = get_extend_line(bldg[i].centroid, vertical_point_on_midaxis, block, False, is_extend_from_end = True)  # [vertical_point_on_midaxis, nearest_points_on_contour]

        vector_midaxis_to_bldg = np.array( [bldg[i].centroid.x - vertical_point_on_midaxis.x, bldg[i].centroid.y - vertical_point_on_midaxis.y] )
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        cross_prod = np.cross(cur_vector_midaxis, vector_midaxis_to_bldg)

        dist_from_midaxis_to_bldg = np.sqrt(vector_midaxis_to_bldg[0] * vector_midaxis_to_bldg[0] + vector_midaxis_to_bldg[1] * vector_midaxis_to_bldg[1])
        # dist_from_midaxis_to_contour = np.double(line_to_contour.length)
        # relative_y = dist_from_midaxis_to_bldg / dist_from_midaxis_to_contour   ####  changed from "dist_from_midaxis_to_contour", without 2.0 multiple
       
        relative_y = 2.0 * dist_from_midaxis_to_bldg / mean_block_width #  changed from "dist_from_midaxis_to_contour", without 2.0 multiple
        # print(dist_from_midaxis_to_bldg, mean_block_width, cross_prod, relative_y)
        if cross_prod <= 0:
            normalized_y.append(-relative_y)
        else:
            normalized_y.append(relative_y)

        ###########################################################################################
        longside, shortside, long_vec, short_vec, _, _ = get_size_with_vector(bldg[i].minimum_rotated_rectangle)
        long_vec = long_vec / np.linalg.norm(long_vec)
        unit_cur_vector_midaxis =  cur_vector_midaxis / np.linalg.norm(cur_vector_midaxis)

        long_angle = np.arccos(np.dot(long_vec, unit_cur_vector_midaxis ))
        long_angle = np.min([np.pi - long_angle, long_angle])

        short_vec = short_vec / np.linalg.norm(short_vec)
        short_angle = np.arccos(np.dot(short_vec, unit_cur_vector_midaxis ))
        short_angle = np.min([np.pi - short_angle, short_angle])

        if short_angle < long_angle:
            currr = shortside
            shortside = longside
            longside = currr

        normalized_size_x = 2 * longside / midaxis_length  ############ because pos is [-1, 1], size can be 2 
        normalized_size_y = 2 * shortside / mean_block_width # changed from divded by "dist_from_midaxis_to_contour"
        normalized_size.append([normalized_size_x, normalized_size_y])
        
    normalized_x = np.array(normalized_x, dtype = np.double)
    normalized_x = 2.0 * normalized_x - 1.0    ############ normalize pos_x to [-1, 1], pos_y already been [-1, 1] 
    normalized_y = np.array(normalized_y, dtype = np.double)
    normalized_pos = np.stack((normalized_x, normalized_y), axis = 1)
    normalized_size = np.array(normalized_size, np.double)

    pos_sort = np.lexsort((normalized_pos[:,1],normalized_pos[:,0]))
    pos_sorted = normalized_pos[pos_sort]
    size_sorted = normalized_size[pos_sort]

    aspect_rto = np.double(mean_block_width) / np.double(midaxis_length)

    return pos_sorted, size_sorted, pos_sort, aspect_rto


#######################################################################################################################
def get_unwarpped_info(bldg, block):
    aspect_rto = get_aspect_ratio(block.minimum_rotated_rectangle)
    blk_longside, blk_shortside = get_size(block.minimum_rotated_rectangle)
    size = []
    pos = []
    for i in range(len(bldg)):
        pos.append([np.double(bldg[i].centroid.x), np.double(bldg[i].centroid.y)])
        minx, miny, maxx, maxy = bldg[i].bounds
        size.append([maxx - minx, maxy - miny])
    
    pos = np.array(pos)
    size = np.array(size)

    pos[:,0] = pos[:,0] / blk_longside 
    pos[:,1] = pos[:,1] / blk_shortside

    size[:,0] = size[:,0] / blk_longside
    size[:,1] = size[:,1] / blk_shortside

    pos_sort = np.lexsort((pos[:,1],pos[:,0]))
    pos_sorted = pos[pos_sort]
    size_sorted = size[pos_sort]

    return pos_sorted, size_sorted, pos_sort, aspect_rto


#######################################################################################################################
def get_polygon_by_pos_size_array(pos, size, block):
    blk_longside, blk_shortside = get_size(block.minimum_rotated_rectangle)
    minx, miny, maxx, maxy = block.bounds

    out = []
    bounds = []
    for i in range(pos.shape[0]):
        ix = pos[i, 0] - size[i, 0] / 2.0
        iy = pos[i, 1] - size[i, 1] / 2.0
        ax = pos[i, 0] + size[i, 0] / 2.0
        ay = pos[i, 1] + size[i, 1] / 2.0
        bounds.append([ix, iy, ax, ay])

    bounds = np.array(bounds, dtype = np.double) # (minx, miny, maxx, maxy)
    bounds[:, 0] = bounds[:, 0] * blk_longside # + minx
    bounds[:, 2] = bounds[:, 2] * blk_longside # + minx

    bounds[:, 1] = bounds[:, 1] * blk_shortside #+ miny
    bounds[:, 3] = bounds[:, 3] * blk_shortside #+ miny
    
    for i in range(bounds.shape[0]):
        bl = (bounds[i, 0], bounds[i, 1])
        br = (bounds[i, 2], bounds[i, 1])
        ur = (bounds[i, 2], bounds[i, 3])
        ul = (bounds[i, 0], bounds[i, 3])
        out.append(Polygon([bl, br, ur, ul]))

    pos[:, 0] = pos[:, 0] * blk_longside
    size[:, 0] = size[:, 0] * blk_longside
    pos[:, 1] = pos[:, 1] * blk_shortside
    size[:, 1] = size[:, 1] * blk_shortside

    return out, pos, size

#######################################################################################################################


def get_bbx(pos, size):
    bl = ( pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0)
    br = ( pos[0] + size[0] / 2.0, pos[1] - size[1] / 2.0)
    ur = ( pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0)
    ul = ( pos[0] - size[0] / 2.0, pos[1] + size[1] / 2.0)
    return bl, br, ur, ul




########################### Input position and size from graph output, and block, and midaxis. Output original bldg with correct position/size/rotation ################################################################
def inverse_warp_bldg_by_midaxis(pos_sorted, size_sorted, midaxis, aspect_rto, rotate_bldg_by_midaxis = True, output_mode = False):
    org_size = np.zeros_like(pos_sorted)
    org_pos = np.zeros_like(pos_sorted)

    pos_sorted[:, 0] = (pos_sorted[:, 0] + 1.0) / 2.0 ############ normalize pos_x [-1, 1] back to [0, 1], pos_y keep [-1, 1] 
    pos_sorted[:, 1] = pos_sorted[:, 1] / 2.0
    size_sorted = size_sorted / 2.0

    midaxis_length = midaxis.length
    mean_block_width = aspect_rto * midaxis_length
    org_size[:, 0] = size_sorted[:, 0] * midaxis_length

    ###############################################################################   same as forward processing   ###################
    relative_cutoff = [0.0]
    vector_midaxis = []
    coords = np.array(midaxis.coords.xy)
    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])
    relative_cutoff = np.array(relative_cutoff)
    vector_midaxis = np.array(vector_midaxis)

    bldgnum = pos_sorted.shape[0]        
    corres_midaxis_vector_idx = []
    for i in range(bldgnum):
        cur_x = pos_sorted[i, 0]
        insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
        if insert_pos > vector_midaxis.shape[0]-1:
            print('\n out of index in vector_midaxis. \n')
            corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
            continue
        corres_midaxis_vector_idx.append(insert_pos)
    corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)
    ###############################################################################   same as forward processing   ###################

    ###############################################################################   get correct position and size   ###################
    for i in range(bldgnum):
        vertical_point_on_midaxis = midaxis.interpolate(pos_sorted[i, 0], normalized=True)
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        if pos_sorted[i, 1] <= 0:
            vec_from_midaxis_to_bldg = np.array([cur_vector_midaxis[1], -cur_vector_midaxis[0]])
        else:
            vec_from_midaxis_to_bldg = np.array([-cur_vector_midaxis[1], cur_vector_midaxis[0]])
        
        vec_from_midaxis_to_bldg = vec_from_midaxis_to_bldg / np.linalg.norm(vec_from_midaxis_to_bldg)

        cur_pos_x = vertical_point_on_midaxis.x + vec_from_midaxis_to_bldg[0] * np.abs(pos_sorted[i, 1]) * (mean_block_width)
        cur_pos_y = vertical_point_on_midaxis.y + vec_from_midaxis_to_bldg[1] * np.abs(pos_sorted[i, 1]) * (mean_block_width)


        org_pos[i, 0], org_pos[i, 1] = cur_pos_x, cur_pos_y
        org_size[i, 1] = size_sorted[i, 1] * mean_block_width   ##   changed from multiply by "line_from_midaxis_to_contour.length"
    ###############################################################################   get correct position and size   ###################
    if output_mode:
        org_pos, org_size = modify_pos_size_arr_overlap(org_pos, org_size)

    ###############################################################################   get original rotation  ###################
    org_bldg = []
    for i in range(org_pos.shape[0]):
        curr_bldg = Polygon(get_bbx(org_pos[i,:], org_size[i,:]))
        if rotate_bldg_by_midaxis:
            cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
            angle = np.arctan2(cur_vector_midaxis[1], cur_vector_midaxis[0]) * 180.0 / np.pi
            curr_bldg = sa.rotate(curr_bldg, angle, origin=(org_pos[i,0], org_pos[i,1]))
        org_bldg.append(curr_bldg)
    
    return org_bldg , org_pos, org_size
#######################################################################################################################



def bldggroup_write_to_obj(fpath, fname, bldg, height = None):
    facade_normal_uv_mtl_face = ['vt 1.000000 0.000000',
    'vt 1.000000 1.000000',
    'vt 0.000000 1.000000',
    'vt 0.000000 0.000000',
    'vt 1.000000 1.000000',
    'vt 1.000000 0.000000',
    'vt 1.000000 0.000000',
    'vt 1.000000 1.000000',
    'vt 0.000000 1.000000',
    'vt 0.000000 0.000000',
    'vt 0.000000 0.000000',
    'vt 0.000000 1.000000',
    'vn 0.0000 0.0000 1.0000',
    'vn -1.0000 0.0000 0.0000',
    'vn 1.0000 0.0000 0.0000',
    'vn 0.0000 0.0000 -1.0000',
    'usemtl facade_front',
    's off'
    ]

    roof_normal_uv_mtl_face = [
    'vt 1.000000 1.000000',
    'vt 0.000000 1.000000',
    'vt 0.000000 0.000000',
    'vt 1.000000 0.000000',
    'vn 0.0000 1.0000 0.0000',
    'usemtl roof',
    's off'
    ]


    bottom_normal_uv_mtl_face = [
    'vt 0.000000 1.000000',
    'vt 1.000000 1.000000',
    'vt 1.000000 0.000000',
    'vt 0.000000 0.000000',
    'vn 0.0000 -1.0000 0.0000',
    'usemtl None',
    's off'
    ]


    terrain_normal_uv_mtl_face = [
    'vt 7.909153 -7.000000',
    'vt -6.909153 -7.000000',
    'vt -6.909153 8.000000',
    'vt 7.909153 8.000000',
    'vn 0.0000 1.0000 0.0000',
    'usemtl terrain',
    's off'
    ]



    bldgnum = len(bldg)
    _, group_bbx = get_bldggroup_parameters(bldg)
    bminx, bminz, bmaxx, bmaxz = group_bbx.bounds

    with open(os.path.join(fpath,fname + '.obj'), 'w') as f:
        f.write("# Blender v3.1.2 OBJ File: ''\n")
        f.write("# www.blender.org\n")
        f.write('mtllib {}.mtl\n'.format(fname))

        for i in range(bldgnum):
            y = 3.0 + 2.0 * random.random()
            minx, minz, maxx, maxz = bldg[i].bounds    # (minx, miny, maxx, maxy)
            v = i * 16
            vt = i * 20
            vn = i * 6

            #########################   Facade front side #########
            f.write('o Facade_front{}\n'.format(i))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, y, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, 0.0, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, y, maxz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, 0.0, maxz))

            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, y, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, 0.0, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, y, maxz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, 0.0, maxz))

            f.writelines("%s\n" % place for place in facade_normal_uv_mtl_face)
            f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(4+v, 1+vt, 1+vn, 3+v, 2+vt, 1+vn, 7+v, 3+vt, 1+vn, 8+v, 4+vt, 1+vn))

            ############################   Back side   ##################
            f.write('o Facade_back{}\n'.format(i))
            f.write('usemtl facade_back\n')
            f.write('s off\n')
            f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(6+v, 11+vt, 4+vn, 5+v, 12+vt, 4+vn, 1+v, 8+vt, 4+vn, 2+v, 7+vt, 4+vn))

            ############################   Two Sides   ##################
            f.write('o Facade_side{}\n'.format(i))
            f.write('usemtl bricks\n')
            f.write('s off\n')
            f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(8+v, 4+vt, 2+vn, 7+v, 3+vt, 2+vn, 5+v, 5+vt, 2+vn, 6+v, 6+vt, 2+vn))
            f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(2+v, 7+vt, 3+vn, 1+v, 8+vt, 3+vn, 3+v, 9+vt, 3+vn, 4+v, 10+vt, 3+vn))

            ############################   Roof   ##################
            f.write('o Roof{}\n'.format(i))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, y, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, y, maxz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, y, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, y, maxz))
            f.writelines("%s\n" % place for place in roof_normal_uv_mtl_face)
            f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(9+v, 13+vt, 5+vn, 11+v, 14+vt, 5+vn, 12+v, 15+vt, 5+vn, 10+v, 16+vt, 5+vn))

            ############################   Bottom  ##################       
            f.write('o Bottom{}\n'.format(i))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, 0.0, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(maxx, 0.0, maxz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, 0.0, minz))
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(minx, 0.0, maxz))
            f.writelines("%s\n" % place for place in bottom_normal_uv_mtl_face)
            f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(15+v, 17+vt, 6+vn, 13+v, 18+vt, 6+vn, 14+v, 19+vt, 6+vn, 16+v, 20+vt, 6+vn))

        # ############################   Terrain  ##################   
        # f.write('o Terrain\n')
        # f.write('v {:.6f} {:.6f} {:.6f}\n'.format(bminx, 0.0, bmaxz))
        # f.write('v {:.6f} {:.6f} {:.6f}\n'.format(bmaxx, 0.0, bmaxz))     
        # f.write('v {:.6f} {:.6f} {:.6f}\n'.format(bminx, 0.0, bminz)) 
        # f.write('v {:.6f} {:.6f} {:.6f}\n'.format(bmaxx, 0.0, bminz))
        # f.writelines("%s\n" % place for place in terrain_normal_uv_mtl_face)
        # f.write('f {}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(17+v, 21+vt, 7+vn, 18+v, 22+vt, 7+vn, 20+v, 23+vt, 7+vn, 19+v, 24+vt, 7+vn))




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





# ########################### Input position and size from graph output, and block, and midaxis. Output original bldg with correct position/size/rotation ################################################################
# def inverse_warp_bldg_by_midaxis_block(pos_sorted, size_sorted, block, midaxis, aspect_rto, rotate_bldg_by_midaxis = True):
#     org_size = np.zeros_like(pos_sorted)
#     org_pos = np.zeros_like(pos_sorted)

#     pos_sorted[:, 0] = (pos_sorted[:, 0] + 1.0) / 2.0 ############ normalize pos_x [-1, 1] back to [0, 1], pos_y keep [-1, 1] 
#     size_sorted = size_sorted / 2.0

#     midaxis_length = midaxis.length
#     mean_block_width = aspect_rto * midaxis_length
#     org_size[:, 0] = size_sorted[:, 0] * midaxis_length

#     ###############################################################################   same as forward processing   ###################
#     relative_cutoff = [0.0]
#     vector_midaxis = []
#     coords = np.array(midaxis.coords.xy)
#     for i in range(1, coords.shape[1]):
#         relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
#         vector_midaxis.append(coords[:, i] - coords[:, i-1])
#     relative_cutoff = np.array(relative_cutoff)
#     vector_midaxis = np.array(vector_midaxis)

#     bldgnum = pos_sorted.shape[0]        
#     corres_midaxis_vector_idx = []
#     for i in range(bldgnum):
#         cur_x = pos_sorted[i, 0]
#         insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
#         if insert_pos > vector_midaxis.shape[0]-1:
#             print('\n out of index in vector_midaxis. \n')
#             corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
#             continue
#         corres_midaxis_vector_idx.append(insert_pos)
#     corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)
#     ###############################################################################   same as forward processing   ###################

#     ###############################################################################   get correct position and size   ###################
#     for i in range(bldgnum):
#         vertical_point_on_midaxis = midaxis.interpolate(pos_sorted[i, 0], normalized=True)
#         cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
#         if pos_sorted[i, 1] <= 0:
#             vec_from_midaxis_to_bldg = np.array([cur_vector_midaxis[1], -cur_vector_midaxis[0]])
#         else:
#             vec_from_midaxis_to_bldg = np.array([-cur_vector_midaxis[1], cur_vector_midaxis[0]])
        
#         vec_from_midaxis_to_bldg = vec_from_midaxis_to_bldg / (np.linalg.norm(vec_from_midaxis_to_bldg) * 100.0)
#         dummy_bldg_direct_point = Point(vertical_point_on_midaxis.x + vec_from_midaxis_to_bldg[0], vertical_point_on_midaxis.y + vec_from_midaxis_to_bldg[1])

#         if pos_sorted[i, 1] >= 0:
#             line_from_midaxis_to_contour = get_extend_line(dummy_bldg_direct_point, vertical_point_on_midaxis, block, False, is_extend_from_end = True)  # [vertical_point_on_midaxis, nearest_points_on_contour]
#         else:
#             line_from_midaxis_to_contour = get_extend_line(dummy_bldg_direct_point, vertical_point_on_midaxis, block, True, is_extend_from_end = True)  # [vertical_point_on_midaxis, nearest_points_on_contour]

#         # cur_pos = line_from_midaxis_to_contour.interpolate(pos_sorted[i, 1] , normalized=True)
#         cur_pos = line_from_midaxis_to_contour.interpolate(pos_sorted[i, 1] * mean_block_width / 2.0)


#         org_pos[i, 0], org_pos[i, 1] = cur_pos.x, cur_pos.y
#         org_size[i, 1] = size_sorted[i, 1] * mean_block_width   ##   changed from multiply by "line_from_midaxis_to_contour.length"
#     ###############################################################################   get correct position and size   ###################

#     ###############################################################################   get original rotation  ###################
#     org_bldg = []
#     for i in range(bldgnum):
#         curr_bldg = Polygon(get_bbx(org_pos[i,:], org_size[i,:]))
#         if rotate_bldg_by_midaxis:
#             cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
#             angle = np.arctan2(cur_vector_midaxis[1], cur_vector_midaxis[0]) * 180.0 / np.pi
#             curr_bldg = sa.rotate(curr_bldg, angle, origin=(org_pos[i,0], org_pos[i,1]))
#         org_bldg.append(curr_bldg)
    
#     return org_bldg
# #######################################################################################################################