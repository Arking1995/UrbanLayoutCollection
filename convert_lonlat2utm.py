import geojson
import fiona
import os
import shapely
import networkx as nx
import pickle
from sklearn.neighbors import KDTree, BallTree
from math import radians, cos, sin, asin, sqrt
import utm
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import math
import json
import matplotlib.pyplot as plt
import osmnx as ox

### ignore the warning
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    #### get the dict_bbx from updated_bbx_list.json
    with open('/home/he425/Dataset/updated_bbx_list.json') as f:
        dict_bbx = json.load(f)

    city_idx = 0
    min_idx = 8
    max_idx = 8  ### 21, 50
    ### loop through the dict_bbx
    for cityname in tqdm(dict_bbx.keys()):
        city_idx += 1
        print('Start processing: ', cityname, ' city_idx: ', city_idx, ' out of 331 cities')

        #### only process cities ranges idx from min_idx to max_idx
        if city_idx < min_idx or city_idx > max_idx:
            continue

        #### read the raw_geo from pickle file
        with open(os.path.join('/home/he425/Dataset/Our_Dataset', cityname, 'raw_geo', cityname + '_blk_bldg_3d.pkl'), 'rb') as f:
            our_rows = pickle.load(f)
        
        ### add a converted utm polygon of each block in our_rows
        for i in tqdm(range(len(our_rows))): ###len(our_rows)
            blk_poly = shapely.geometry.shape(our_rows[i]['blk_TIGER_geometry'])
            proj_blk_poly, crs = ox.projection.project_geometry(blk_poly)

            our_rows[i]['blk_utm_geometry'] = proj_blk_poly
            our_rows[i]['blk_utm_crs'] = crs.to_dict()
            # print(our_rows[i]['blk_utm_crs'])

            ### convert fiona.properties to dict
            our_rows[i]['blk_TIGER_properties'] = dict(our_rows[i]['blk_TIGER_properties'])
            # print(type(our_rows[i]['blk_TIGER_properties']))
            # print(our_rows[i]['blk_TIGER_properties'])

            ### add a converted utm polygon of each building in our_rows
            for j in range(len(our_rows[i]['bldg_3D_geometry'])):
                bldg_poly = shapely.geometry.shape(our_rows[i]['bldg_3D_geometry'][j][1])
                proj_bldg_poly, crs = ox.projection.project_geometry(bldg_poly)
                our_rows[i]['bldg_3D_geometry'][j].append(proj_bldg_poly)
        
        ### save the our_rows to pickle file
        with open(os.path.join('/home/he425/Dataset/Our_Dataset', cityname, 'raw_geo', cityname + '_blk_bldg_3d_utm.pkl'), 'wb') as f:
            pickle.dump(our_rows, f)
            

