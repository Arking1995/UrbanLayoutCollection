import osmnx as ox
import os
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import fiona
import pickle
import json

if __name__ == "__main__":
    ###### choose the city
    # city = 'chicago'

    ### read dict_bbx from updated_bbx_list.json
    with open('/home/he425/Dataset/updated_bbx_list.json') as f:
        dict_bbx = json.load(f)

    city_idx = 0    
    ### loop through the dict_bbx
    for city in tqdm(dict_bbx.keys()):
        city_idx += 1
        print('Start processing: ', city, ' city_idx: ', city_idx, ' out of 331 cities')
        if city_idx != 8:
            print('Skip other cities')
            continue

        pattern = r"[-+]?\d*\.\d+|\d+"

        output_path = '/home/he425/Dataset/OSM_Dataset'
        if not os.path.exists(os.path.join(output_path, city)):
            os.mkdir(os.path.join(output_path, city))

        ### get the bounding box of the city
        bbx = dict_bbx[city]

        tags = {"building": True}

        ### get the footprints from osm, it is a geopandas.GeoDataFrame
        all_footprints = ox.geometries_from_bbox(bbx[0], bbx[1], bbx[2], bbx[3], tags)

        # # #### save the geopandas.GeoDataFrame to a pickle file
        # # all_footprints.to_pickle(os.path.join(output_path, city, city + '_raw.pkl'))

        # #### load the building footprints from a pickle file
        # all_footprints = pd.read_pickle(os.path.join(output_path, city, city + '_raw.pkl'))

        print('found ', len(all_footprints), ' footprints in ', city)

        idx = 0
        combined_rows = []
        bldg_loc_list = []
        no_height_list = []
        ### check "geometry" of all footprints. And check if the geometry is Polygon. keep only the "geometry" column to a new geopandas.GeoDataFrame: bldg_footprints
        for row in tqdm(all_footprints.iterrows()):
            o_row = {}
            if isinstance(row[1]['geometry'], shapely.geometry.polygon.Polygon):
                o_row['geometry'] = row[1]['geometry']
                ### get the centroid of the polygon
                bldg_loc_list.append([row[1]['geometry'].centroid.x, row[1]['geometry'].centroid.y])
                # print(type(row[1]['height']), row[1]['height'])
                #### check if 'height' column is not NaN
                if 'height' in row[1] and pd.notnull(row[1]['height']):
                    ### get the float height number from the string
                    if type(row[1]['height']) == str:
                        height = float(re.findall(pattern, row[1]['height'])[0])
                    else:
                        no_height_list.append(idx)
                        height = float(row[1]['height'])
                else:
                    height = -1.0
            else:
                continue
            
            o_row['properties'] = {'id': idx, 'height': height}
            idx += 1
            combined_rows.append(o_row)
        
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int', 'height': 'float'},
        }

        with fiona.open(os.path.join(output_path, city, 'osm_' + city + '_bldg.geojson'), 'w', 'GeoJSON', schema) as f:
            f.writerecords(combined_rows)

        ### save the building location list to a pickle file
        with open(os.path.join(output_path, city, 'osm_' + city + '_bldg_loc_list.pkl'), 'wb') as f:
            pickle.dump(bldg_loc_list, f)

        ### save the no height list to a pickle file
        with open(os.path.join(output_path, city, 'osm_' + city + '_no_height_list.pkl'), 'wb') as f:
            pickle.dump(no_height_list, f)