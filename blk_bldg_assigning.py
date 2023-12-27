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
import json


if __name__ == "__main__":

    MS_data_path = '/home/he425/Dataset/MS_Building_Dataset'
    osm_data_path = '/home/he425/Dataset/OSM_Dataset'
    TG_data_path = '/home/he425/Dataset/TIGER_Dataset/Block'
    State_Bound_path = '/home/he425/Dataset/TIGER_Dataset/State_Bound/tl_2022_us_state'
    output_path = '/home/he425/Dataset/Our_Dataset'

    buffer_radius = 500.0 / 6371000.0

    ### load the bbx_list from bbx_list.json
    with open('/home/he425/Dataset/updated_bbx_list.json') as f:
        dict_bbx = json.load(f)

    city_idx = 0
    min_idx = 51
    max_idx = 51 
    ### loop through the dict_bbx
    for cityname in tqdm(dict_bbx.keys()):
        city_idx += 1
        print('Start processing city: ', cityname, ' city_idx: ', city_idx, ' out of 331 cities')        
        
        #### only process cities ranges idx 1-5
        if city_idx < min_idx or city_idx > max_idx:
            continue

        ### setup random seed
        np.random.seed(42)

        ### mkdir for the city if not exist
        if not os.path.exists(os.path.join(output_path, cityname)):
            os.mkdir(os.path.join(output_path, cityname))
        
        ### mkdir for shapefile if not exist
        if not os.path.exists(os.path.join(output_path, cityname, 'shapefile')):
            os.mkdir(os.path.join(output_path, cityname, 'shapefile'))

        ### mkdir for raw_geo if not exist
        if not os.path.exists(os.path.join(output_path, cityname, 'raw_geo')):
            os.mkdir(os.path.join(output_path, cityname, 'raw_geo'))

        ##########################################################################
        ################### get list of city block geometries #################
        ##########################################################################
        ### get the bounding box of the city
        bbx = dict_bbx[cityname]

        ### get the shapely polygon of bbx
        aoi_poly = shapely.geometry.box(bbx[2], bbx[1], bbx[3], bbx[0])

        intersect_state = []
        ### load the state boundary shapefile of the US by fiona, get the STATEEFP of the state that the bbx locates
        with fiona.open(os.path.join(State_Bound_path, 'tl_2022_us_state.shp'), 'r') as f:
            st_bd_data = f

            ### check which of the state in st_bd_data that the bbx locates, and get the STATEEFP of that state
            for state in st_bd_data:
                state_shape = shapely.geometry.shape(state['geometry'])
                if state_shape.contains(aoi_poly) or state_shape.intersects(aoi_poly):
                    state_FIP = state['properties']['STATEFP']
                    print(state['properties']['STATEFP'], state['properties']['NAME'])
                    intersect_state.append(state['properties']['STATEFP'])

        ### combine the block shapefile of intersect_state            
        blk_geom_list = []
        blk_attr_list = []
        for ii in intersect_state:
            ### compose the selected block shapefile directory by the state_FIP
            blk_dat_fname = 'tl_2022_'+ str(ii) + '_tabblock20'
            blk_dat_shp_fname = os.path.join(TG_data_path, blk_dat_fname, blk_dat_fname + '.shp')

            ### load the city block shapefile of the selected state, get all city block geometris and attributes within the bbx
            ### also clip the city block geometries by the bbx, output clipped city block shapefile
            with fiona.open(blk_dat_shp_fname, 'r') as f:
                blk_data = f
                ### get the city block geometries and attributes within the bbx
                for blk in blk_data:
                    blk_shape = shapely.geometry.shape(blk['geometry'])
                    if blk_shape.intersects(aoi_poly):
                        blk_geom_list.append(blk_shape)
                        blk_attr_list.append(blk['properties'])
                schema = blk_data.schema

        ### save the city block geometries and attributes within the bbx to a shapefile
        with fiona.open(os.path.join(output_path, cityname, 'shapefile', cityname + '_blk_list.shp'), 'w', driver='ESRI Shapefile', schema=schema, crs = blk_data.crs) as f:
            for i in range(len(blk_geom_list)):
                f.write({
                    'geometry': shapely.geometry.mapping(blk_geom_list[i]),
                    'properties': blk_attr_list[i],
                })

        ### save the city block geometries and attributes within the bbx to a pickle file
        with open(os.path.join(output_path, cityname, 'shapefile', cityname + '_clipped_blk.pkl'), 'wb') as f:
            pickle.dump([blk_geom_list, blk_attr_list], f)




        #########################################################################
        ######## get building geometries within each city block #################
        #########################################################################    
        print('Finish cliping block shapefile, start grouping...')
        ### load the city block geometries and attributes within the bbx from pickle file
        with open(os.path.join(output_path, cityname, 'shapefile',cityname + '_clipped_blk.pkl'), 'rb') as f:
            blk_geom_list, blk_attr_list = pickle.load(f)
        
        ### Load GeoJSON data from a file or a string (in this example, we load from a file)
        with open(os.path.join(MS_data_path, cityname, cityname + '_bldg.geojson'), 'rb') as f:
            ms_bldg_data = geojson.load(f)

        ### load the list of locations of all buildings in MS dataset
        with open(os.path.join(MS_data_path, cityname, cityname + '_bldg_loc_list.pkl'), 'rb') as f:
            ms_bldg_loc_list = pickle.load(f)

        ### load the osm building footprints
        with open(os.path.join(osm_data_path, cityname, 'osm_' + cityname + '_bldg.geojson'), 'rb') as f:
            osm_bldg_data = geojson.load(f)
        
        ### load the list of locations of all buildings in osm dataset
        with open(os.path.join(osm_data_path, cityname, 'osm_' + cityname + '_bldg_loc_list.pkl'), 'rb') as f:
            osm_bldg_loc_list = pickle.load(f)

        ### get the numpy array of loations of all buildings in MS dataset
        ms_bldg_loc_list = np.array(ms_bldg_loc_list)
        ### convert the coordinates to radians
        ms_bldg_loc_list_rad = np.deg2rad(ms_bldg_loc_list)
        ### reverse the columns of the numpy array of loations of all buildings in MS dataset --> lon/lat to lat/lon
        ms_bldg_loc_list_rad = ms_bldg_loc_list_rad[:, [1, 0]]

        ### get the numpy array of loations of all buildings in osm dataset
        osm_bldg_loc_list = np.array(osm_bldg_loc_list)
        ### convert the coordinates to radians
        osm_bldg_loc_list_rad = np.deg2rad(osm_bldg_loc_list)
        ### reverse the columns of the numpy array of loations of all buildings in osm dataset --> lon/lat to lat/lon
        osm_bldg_loc_list_rad = osm_bldg_loc_list_rad[:, [1, 0]]


        ### build the balltree of the locations of all buildings in osm dataset
        osm_bldg_loc_balltree = BallTree(osm_bldg_loc_list_rad, metric='haversine')
        ### build the balltree of the locations of all buildings in MS dataset
        ms_bldg_loc_balltree = BallTree(ms_bldg_loc_list_rad, metric='haversine')


        ## loop each city block centroid, search the building geometries within 500m buffer 
        total_blk_num = len(blk_geom_list)
        our_rows = []
        blk_idx = 0
        bldg_idx = 0

        print('Total block number before processing: ', total_blk_num)
        print('Finish loading data, start processing...')    
        for i in tqdm(range(total_blk_num)): ###range(total_blk_num)
            ### get the city block centroid
            blk_poly = blk_geom_list[i]
            blk_loc_centroid = blk_poly.centroid.coords[0]
            ### convert the centroid coordinates to radians and in lat/lon order
            blk_loc_centroid_rad = np.deg2rad([blk_loc_centroid[1], blk_loc_centroid[0]])
            TIGER_blk_bldg_num = blk_attr_list[i]['HOUSING20']

            ### check if HOUSING20 (building number) of the city block is 0 or not, if so, skip this city block
            # if TIGER_blk_bldg_num == 0:
            #     o_row = {}
            #     o_row['blk_TIGER_geometry'] = blk_poly
            #     o_row['blk_TIGER_properties'] = blk_attr_list[i]
            #     o_row['bldg_3D_geometry'] = []
            #     o_row['blk_id'] = blk_idx
            #     o_row['bldg_total_num'] = 0
            #     blk_idx += 1
            #     our_rows.append(o_row)
            #     continue

            ### get the list of the index of building geometries within 500m buffer of the blk_loc_centroid
            ms_bldg_idx_list = ms_bldg_loc_balltree.query_radius([blk_loc_centroid_rad], r=buffer_radius, count_only=False, return_distance=False)[0]

            ### get osm bldg list
            osm_bldg_idx_list = osm_bldg_loc_balltree.query_radius([blk_loc_centroid_rad], r=buffer_radius, count_only=False, return_distance=False)[0]

            ### check the corresponding building geometries is within the city block polygon or not, save the sub-list of building geometries within the city block polygon
            ms_overlap_bldg_idx_list = []
            ms_sum_area = 0.0
            for j in range(len(ms_bldg_idx_list)):
                bldg_poly = shapely.geometry.shape(ms_bldg_data['features'][ms_bldg_idx_list[j]]['geometry']).buffer(0.0)
                if blk_poly.contains(bldg_poly):
                    ms_sum_area += bldg_poly.area
                    ms_overlap_bldg_idx_list.append(ms_bldg_idx_list[j])
            
            ### check also osm bldg list
            osm_overlap_bldg_idx_list = []
            osm_sum_area = 0.0
            for j in range(len(osm_bldg_idx_list)):
                bldg_poly = shapely.geometry.shape(osm_bldg_data['features'][osm_bldg_idx_list[j]]['geometry']).buffer(0.0)
                if blk_poly.contains(bldg_poly):
                    osm_sum_area += bldg_poly.area
                    osm_overlap_bldg_idx_list.append(osm_bldg_idx_list[j])
                
            ### compare the bldg number and area of MS and OSM, choose the larger one to process, then combine geometric non-overlapped building from other dataset
            if len(ms_overlap_bldg_idx_list) >= len(osm_overlap_bldg_idx_list):
                more_bldg_idx_list = ms_overlap_bldg_idx_list
                compensate_bldg_idx_list = osm_overlap_bldg_idx_list
                more_bldg_data = ms_bldg_data
                compensate_bldg_data = osm_bldg_data
                m_tag = 'MS'
                cm_tag = 'OSM'
            else:
                more_bldg_idx_list = osm_overlap_bldg_idx_list
                compensate_bldg_idx_list = ms_overlap_bldg_idx_list
                more_bldg_data = osm_bldg_data
                compensate_bldg_data = ms_bldg_data
                m_tag = 'OSM'
                cm_tag = 'MS'


            bldg_geom_height_list = []
            ### append all building footprints from more_bldg_idx_list to o_row['bldg_geometry'], and compensate the non-overlapped building footprints from compensate_bldg_idx_list
            for idx in more_bldg_idx_list:
                m_bldg_poly = shapely.geometry.shape(more_bldg_data['features'][idx]['geometry'])
                m_bldg_height = more_bldg_data['features'][idx]['properties']['height']
                bldg_geom_height_list.append([m_bldg_poly, m_bldg_height, m_tag, m_tag])

            for ix in compensate_bldg_idx_list:
                cm_bldg_poly = shapely.geometry.shape(compensate_bldg_data['features'][ix]['geometry'])
                cm_bldg_height = compensate_bldg_data['features'][ix]['properties']['height']
                ### check if cm_bldg_poly is overlapped with any building footprints from more_bldg_idx_list
                overlap_flag = False
                for bldg_geom_height in bldg_geom_height_list:
                    if bldg_geom_height[0].intersects(cm_bldg_poly):
                        overlap_flag = True
                        if m_tag == 'OSM' and bldg_geom_height[1] < 0.0:
                            ### compensate the height of the overlapped building footprints from MS dataset
                            bldg_geom_height[1] = cm_bldg_height
                            bldg_geom_height[3] = cm_tag
                        break
                ### if not overlapped, append the cm_bldg_poly to o_row['bldg_geometry']
                if not overlap_flag:
                    bldg_geom_height_list.append([cm_bldg_poly, cm_bldg_height, cm_tag, cm_tag])
                    

            #### when use OSM there is no height info typically.

            #### get the average height of all positive height values
            bldg_height_list = []
            for bldg_geom_height in bldg_geom_height_list:
                if bldg_geom_height[1] > 0.0:
                    bldg_height_list.append(bldg_geom_height[1])
            
            if len(bldg_height_list) > 0:
                avg_bldg_height = np.mean(bldg_height_list)
            else:
                avg_bldg_height = 5.0
            
            ### assign the average height added random noise to all negative height building footprints
            for bldg_geom_height in bldg_geom_height_list:
                if bldg_geom_height[1] < 0.0:
                    bldg_geom_height[1] = avg_bldg_height + np.random.normal(0.0, 1.0)
                    ### modify the tag of the building height by adding '_avg'
                    bldg_geom_height[3] = bldg_geom_height[3] + '_avg'
                else:
                    bldg_geom_height[3] = bldg_geom_height[3] + '_org'

            
            ### give each building footprint a unique id as the first element
            for bldg_geom_height in bldg_geom_height_list:
                bldg_geom_height.insert(0, bldg_idx)
                bldg_idx += 1

            # print(blk_idx, bldg_geom_height_list)

            o_row = {}
            o_row['blk_TIGER_geometry'] = blk_poly
            o_row['blk_TIGER_properties'] = blk_attr_list[i]
            o_row['bldg_3D_geometry'] = bldg_geom_height_list ####[bldg_id, bldg_poly, bldg_height, bldg_contour_tag, bldg_height_tag]
            o_row['blk_id'] = blk_idx
            o_row['bldg_total_num'] = len(bldg_geom_height_list)
            blk_idx += 1
            our_rows.append(o_row)

        print('Total block number after processing: ', len(our_rows))
        print('Finish loading processing, start saving...')

        ### save the list of building geometries and attributes within the bbx to a pickle file
        with open(os.path.join(output_path, cityname, 'raw_geo', cityname + '_blk_bldg_3d.pkl'), 'wb') as f:
            pickle.dump(our_rows, f)

        ### load the list of building geometries and attributes within the bbx from pickle file
        with open(os.path.join(output_path, cityname, 'raw_geo', cityname + '_blk_bldg_3d.pkl'), 'rb') as f:
            our_rows = pickle.load(f)
        print('Total block number after loading: ', len(our_rows))

        ### save all building geometries and height to a shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int', 'height': 'float', 'contour_tag': 'str', 'height_tag': 'str', 'blk_id': 'int'},
        }
        with fiona.open(os.path.join(output_path, cityname, 'raw_geo', cityname + '_bldg_3d.shp'), 'w', driver='ESRI Shapefile', schema=schema, crs = "EPSG:4326") as f:
            for row in our_rows:
                for bldg_geom_height in row['bldg_3D_geometry']:
                    f.write({
                        'geometry': shapely.geometry.mapping(bldg_geom_height[1]),
                        'properties': {'id': bldg_geom_height[0], 'height': float(bldg_geom_height[2]), 'contour_tag': bldg_geom_height[3], 'height_tag': bldg_geom_height[4], 'blk_id': row['blk_id']},
                    })


        ### save all block geometries and attributes to a shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {'blk_id': 'int'},
        }        

        with fiona.open(os.path.join(output_path, cityname, 'raw_geo', cityname + '_blk_3d.shp'), 'w', driver='ESRI Shapefile', schema=schema, crs = "EPSG:4326") as f: 
            for row in our_rows:
                f.write({
                    'geometry': shapely.geometry.mapping(row['blk_TIGER_geometry']),
                    'properties': {'blk_id': row['blk_id']},
                })
             
