import geojson
import os
import shapely
import networkx as nx
import pickle
from math import radians, cos, sin, asin, sqrt
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import math
import json
import matplotlib.pyplot as plt

from geometry_utils import *
from graph_util import *


def process(our_rows):
    chunk_length = len(our_rows)
    graph_prep_list = []
    # m_num = 0
    # c_num = 0


    for i in range(chunk_length):
        graph_prep_dict = {}
        blk_id = our_rows[i]['blk_id']
        blk_utm_poly = shapely.geometry.shape(our_rows[i]['blk_utm_geometry'])
        bldg_poly_list = our_rows[i]['bldg_3D_geometry']   ####[bldg_id, bldg_poly, bldg_height, bldg_contour_tag, bldg_height_tag]
        bldg_num = len(bldg_poly_list)
        bldg_num1 = our_rows[i]['bldg_total_num']

           
        ### assert bldg_num == bldg_num1
        assert bldg_num == bldg_num1

        # print('blk_id: ', blk_id, ' bldg_num: ', bldg_num)        
        #####################################################
        # ### get the max bldg num
        # if bldg_num > m_num:
        #     m_num = bldg_num

        # if bldg_num > max_bldg:
        #     overmax_blk_bldg_num_list.append(bldg_num)

        # blk_bldg_num_list.append(bldg_num)
        # ### count the number of blocks that have more than max_bldg
        # c_num += (bldg_num > max_bldg)

        #####################################################


        ### ignore the buildings more than max_bldg
        clipped_bldg_poly_list = bldg_poly_list[:max_bldg]

        clipped_bldg_num = len(clipped_bldg_poly_list)
        filtered_id = np.arange(1, len(clipped_bldg_poly_list)+1)

        bldg_id_list = []
        ### got the bldg_idx of the block
        for ii in range(len(clipped_bldg_poly_list)):
            bldg_id_list.append(clipped_bldg_poly_list[ii][0])
 
        # ### remove the building that is too small
        # clipped_bldg_poly_list, filtered_id = remove_small_bldg(clipped_bldg_poly_list, filtered_id, min_area = 1.0)

        bldg_num = len(clipped_bldg_poly_list)

        if bldg_num == 0:
            ### no building in the block, skip canoical transform
            graph_prep_dict['blk_id'] = blk_id
            graph_prep_dict['bldg_num'] = 0
            graph_prep_dict['blk_TIGER_properties'] = our_rows[i]['blk_TIGER_properties']
            graph_prep_dict['blk_utm_geometry'] = our_rows[i]['blk_utm_geometry']
            graph_prep_dict['blk_utm_crs'] = our_rows[i]['blk_utm_crs']
            graph_prep_list.append(graph_prep_dict)
            continue

        ### get the canonical transform
        blk_azimuth, blk_bbx = get_block_parameters(blk_utm_poly)

        norm_utm_bldg_poly = norm_block_to_horizonal(clipped_bldg_poly_list, blk_azimuth, blk_utm_poly)
        norm_utm_blk_poly = norm_block_to_horizonal([blk_utm_poly], blk_azimuth, blk_utm_poly)[0]

        # print('blk_utm_poly: ', blk_utm_poly)  
        # print('norm_utm_blk_poly: ', norm_utm_blk_poly)

        simp_thres = simpthres_select(norm_utm_blk_poly)
        norm_utm_blk_poly = norm_utm_blk_poly.simplify(simp_thres)

        blk_shape_rto = np.double(norm_utm_blk_poly.area) / (np.double(norm_utm_blk_poly.minimum_rotated_rectangle.area) + 1e-6)
        blk_length_rto = norm_utm_blk_poly.length / (norm_utm_blk_poly.minimum_rotated_rectangle.length + 1e-6)
        # print('blk_shape_rto: ', blk_shape_rto, ' blk_length_rto: ', blk_length_rto)


        ##############  get the main axis #######################################
        # print(list(norm_utm_blk_poly.exterior.coords))
        ### if geom is multipolygon, use the largest polygon
        if norm_utm_blk_poly.geom_type == 'MultiPolygon':
            max_area = 0
            for jj in list(norm_utm_blk_poly.geoms):
                if jj.area > max_area:
                    max_area = jj.area
                    norm_utm_blk_poly = jj

        exterior_polyline = list(norm_utm_blk_poly.exterior.coords)[:-1]
        exterior_polyline.reverse()
        poly_list = []
        for ix in range(len(exterior_polyline)):
            poly_list.append(exterior_polyline[ix])
        poly_block = skgeom.Polygon(poly_list)

        try:
            skel = skgeom.skeleton.create_interior_straight_skeleton(poly_block)
            G, longest_skel = get_polyskeleton_longest_path(skel, poly_block)
        except:
            print('idx: ', i, ' blk_id: ', blk_id, ' has no skeleton')
            continue

        medaxis = modified_skel_to_medaxis(longest_skel, norm_utm_blk_poly)
        if medaxis.geom_type == 'GeometryCollection' or medaxis.geom_type == 'MultiLineString':
            for jj in list(medaxis.geoms):
                if jj.geom_type == 'LineString':
                    medaxis = jj
                    continue
        if medaxis.geom_type != 'LineString':
            continue

        #############   wrap all building locations and sizes ###############################################################
        if (blk_shape_rto < 0.9 or blk_length_rto > 1.25) and clipped_bldg_num > 3:
            pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(norm_utm_bldg_poly, norm_utm_blk_poly, medaxis)
        else:
            minx, miny, maxx, maxy = norm_utm_blk_poly.bounds
            midy = ( miny + maxy ) / 2.0
            medaxis = norm_utm_blk_poly.intersection(LineString([(minx, midy), (maxx, midy)]))
            if medaxis.geom_type == 'MultiLineString':    # some time the bldg is U-shape, so medaxis cut it into multistrings, in this case, use block's convex hull
                medaxis = norm_utm_blk_poly.convex_hull.intersection(LineString([(minx, midy), (maxx, midy)]))
            pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(norm_utm_bldg_poly, norm_utm_blk_poly, medaxis)  ####     aspect_rto = np.double(mean_block_width) / np.double(midaxis_length)
        
        filtered_id = filtered_id[xsort_idx]

        #################   Get real / syntheric bldg shape, and iou  ########################################################
        shape_list = []
        iou_list = []
        for ii in range(len(norm_utm_bldg_poly)):
            bldg_iou = norm_utm_bldg_poly[ii][5].area / norm_utm_bldg_poly[ii][5].minimum_rotated_rectangle.area
            if bldg_iou > 0.95:
                b_shape = 0
            else:
                b_shape, iou, b_height, b_width, theta = get_bldg_features(norm_utm_bldg_poly[ii][5])  # iou is the iou after template fitting, not original iou
            shape_list.append(b_shape)
            iou_list.append(bldg_iou)
        shape_list = np.array(shape_list)
        iou_list = np.array(iou_list)
        shape_xsorted = shape_list[xsort_idx]
        iou_xsorted = iou_list[xsort_idx]

        ########### remove overlapped buildings and assign to rows ############################################################
        # print(pos_xsorted, size_xsorted, filtered_id)
        # print('before de-overlapping', len(pos_xsorted), len(size_xsorted), filtered_id )
        pos_xsorted, size_xsorted, filtered_id = modify_pos_size_arr_overlap(pos_xsorted, size_xsorted, height = filtered_id)
        # print('after de-overlapping', len(pos_xsorted), len(size_xsorted), filtered_id )
        row_assign, each_row_num, all_rowidx = generate_4row_assign(pos_xsorted, size_xsorted, maxlen = max_col)

        # print('after assignment: ', len(filtered_id), len(size_xsorted), len(all_rowidx),len(filtered_id[all_rowidx]))
        rownum = len(row_assign)
        cur_maxlen = max(each_row_num) if len(each_row_num) > 0 else 0

        blk_img, blk_max_dim = cv2_transfer_coords_to_binary_mask(norm_utm_blk_poly, 64)

        # print(blk_max_dim, aspect_rto)
        # print('bldg_idx after filtering and sorting: ', np.array(bldg_id_list)[filtered_id-1])

        ### slice the building height from the bldg_poly_list by filtered_id
        bldg_height_list = []
        for ii in range(len(filtered_id)):
            bldg_height_list.append(clipped_bldg_poly_list[filtered_id[ii]-1][2])
        
        # print('bldg_height_list: ', bldg_height_list)
        # print('bldg_list: ', clipped_bldg_poly_list)
        
        graph_prep_dict['blk_id'] = blk_id
        graph_prep_dict['row_assign'] = row_assign
        graph_prep_dict['pos_xsorted'] = pos_xsorted
        graph_prep_dict['size_xsorted'] = size_xsorted
        graph_prep_dict['shape_xsorted'] = shape_xsorted
        graph_prep_dict['iou_xsorted'] = iou_xsorted
        graph_prep_dict['filtered_id'] = filtered_id
        graph_prep_dict['blk_aspect_rto'] = aspect_rto
        graph_prep_dict['medaxis'] = medaxis
        graph_prep_dict['blk_img'] = blk_img
        graph_prep_dict['blk_max_dim'] = blk_max_dim
        graph_prep_dict['blk_longside_scale'] = np.double(medaxis.length) / np.double(longside_scale)
        graph_prep_dict['bldg_id_list'] = np.array(bldg_id_list)[filtered_id-1]
        graph_prep_dict['bldg_poly_list'] = clipped_bldg_poly_list
        graph_prep_dict['bldg_num'] = bldg_num
        graph_prep_dict['height_xsorted'] = bldg_height_list

        graph_prep_dict['blk_recover_poly'] = norm_utm_blk_poly
        graph_prep_dict['blk_TIGER_properties'] = our_rows[i]['blk_TIGER_properties']
        graph_prep_dict['blk_utm_geometry'] = our_rows[i]['blk_utm_geometry']
        graph_prep_dict['blk_utm_crs'] = our_rows[i]['blk_utm_crs']

        graph_prep_list.append(graph_prep_dict)

        # print(blk_id, 'blk_id info: ', aspect_rto, pos_xsorted, size_xsorted)
        # org_bldg,_,_ = inverse_warp_bldg_by_midaxis(pos_xsorted.copy(), size_xsorted.copy(), medaxis, aspect_rto)
        # plt.plot(*norm_utm_blk_poly.exterior.xy, color = 'black')
        # plt.plot(*medaxis.coords.xy, color = 'red')
        # for ii in range(len(org_bldg)):
        #     plt.plot(*org_bldg[ii].exterior.xy, color = 'blue')                
        # plt.savefig(os.path.join(output_vis_dir, str(blk_id) + '.jpg'))
        # plt.clf()       

        print('blk_id: ', blk_id, ' bldg_num: ', bldg_num)        

    return graph_prep_list
        


max_row = 4
max_col = 30
longside_scale = 400.0
max_bldg = max_row * max_col

num_cores = 64

if __name__ == "__main__":

    raw_geo_path = '/media/dummy1/he425/Hierarchy_Dataset/Our_Dataset'

    ### read dict_bbx from updated_bbx_list.json
    with open('/media/dummy1/he425/Hierarchy_Dataset/updated_bbx_list.json') as f:
        dict_bbx = json.load(f)
    
    city_idx = 0
    minn = 31
    maxx = 331  ####(0, 1 all not finished) (2, 3), (4 finished, left 5- 7), (8,15), (16-20 finished, 30)
    ### loop through the dict_bbx
    finished = [4, 16, 17, 18, 19, 20]
    for cityname in dict_bbx.keys():
        # cityname = 'Aurora[1]'
        city_idx += 1
        print('city_idx: ', city_idx, ' cityname: ', cityname)
        if city_idx < minn or city_idx > maxx or city_idx in finished:
            continue             

        ### load the list of building geometries and attributes within the bbx from pickle file
        with open(os.path.join(raw_geo_path, cityname, 'raw_geo', cityname + '_blk_bldg_3d_utm.pkl'), 'rb') as f:
            our_rows = pickle.load(f)

        ### output dir
        output_vis_dir = os.path.join(raw_geo_path, cityname, 'visual_check')    
        if not os.path.exists(output_vis_dir):
            os.makedirs(output_vis_dir)
        
        ### output dir
        output_graph_dir = os.path.join(raw_geo_path, cityname, 'graph')    
        if not os.path.exists(output_graph_dir):
            os.makedirs(output_graph_dir)
        
        # total_blk_num = len(our_rows)
        # blk_bldg_num_list = []
        # overmax_blk_bldg_num_list = []

        graph_prep_list = []
        ### get the max bldg num

        raw_blk_num = len(our_rows)  ### len(our_rows)
        chunk_size = math.ceil(raw_blk_num / num_cores)
        chunks = [our_rows[i:i + chunk_size] for i in range(0, raw_blk_num, chunk_size)]

        print(len(chunks))
        count_procs = 0
        # Use multiprocessing to process each chunk in parallel
        with Pool(num_cores) as pool:
            # results = pool.imap_unordered(multi_process, chunks)
            for i, result in enumerate(pool.imap_unordered(process, chunks), 1):
                graph_prep_list.extend(result)
                count_procs += 1 
                print('processed: ', count_procs, ' chunks')

        # print(len(graph_prep_list))

            
        #### save the graph_prep_list to pickle file
        with open(os.path.join(output_graph_dir, cityname + '_graph_prep_list.pkl'), 'wb') as f:
            pickle.dump(graph_prep_list, f)

        
        # ### load the graph_prep_list from pickle file
        # with open(os.path.join(output_graph_dir, cityname + '_graph_prep_list.pkl'), 'rb') as f:    
        #     graph_prep_list = pickle.load(f)
        
        # #### generate the graph
        # graph_list = []

        # for i in range(100):
        #     if graph_prep_list[i]['bldg_num'] == 0:
        #         continue
        #     # print(graph_prep_list[i])
        #     g = geoarray_to_dense_4row_grid_addheight(graph_prep_list[i], max_col, max_row)
        #     nx.write_gpickle(g, os.path.join(output_graph_dir, 'test', str(graph_prep_list[i]['blk_id']) + ".gpickle"), 4)




    
        








