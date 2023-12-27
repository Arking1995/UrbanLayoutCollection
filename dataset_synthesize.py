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

from graph_util import geoarray_to_dense_4row_grid_addheight
from geometry_utils import inverse_warp_bldg_by_midaxis


# if __name__ == '__main__':

#     raw_geo_path = '/media/dummy1/he425/Hierarchy_Dataset/Our_Dataset'
#     prep_geo_path = '/media/dummy1/he425/Hierarchy_Dataset/Prep_Dataset'

#     ### read dict_bbx from updated_bbx_list.json
#     with open('/media/dummy1/he425/Hierarchy_Dataset/updated_bbx_list.json') as f:
#         dict_bbx = json.load(f)
    
#     city_idx = 0
#     ### loop through the dict_bbx
#     pos_list = []
#     size_list = []
#     height_list = []
#     blk_count = 0
#     blk_with_bldg_count = 0
#     for cityname in dict_bbx.keys():
#         city_idx += 1
#         print('city_idx: ', city_idx, ' cityname: ', cityname)
#         # if city_idx > 1:
#         #     break

#         ### load the graph_prep_list from pickle file
#         with open(os.path.join(raw_geo_path, cityname, 'graph', cityname + '_graph_prep_list.pkl'), 'rb') as f:    
#             graph_prep_list = pickle.load(f)
        
#         #### generate the graph
#         graph_list = []
#         total_blk_num = len(graph_prep_list)
#         for i in range(total_blk_num):
#             if i % 1000 == 0:
#                 print('i: ', i)
#             blk_count += 1

#             if graph_prep_list[i]['bldg_num'] == 0:
#                 continue
            
#             blk_with_bldg_count += 1

#             c_size = graph_prep_list[i]['size_xsorted']
#             ### clip the value of 2nd column in c_size > 3 to 3
#             c_size = np.clip(c_size, a_min=-5, a_max=5)

#             c_pos = graph_prep_list[i]['pos_xsorted']
#             ### clip the value of 2nd column in c_size > 3 to 3
#             c_pos = np.clip(c_pos, a_min=-5, a_max=5)

#             c_height = graph_prep_list[i]['height_xsorted']
#             ### clip the value of 2nd column in c_size > 3 to 3
#             c_height = np.clip(c_height, a_min=0.0, a_max=None)

#             pos_list.append(c_pos)
#             size_list.append(c_size)
#             height_list.append(c_height)

    
#     pos_arr = np.concatenate(pos_list, axis=0)
#     size_arr = np.concatenate(size_list, axis=0)
#     height_arr = np.concatenate(height_list, axis=0)

#     print('pos_arr.shape: ', pos_arr.shape)
#     print('size_arr.shape: ', size_arr.shape)
#     print('height_arr.shape: ', height_arr.shape)
#     print('Total number of blocks: ', blk_count)
#     print('Total number of blocks with buildings: ', blk_with_bldg_count)

#     ### save the [pos_arr, size_arr, height_arr] to a single pickle file
#     with open(os.path.join(prep_geo_path, 'pos_size_height.pkl'), 'wb') as f:
#         pickle.dump([pos_arr, size_arr, height_arr], f)
    

#     ### load the [pos_arr, size_arr, height_arr] from a single pickle file
#     with open(os.path.join(prep_geo_path, 'pos_size_height.pkl'), 'rb') as f:
#         pos_arr, size_arr, height_arr = pickle.load(f)
    
#     ### get the mean, std  of pos_arr, size_arr, height_arr, save to a single pickle file
#     pos_mean = np.mean(pos_arr, axis=0)
#     pos_std = np.std(pos_arr, axis=0)
#     size_mean = np.mean(size_arr, axis=0)
#     size_std = np.std(size_arr, axis=0)
#     height_mean = np.mean(height_arr, axis=0)
#     height_std = np.std(height_arr, axis=0)

#     # print('pos_max: ', np.amax(pos_arr, axis=0))
#     # print('pos_min: ', np.amin(pos_arr, axis=0))

#     # print('size_max: ', np.amax(size_arr, axis=0))
#     # print('size_min: ', np.amin(size_arr, axis=0))

#     # print('height_max: ', np.amax(height_arr, axis=0))
#     # print('height_min: ', np.amin(height_arr, axis=0))

#     # print('pos_mean: ', pos_mean)
#     # print('pos_std: ', pos_std)
#     # print('size_mean: ', size_mean)
#     # print('size_std: ', size_std)
#     # print('height_mean: ', height_mean)
#     # print('height_std: ', height_std)
    
#     # #### plot the histogram of pos_arr, size_arr, height_arr in a single figure
#     # fig, axs = plt.subplots(3, 2, figsize=(10, 10))
#     # axs[0, 0].hist(pos_arr[:,0], bins=100)
#     # axs[0, 0].set_title('pos_x')
#     # axs[0, 1].hist(pos_arr[:,1], bins=100)
#     # axs[0, 1].set_title('pos_y')
#     # axs[1, 0].hist(size_arr[:,0], bins=100)
#     # axs[1, 0].set_title('size_x')
#     # axs[1, 1].hist(size_arr[:,1], bins=100)
#     # axs[1, 1].set_title('size_y')
#     # axs[2, 0].hist(height_arr, bins=100)
#     # axs[2, 0].set_title('height')
#     # plt.savefig(os.path.join(prep_geo_path, 'pos_size_height_hist.jpg'))
#     # plt.clf()

#     ### save the [pos_mean, pos_std, size_mean, size_std, height_mean, height_std] to a single pickle file
#     with open(os.path.join(prep_geo_path, 'pos_size_height_mean_std.pkl'), 'wb') as f:
#         pickle.dump([pos_mean, pos_std, size_mean, size_std, height_mean, height_std], f)
    


def get_z_norm_para(d3filename, output_filename):
    ### load the [pos_arr, size_arr, height_arr] from a single pickle file
    with open(os.path.join(d3filename), 'rb') as f:
        pos_arr, size_arr, height_arr = pickle.load(f)
    
    ### get the mean, std  of pos_arr, size_arr, height_arr, save to a single pickle file
    pos_mean = np.mean(pos_arr, axis=0)
    pos_std = np.std(pos_arr, axis=0)
    size_mean = np.mean(size_arr, axis=0)
    size_std = np.std(size_arr, axis=0)
    height_mean = np.mean(height_arr, axis=0)
    height_std = np.std(height_arr, axis=0)

    ### save the [pos_mean, pos_std, size_mean, size_std, height_mean, height_std] to a single pickle file
    with open(os.path.join(output_filename), 'wb') as f:
        pickle.dump([pos_mean, pos_std, size_mean, size_std, height_mean, height_std], f)



def get_3d_file(raw_geo_path, bbx_path, output_path):
    raw_geo_path = '/media/dummy1/he425/Hierarchy_Dataset/Our_Dataset'

    ### read dict_bbx from updated_bbx_list.json
    with open(bbx_path) as f:
        dict_bbx = json.load(f)
    
    city_idx = 0
    ### loop through the dict_bbx
    pos_list = []
    size_list = []
    height_list = []
    blk_count = 0
    blk_with_bldg_count = 0
    for cityname in dict_bbx.keys():
        city_idx += 1
        print('city_idx: ', city_idx, ' cityname: ', cityname)
        # if city_idx > 1:
        #     break

        ### load the graph_prep_list from pickle file
        with open(os.path.join(raw_geo_path, cityname, 'graph', cityname + '_graph_prep_list.pkl'), 'rb') as f:    
            graph_prep_list = pickle.load(f)
        
        #### generate the graph
        graph_list = []
        total_blk_num = len(graph_prep_list)
        for i in range(total_blk_num):
            if i % 1000 == 0:
                print('i: ', i)
            blk_count += 1

            if graph_prep_list[i]['bldg_num'] == 0:
                continue
            
            blk_with_bldg_count += 1

            c_size = graph_prep_list[i]['size_xsorted']
            ### clip the value of 2nd column in c_size > 3 to 3
            c_size = np.clip(c_size, a_min=-5, a_max=5)

            c_pos = graph_prep_list[i]['pos_xsorted']
            ### clip the value of 2nd column in c_size > 3 to 3
            c_pos = np.clip(c_pos, a_min=-5, a_max=5)

            c_height = graph_prep_list[i]['height_xsorted']
            ### clip the value of 2nd column in c_size > 3 to 3
            c_height = np.clip(c_height, a_min=0.0, a_max=None)

            pos_list.append(c_pos)
            size_list.append(c_size)
            height_list.append(c_height)

    
    pos_arr = np.concatenate(pos_list, axis=0)
    size_arr = np.concatenate(size_list, axis=0)
    height_arr = np.concatenate(height_list, axis=0)

    print('pos_arr.shape: ', pos_arr.shape)
    print('size_arr.shape: ', size_arr.shape)
    print('height_arr.shape: ', height_arr.shape)
    print('Total number of blocks: ', blk_count)
    print('Total number of blocks with buildings: ', blk_with_bldg_count)

    ### save the [pos_arr, size_arr, height_arr] to a single pickle file
    with open(os.path.join(output_path), 'wb') as f:
        pickle.dump([pos_arr, size_arr, height_arr], f)







def norm1_prepreocess(znorm_para, preprocess_para, c_size, c_pos, c_height):
    pos_mean, pos_std, size_mean, size_std, height_mean, height_std = znorm_para
    hgt_bases = np.log(20)

    [w_0, w_1], [ws_0, ws_1, sz_min_log, sz_max_log], [wh, hg_min, hg_max] = preprocess_para
    c_pos = (c_pos - pos_mean) / pos_std
    c_pos[:, 0] = (c_pos[:, 0] + w_0 * 0.5)/ w_0
    c_pos[:, 1] = (c_pos[:, 1] + w_1 * 0.5)/ w_1
    c_pos = np.clip(c_pos, a_min=0.0, a_max=1.0)

    c_size = np.clip(c_size, a_min=0.0, a_max=None)
    c_size = (c_size - size_mean) / size_std
    c_size[:, 0] = (c_size[:, 0] + 0.5 * ws_0) / ws_0
    c_size[:, 1] = (c_size[:, 1] + 0.5 * ws_1) / ws_1
    ## curve the histogram of size_arr[:, 0] and size_arr[:, 1] to more uniform
    epsilon = 1e-10
    c_size = c_size + epsilon
    log_transformed = np.log(c_size)
    c_size = (log_transformed - sz_min_log) / (sz_max_log - sz_min_log)

    mask = c_height > 3.0 * height_std + height_mean
    mask_re = np.logical_not(mask)
    c_height[mask_re] = (c_height[mask_re] - height_mean) / height_std
    c_height[mask] = np.log(c_height[mask]) / hgt_bases * 0.5 + 3.0

    c_height = (c_height + 0.5 * wh) / wh
    c_height = (c_height - hg_min) / (hg_max - hg_min)

    return c_size, c_pos, c_height


### inverse norm1_prepreocess
def inverse_norm1_preprocess(znorm_para, preprocess_para, c_size, c_pos, c_height):
    pos_mean, pos_std, size_mean, size_std, height_mean, height_std = znorm_para
    hgt_bases = np.log(20)
    [w_0, w_1], [ws_0, ws_1, sz_min_log, sz_max_log], [wh, hg_min, hg_max] = preprocess_para

    c_pos[:, 0] = c_pos[:, 0] * w_0 - w_0 * 0.5
    c_pos[:, 1] = c_pos[:, 1] * w_1 - w_1 * 0.5
    c_pos = c_pos * pos_std + pos_mean

    c_size = c_size * (sz_max_log - sz_min_log) + sz_min_log
    c_size = np.exp(c_size)
    c_size[:, 0] = c_size[:, 0] * ws_0 - 0.5 * ws_0
    c_size[:, 1] = c_size[:, 1] * ws_1 - 0.5 * ws_1
    c_size = c_size * size_std + size_mean

    c_height = c_height * (hg_max - hg_min) + hg_min
    c_height = c_height * wh - 0.5 * wh
    mask = c_height > 3.0
    mask_re = np.logical_not(mask)
    c_height[mask_re] = c_height[mask_re] * height_std + height_mean
    c_height[mask] = np.exp((c_height[mask] - 3.0) * hgt_bases / 0.5)

    return c_size, c_pos, c_height
    


def get_preprocess_para(d3filename, z_norm_para_filename, output_filename):
    ### load the [pos_arr, size_arr, height_arr] from a single pickle file
    with open(os.path.join(d3filename), 'rb') as f:
        pos_arr, size_arr, height_arr = pickle.load(f)

    ### load the [pos_mean, pos_std, size_mean, size_std, height_mean, height_std] from a single pickle file
    with open(os.path.join(prep_geo_path, z_norm_para_filename), 'rb') as f:
        pos_mean, pos_std, size_mean, size_std, height_mean, height_std = pickle.load(f)    

    ### norm pos_arr by mean and std, and scale to [0, 1]
    pos_arr = (pos_arr - pos_mean) / pos_std
    w_0 = np.percentile(pos_arr[:, 0], 99.9) - np.percentile(pos_arr[:, 0], 0.1)
    w_1 = np.percentile(pos_arr[:, 1], 99.9) - np.percentile(pos_arr[:, 1], 0.1)
    pos_arr[:, 0] = (pos_arr[:, 0] + w_0 * 0.5)/ w_0
    pos_arr[:, 1] = (pos_arr[:, 1] + w_1 * 0.5)/ w_1
    pos_arr = np.clip(pos_arr, a_min=0.0, a_max=1.0)
    print('max and min and mean of pos_arr: ', np.amax(pos_arr, axis=0), np.amin(pos_arr, axis=0), np.mean(pos_arr, axis=0), np.std(pos_arr, axis=0))

    ### norm size_arr by mean and std, and scale to [0, 1]
    # Apply logarithmic transformation
    size_arr = np.clip(size_arr, a_min=0.0, a_max=None)
    print('max and min and mean of size_arr: ', np.amax(size_arr, axis=0), np.amin(size_arr, axis=0), np.mean(size_arr, axis=0), np.std(size_arr, axis=0))
    size_arr = (size_arr - size_mean) / size_std
    ws_0 = np.percentile(size_arr[:, 0], 99.9) - np.percentile(size_arr[:, 0], 0.1)
    ws_1 = np.percentile(size_arr[:, 1], 99.9) - np.percentile(size_arr[:, 1], 0.1)
    size_arr[:, 0] = (size_arr[:, 0] + 0.5 * ws_0) / ws_0
    size_arr[:, 1] = (size_arr[:, 1] + 0.5 * ws_1) / ws_1
    ## curve the histogram of size_arr[:, 0] and size_arr[:, 1] to more uniform
    epsilon = 1e-10
    size_arr = size_arr + epsilon
    log_transformed = np.log(size_arr)
    sz_min_log = np.amin(log_transformed, axis=0)
    sz_max_log = np.amax(log_transformed, axis=0)
    # print('max and min and mean of log_transformed: ', np.amax(log_transformed, axis=0), np.amin(log_transformed, axis=0), np.mean(log_transformed, axis=0), np.std(log_transformed, axis=0))
    size_arr = (log_transformed - sz_min_log) / (sz_max_log - sz_min_log)
    print('max and min and mean of size_arr: ', np.amax(size_arr, axis=0), np.amin(size_arr, axis=0), np.mean(size_arr, axis=0), np.std(size_arr, axis=0))


    ### norm height_arr by log(20)
    bases = np.log(20)
    # height_arr[height_arr>0.0] = np.log(height_arr[height_arr>0.0]) / bases * 0.8
    ### mask the height_arr with height_arr > 3* std + mean
    mask = height_arr > 3.0 * height_std + height_mean
    mask_re = np.logical_not(mask)
    ### normalize the height_arr with height mean and std
    height_arr[mask_re] = (height_arr[mask_re] - height_mean) / height_std
    # print('max and min and mean of masked height_arr: ', np.amax(height_arr[mask_re]), np.amin(height_arr[mask_re]), np.mean(height_arr[mask_re])  )
    ### take log of the height_arr with height_arr > 3* std + mean
    height_arr[mask] = np.log(height_arr[mask]) / bases * 0.5 + 3.0
    print('max and min and mean of masked height_arr: ', np.amax(height_arr[mask]), np.amin(height_arr[mask]), np.mean(height_arr[mask]))

    ### normalize height to [0, 1]
    wh = np.percentile(height_arr, 99.9) - np.percentile(height_arr, 0.1)
    height_arr = (height_arr + 0.5 * wh) / wh
    print('wh : ', wh)
    print('max and min and mean of height_arr: ', np.amax(height_arr, axis=0), np.amin(height_arr, axis=0), np.mean(height_arr, axis=0), np.std(height_arr, axis=0))

    #### scale the height_arr to [0, 1]
    hg_min = np.amin(height_arr, axis=0)
    hg_max = np.amax(height_arr, axis=0)
    height_arr = (height_arr - hg_min) / (hg_max - hg_min)
    print('max and min and mean of height_arr: ', np.amax(height_arr, axis=0), np.amin(height_arr, axis=0), np.mean(height_arr, axis=0), np.std(height_arr, axis=0))


    # #### plot the histogram of pos_arr, size_arr, height_arr in a single figure
    # fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    # axs[0, 0].hist(pos_arr[:,0], bins=100)
    # axs[0, 0].set_title('pos_x')
    # axs[0, 1].hist(pos_arr[:,1], bins=100)
    # axs[0, 1].set_title('pos_y')
    # axs[1, 0].hist(size_arr[:,0], bins=100)
    # axs[1, 0].set_title('size_x')
    # axs[1, 1].hist(size_arr[:,1], bins=100)
    # axs[1, 1].set_title('size_y')
    # axs[2, 0].hist(height_arr, bins=100)
    # axs[2, 0].set_title('height')
    # plt.savefig(os.path.join(prep_geo_path, 'pos_size_height_hist.jpg'))
    # plt.clf()

    ### save the w_0, w_1, ws_0, ws_1 to a single pickle file
    with open(os.path.join(output_filename), 'wb') as f:
        pickle.dump([[w_0, w_1], [ws_0, ws_1, sz_min_log, sz_max_log], [wh, hg_min, hg_max]], f)



def z_norm_logheight(c_size, c_pos, c_height, z_norm_para):
    pos_mean, pos_std, size_mean, size_std, height_mean, height_std = z_norm_para
    bases = np.log(20)
    ### normalize size by size mean and std
    c_size = (c_size - size_mean) / size_std
    ### clip the value of 2nd column in c_size > 3 to 3
    c_size = np.clip(c_size, a_min=-5, a_max=5)

    ### normalize pos by pos mean and std
    c_pos = (c_pos - pos_mean) / pos_std
    ### clip the value of 2nd column in c_size > 3 to 3
    c_pos = np.clip(c_pos, a_min=-5, a_max=5)

    ### clip negative height to 0
    c_height[c_height<0.0] = 1.0
    ### normalize the rest of the height by height mean and std
    c_height = np.log(c_height) / bases * 0.8

    return c_size, c_pos, c_height


### inverse version of z_norm_logheight
def inverse_z_norm_logheight(c_size, c_pos, c_height, z_norm_para):
    pos_mean, pos_std, size_mean, size_std, height_mean, height_std = z_norm_para
    bases = np.log(20)
    ### inverse normalize size by size mean and std
    c_size = c_size * size_std + size_mean
    ### inverse normalize pos by pos mean and std
    c_pos = c_pos * pos_std + pos_mean
    ### inverse normalize height by height mean and std
    c_height = np.exp(c_height * bases / 0.8)

    return c_size, c_pos, c_height




def z_norm_normheight(c_size, c_pos, c_height, z_norm_para):
    pos_mean, pos_std, size_mean, size_std, height_mean, height_std = z_norm_para
    bases = np.log(20)
    ### normalize size by size mean and std
    c_size = (c_size - size_mean) / size_std
    ### clip the value of 2nd column in c_size > 3 to 3
    c_size = np.clip(c_size, a_min=-5, a_max=5)

    ### normalize pos by pos mean and std
    c_pos = (c_pos - pos_mean) / pos_std
    ### clip the value of 2nd column in c_size > 3 to 3
    c_pos = np.clip(c_pos, a_min=-5, a_max=5)

    ### clip negative height to 0
    c_height[c_height<0.0] = 0.0
    mask = c_height > 3.0 * height_std + height_mean
    mask_re = np.logical_not(mask)
    c_height[mask_re] = (c_height[mask_re] - height_mean) / height_std
    ### take log of the c_height with c_height > 3* std + mean
    c_height[mask] = np.log(c_height[mask]) / bases * 0.5 + 3.0

    return c_size, c_pos, c_height


##### inverse version of z_norm_normheight
def inverse_z_norm_normheight(c_size, c_pos, c_height, z_norm_para):
    pos_mean, pos_std, size_mean, size_std, height_mean, height_std = z_norm_para
    bases = np.log(20)
    ### inverse normalize size by size mean and std
    c_size = c_size * size_std + size_mean
    ### inverse normalize pos by pos mean and std
    c_pos = c_pos * pos_std + pos_mean

    mask = c_height > 3.0
    mask_re = np.logical_not(mask)
    ### inverse normalize height by height mean and std
    c_height[mask_re] = c_height[mask_re] * height_std + height_mean
    ### inverse take log of the c_height with c_height > 3
    c_height[mask] = np.exp((c_height[mask] - 3.0) * bases / 0.5)

    return c_size, c_pos, c_height





if __name__ == '__main__':

    raw_geo_path = '/media/dummy1/he425/Hierarchy_Dataset/Our_Dataset'
    bbx_path = '/media/dummy1/he425/Hierarchy_Dataset/updated_bbx_list.json'

    prep_geo_path = '/media/dummy1/he425/Hierarchy_Dataset/Prep_Dataset_normheight_new'
    output_vis_dir = '/media/dummy1/he425/Hierarchy_Dataset/test_vis'

    #### mkdir for prep_geo_path
    if not os.path.exists(prep_geo_path):
        os.mkdir(prep_geo_path)

    d3_path = os.path.join('/media/dummy1/he425/Hierarchy_Dataset/Prep_Dataset_logheight/pos_size_height.pkl')
    z_norm_path = os.path.join(prep_geo_path, 'pos_size_height_mean_std.pkl')

    # get_3d_file(raw_geo_path, bbx_path, d3_path)
    # get_z_norm_para(d3_path, z_norm_path)    


    ### read dict_bbx from updated_bbx_list.json
    with open(bbx_path) as f:
        dict_bbx = json.load(f)
    
    ### mkdir for processed graph
    output_graph_dir = os.path.join(prep_geo_path, 'processed_all')
    if not os.path.exists(output_graph_dir):
        os.mkdir(output_graph_dir)

    ### mkdir for processed graph
    output_bldg_graph_dir = os.path.join(prep_geo_path, 'processed_with_bldg')
    if not os.path.exists(output_bldg_graph_dir):
        os.mkdir(output_bldg_graph_dir)

    ### load the [pos_mean, pos_std, size_mean, size_std, height_mean, height_std] from a single pickle file
    with open(os.path.join(z_norm_path), 'rb') as f:
        z_norm_para = pickle.load(f)

    print(z_norm_para)

    max_col = 30
    max_row = 4
    bases = np.log(20)
    city_idx = 0
    uni_blk_id = 0
    uni_blk_with_bldg_id = 0
    total_bldg_num = 0
    ### loop through the dict_bbx
    for cityname in tqdm(dict_bbx.keys()):
        city_idx += 1
        # print('city_idx: ', city_idx, ' cityname: ', cityname)
        # if city_idx != 50:
        #     continue

        ### load the graph_prep_list from pickle file
        with open(os.path.join(raw_geo_path, cityname, 'graph', cityname + '_graph_prep_list.pkl'), 'rb') as f:    
            graph_prep_list = pickle.load(f)
        
        #### generate the graph
        graph_list = []
        total_blk_num = len(graph_prep_list)
        for i in tqdm(range(total_blk_num)): #range(total_blk_num)
            # if i % 1000 == 0:
            #     print('i: ', i)

            if graph_prep_list[i]['bldg_num'] != 0:

                total_bldg_num += graph_prep_list[i]['bldg_num']

                # norm_utm_blk_poly = graph_prep_list[i]['blk_recover_poly']
                # medaxis = graph_prep_list[i]['medaxis']
                # # print(graph_prep_list[i]['blk_id'], graph_prep_list[i]['pos_xsorted'], graph_prep_list[i]['size_xsorted'])
                # org_bldg,_,_ = inverse_warp_bldg_by_midaxis(graph_prep_list[i]['pos_xsorted'].copy(), graph_prep_list[i]['size_xsorted'].copy(), medaxis, graph_prep_list[i]['blk_aspect_rto'])
                # plt.plot(*norm_utm_blk_poly.exterior.xy, color = 'black')
                # plt.plot(*medaxis.coords.xy, color = 'red')
                # for ii in range(len(org_bldg)):
                #     plt.plot(*org_bldg[ii].exterior.xy, color = 'blue')                
                # plt.savefig(os.path.join(output_vis_dir, str(uni_blk_with_bldg_id) + '.jpg'))
                # plt.clf()       

                c_size = graph_prep_list[i]['size_xsorted']
                c_height = np.array(graph_prep_list[i]['height_xsorted'])
                c_pos = graph_prep_list[i]['pos_xsorted']

                # norm_c_size, norm_c_pos, norm_c_height = norm1_prepreocess(z_norm_para, preprocess_para, c_size.copy(), c_pos.copy(), c_height.copy())
                # norm_c_size, norm_c_pos, norm_c_height = z_norm_logheight(c_size.copy(), c_pos.copy(), c_height.copy(), z_norm_para)
                norm_c_size, norm_c_pos, norm_c_height = z_norm_normheight(c_size.copy(), c_pos.copy(), c_height.copy(), z_norm_para)
                # pos_mean, pos_std, size_mean, size_std, height_mean, height_std = z_norm_para

                # norm_c_size = (c_size - size_mean) / size_std
                # norm_c_size = np.clip(norm_c_size, a_min=-5, a_max=5)
                # norm_c_pos = (c_pos - pos_mean) / pos_std
                # norm_c_pos = np.clip(norm_c_pos, a_min=-5, a_max=5)
                # c_height[c_height<0.0] = 1.0
                # norm_c_height = np.log(c_height) / bases * 0.5
                # norm_c_height = np.clip(norm_c_height, a_min=0.0, a_max=None)
                #### check if any nan in norm_c_height, norm_c_size, norm_c_pos, if so, set them to 0
                norm_c_height[np.isnan(norm_c_height)] = 0.0
                norm_c_size[np.isnan(norm_c_size)] = 0.0
                norm_c_pos[np.isnan(norm_c_pos)] = 0.0

                # print('max and min and mean of norm_c_size: ', np.amax(norm_c_size, axis=0), np.amin(norm_c_size, axis=0), np.mean(norm_c_size, axis=0), np.std(norm_c_size, axis=0))
                # print('max and min and mean of norm_c_pos: ', np.amax(norm_c_pos, axis=0), np.amin(norm_c_pos, axis=0), np.mean(norm_c_pos, axis=0), np.std(norm_c_pos, axis=0))
                # print('max and min and mean of norm_c_height: ', np.amax(norm_c_height, axis=0), np.amin(norm_c_height, axis=0), np.mean(norm_c_height, axis=0), np.std(norm_c_height, axis=0))
                # norm_c_size, norm_c_pos, norm_c_height = z_norm_normheight(c_size.copy(), c_pos.copy(), c_height.copy(), z_norm_para)

                # ### check the inverse version of z_norm_logheight
                # # c_size1, c_pos1, c_height1 = inverse_z_norm_logheight(norm_c_size.copy(), norm_c_pos.copy(), norm_c_height.copy(), z_norm_para)
                # c_size1, c_pos1, c_height1 = inverse_z_norm_normheight(norm_c_size.copy(), norm_c_pos.copy(), norm_c_height.copy(), z_norm_para)
                # print('difference between c_size and c_size1: ', np.amax(c_size - c_size1), np.amin(c_size - c_size1))
                # print('difference between c_pos and c_pos1: ', np.amax(c_pos - c_pos1), np.amin(c_pos - c_pos1))
                # print('difference between c_height and c_height1: ', np.amax(c_height - c_height1), np.amin(c_height - c_height1))
                

                graph_prep_list[i]['size_xsorted'] = norm_c_size
                graph_prep_list[i]['pos_xsorted'] = norm_c_pos
                graph_prep_list[i]['height_xsorted'] = norm_c_height
                graph_prep_list[i]['city_name'] = cityname                

                g = geoarray_to_dense_4row_grid_addheight(graph_prep_list[i], max_col, max_row, uni_blk_with_bldg_id)
                nx.write_gpickle(g, os.path.join(output_bldg_graph_dir, str(uni_blk_with_bldg_id) + ".gpickle"), 4)
                uni_blk_with_bldg_id += 1
                nx.write_gpickle(g, os.path.join(output_graph_dir, str(uni_blk_id) + ".gpickle"), 4)
                uni_blk_id += 1
            else:
                graph_prep_list[i]['city_name'] = cityname                
                g = geoarray_to_dense_4row_grid_addheight(graph_prep_list[i], max_col, max_row, uni_blk_id)
                nx.write_gpickle(g, os.path.join(output_graph_dir, str(uni_blk_id) + ".gpickle"), 4)
                uni_blk_id += 1


    print('uni_blk_id: ', uni_blk_id)
    print('uni_blk_with_bldg_id: ', uni_blk_with_bldg_id)
    print('total_bldg_num: ', total_bldg_num)


