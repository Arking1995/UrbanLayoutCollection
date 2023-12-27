import pandas as pd
import geopandas as gpd
import shapely.geometry
import mercantile
from tqdm import tqdm
import os
import tempfile
import fiona
from sklearn.neighbors import KDTree, BallTree
import pickle
import json

#### a function that compare the size of two files in string format
def compare_size(str1, str2):
    ### remove the last character 'B' in the string
    str1 = str1[:-1]
    str2 = str2[:-1]

    if str1[-1] == 'K':
        size1 = float(str1[:-1]) * 1024
    elif str1[-1] == 'M':
        size1 = float(str1[:-1]) * 1024 * 1024
    elif str1[-1] == 'G':
        size1 = float(str1[:-1]) * 1024 * 1024 * 1024
    else:
        size1 = float(str1)
    if str2[-1] == 'K':
        size2 = float(str2[:-1]) * 1024
    elif str2[-1] == 'M':
        size2 = float(str2[:-1]) * 1024 * 1024
    elif str2[-1] == 'G':
        size2 = float(str2[:-1]) * 1024 * 1024 * 1024
    else:
        size2 = float(str2)
    if size1 > size2:
        return 1
    elif size1 < size2:
        return 2
    else:
        return 0



### transfer north, south, east, west to four coordinates of a rectangle, and the final coordinate the same as the first one
def transfer_to_rectangle(north, south, east, west):
    return [[west, north], [east, north], [east, south], [west, south], [west, north]]





if __name__ == "__main__":

    # ###### choose the city
    # city = 'chicago'

    ### read dict_bbx from updated_bbx_list.json
    with open('/home/he425/Dataset/updated_bbx_list.json') as f:
        dict_bbx = json.load(f)

    city_idx = 0
    ### loop through the dict_bbx
    for city in tqdm(dict_bbx.keys()):
        # if city == 'Chicago':
        #     continue
        ### skip the first 7 cities

        city_idx += 1
        print('Start processing: ', city, ' city_idx: ', city_idx, ' out of 331 cities')
        if city_idx != 8:
            # print('Skip the first 72 cities')
            continue

        ### mkdir for the city if not exist
        output_path = '/home/he425/Dataset/MS_Building_Dataset'
        if not os.path.exists(os.path.join(output_path, city)):
            os.mkdir(os.path.join(output_path, city))

        ### get the bounding box of the city
        bbx = dict_bbx[city]

        #### get the coordinates of the rectangle by function transfer_to_rectangle
        aoi_geom = {
            "coordinates": [
                transfer_to_rectangle(bbx[0], bbx[1], bbx[2], bbx[3])
            ],
            "type": "Polygon",
        }

        aoi_shape = shapely.geometry.shape(aoi_geom)
        minx, miny, maxx, maxy = aoi_shape.bounds

        output_path = '/home/he425/Dataset/MS_Building_Dataset'
        output_fn = os.path.join(output_path, city, city + "_bldg.geojson")

        quad_keys = set()
        for tile in list(mercantile.tiles(minx, miny, maxx, maxy, zooms=9)):
            quad_keys.add(int(mercantile.quadkey(tile)))
        quad_keys = list(quad_keys)
        print(f"The input area spans {len(quad_keys)} tiles: {quad_keys}")


        df = pd.read_csv(
            "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
        )

        idx = 0
        combined_rows = []
        Negative_height_list = []
        bldg_loc_list = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download the GeoJSON files for each tile that intersects the input geometry
            tmp_fns = []
            tmp_height_fns = []
            for quad_key in tqdm(quad_keys):
                rows = df[df["QuadKey"] == quad_key]
                if rows.shape[0] == 1:
                    url = rows.iloc[0]["Url"]

                    df2 = pd.read_json(url, lines=True)
                    df2["geometry"] = df2["geometry"].apply(shapely.geometry.shape)

                    gdf = gpd.GeoDataFrame(df2, crs=4326)
                    fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                    tmp_fns.append(fn)
                    if not os.path.exists(fn):
                        gdf.to_file(fn, driver="GeoJSON")
                elif rows.shape[0] > 1:
                    print('Total number of tile files found: ', rows.shape[0]) 
                    ## print Size of each file
                    for i in range(rows.shape[0]):
                        print(i, rows.iloc[i]['Size'])
                    ### choose the file with larger size in row['Size']
                    Size = '0.0B'
                    for i in range(rows.shape[0]):
                        if compare_size(rows.iloc[i]['Size'], str(Size)) == 1 or compare_size(rows.iloc[i]['Size'], str(Size)) == 0:
                            Size = rows.iloc[i]['Size']
                            url = rows.iloc[i]["Url"]

                    df2 = pd.read_json(url, lines=True)
                    # print(df2)
                    # df2['height'] = df2['height'].astype(float)
                    df2["geometry"] = df2["geometry"].apply(shapely.geometry.shape)

                    gdf = gpd.GeoDataFrame(df2, crs=4326)
                    fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                    tmp_fns.append(fn)
                    if not os.path.exists(fn):
                        gdf.to_file(fn, driver="GeoJSON")

                    print(f"Multiple rows found for QuadKey: {quad_key}")            
                    # raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
                else:
                    raise ValueError(f"QuadKey not found in dataset: {quad_key}")

            # Merge the GeoJSON files into a single file
            for fn in tmp_fns:
                with fiona.open(fn, "r") as f:
                    for row in tqdm(f):
                        row = dict(row)
                        # print(idx, row)
                        # print(row["properties"]['properties'])
                        shape = shapely.geometry.shape(row["geometry"])

                        if aoi_shape.contains(shape):
                            if "id" in row:
                                del row["id"]
                            
                            ### get the location of each building center
                            # bldg_loc_list.append([row["geometry"]["coordinates"][0][0][0], row["geometry"]["coordinates"][0][0][1]])
                            bldg_loc_list.append([shape.centroid.x, shape.centroid.y])

                            # row["height"] = row["properties"]['properties']["height"]
                            if row["properties"]['properties']["height"] < 0.0:
                                # print('Negative height found: ', row["properties"]['properties']["height"], ' at: ', idx)
                                Negative_height_list.append(idx)

                            row["properties"] = {"id": idx, "height": row["properties"]['properties']["height"]}
                            idx += 1
                            combined_rows.append(row)

        schema = {"geometry": "Polygon", "properties": {"id": "int", "height": "float"}}

        with fiona.open(output_fn, "w", driver="GeoJSON", crs="EPSG:4326", schema=schema) as f:
            f.writerecords(combined_rows)
        
        print('Total number of buildings: ', idx)

        ### save the negative height list to a pickle file
        with open(os.path.join(output_path, city, city + '_negative_height_list.pkl'), 'wb') as f:
            pickle.dump(Negative_height_list, f)
        
        ### save the building location list to a pickle file
        with open(os.path.join(output_path, city, city + '_bldg_loc_list.pkl'), 'wb') as f:
            pickle.dump(bldg_loc_list, f)
        
        # ### build the location ball tree of all buildings
        # bldg_loc_balltree = BallTree(bldg_loc_list, metric='haversine')

        # ### save the location ball tree of all buildings to a pickle file
        # with open(os.path.join(output_path, city + '_bldg_loc_ball_tree.pkl'), 'wb') as f:
        #     pickle.dump(bldg_loc_balltree, f)
        
