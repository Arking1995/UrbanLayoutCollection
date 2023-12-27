# Urban layout collection and preprocessing
To collect urban layouts from Microsoft Building Footprints and OpenStreetMap, and preprocess them into hierarchy data structure.

## Environment
```
osmnx==1.1.2
tqdm
matplotlib
fiona
geopandas
shapely
```
and [utm](https://github.com/Turbo87/utm) package for coordinate convertion from UTM-WGS84

## User Instruction
The codes are not initially prepared as public APIs/libraries, but feel free to make your own branchs.

Download and clip urban layouts from MS building footprints:
```
python ms_bbx_clip.py 
```

Download and clip urban layouts from OpenStreetMap:
```
python osm_bbx_clip.py
```

Make hierarchy city block and building layout data structure:
```
python blk_bldg_assign.py
```

Multiprocess canonical transformation of building layouts (CPU heavy):
```
python multiprocess_canonical_transform.py
```
