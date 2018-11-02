import tensorflow as tf
import numpy as np
import cv2

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import shutil
import uuid
import copy
import os
import cv2
import json
import gc

from pathos.multiprocessing import ProcessingPool as Pool
pool = Pool(processes=4)

def pure_name(name):
    name = name[name.rfind('/') + 1 : name.rfind('.')]
    name = name.replace('_L', '')
    return name

from pathlib import Path
sys.path.append('/home/undead/reps/tf_models/object_detection/datasets/')

import navmii_dataset_utils as navmii_utils

custom_label_colours = [(0, 0, 0), (128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 void label, 1 = road, 2 = sidewalk, 3 = unlabeled
                ,(190, 153, 153), (153, 153, 153)
                # 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(69, 129, 180), (219, 19, 60)
                # 9 = sky, 10 = person
                ,(0, 0, 142), (0, 0, 69)
                # 11 = car, 12 = road mark
                ,(0, 0, 230)]
                # 13 = bike
                
autovision_label_colours = [(0,  0, 0), (58, 156, 147), (0, 177, 247), (94, 30, 104), (191, 119, 56), (182, 179, 182),
                            (102, 102, 102), (243, 15, 190), (230, 225, 54), (60, 112, 60), (146, 243, 146),
                            (122, 3, 51), (164, 216, 255), (250, 0, 55), (178, 20, 50), (0, 30, 130)]


# label_only_mark = [(0, 0, 0), (128, 64, 128), (244, 35, 231), (69, 69, 69)
#                 # 0 void label, 1 = road, 2 = sidewalk, 3 = building
#                 ,(250, 0, 0)]
#                 # 4 = road mark
label_only_mark = [(0, 0, 0), (128, 64, 128)
                # 0 void label, 1 = road
                ,(250, 0, 0)]
                # 2 = road mark

deeplab_label_colours = [(0, 0, 0), (128, 64, 128), (244, 35, 231), (69, 69, 69)
                # -1 void label, 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = road mark
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = truck, 16 = train, 17 = motocycle
                ,(119, 10, 32), (150, 50, 200)]
                # 18 = bicycle, 19 = tunnel


autovision_to_autovision = {
    'unlabeled' : 'unlabeled',
     'debris' : 'debris',
      'sky' : 'sky',
       'obstacle' : 'obstacle',
        'ground' : 'ground',
         'road' : 'road',
               'building' : 'building',
                'pipe' : 'pipe',
                 'building_material' : 'building_material',
                  'big_vegetation' : 'big_vegetation',
                   'small_vegetation' : 'small_vegetation',
               'manhole' : 'manhole', 'water' : 'water', 'person' : 'person', 'animal' : 'animal', 'vehicle' : 'vehicle'
}

camvid_to_city_map = {'Animal' : 'unlabeled',
'Archway' : 'building',
'Bicyclist' : 'bicycle',
'Bridge' : 'bridge',
'Building' : 'building',
'Car' : 'car',
'CartLuggagePram' : 'car',
'Child' : 'person',
'Column_Pole' : 'pole',
'Fence' : 'fence',
'LaneMkgsDriv' : 'rail track',
'LaneMkgsNonDriv' : 'rail track',
'Misc_Text' : 'traffic sign',
'MotorcycleScooter' : 'motocycle',
'OtherMoving' : 'motocycle',
'ParkingBlock' : 'parking',
'Pedestrian' : 'person',
'Road' : 'road',
'RoadShoulder' : 'building',
'Sidewalk' : 'sidewalk',
'SignSymbol' : 'traffic sign',
'Sky' : 'sky',
'SUVPickupTruck' : 'truck',
'TrafficCone' : 'pole',
'TrafficLight' : 'traffic light',
'Train' : 'train',
'Tree' : 'vegetation',
'Truck_Bus' : 'truck',
'Tunnel' : 'tunnel',
'VegetationMisc' : 'vegetation',
'Void' : 'unlabeled',
'Wall' : 'building'
}

camvid_to_city_reduced = {'Animal' : 'unlabeled',
'Archway' : 'building',
'Bicyclist' : 'rider',
'Bridge' : 'building',
'Building' : 'building',
'Car' : 'car',
'CartLuggagePram' : 'car',
'Child' : 'person',
'Column_Pole' : 'pole',
'Fence' : 'fence',
'LaneMkgsDriv' : 'road mark',
'LaneMkgsNonDriv' : 'road mark',
'Misc_Text' : 'traffic sign',
'MotorcycleScooter' : 'motocycle',
'OtherMoving' : 'motocycle',
'ParkingBlock' : 'fence',
'Pedestrian' : 'person',
'Road' : 'road',
'RoadShoulder' : 'building',
'Sidewalk' : 'sidewalk',
'SignSymbol' : 'traffic sign',
'Sky' : 'sky',
'SUVPickupTruck' : 'truck',
'TrafficCone' : 'pole',
'TrafficLight' : 'traffic light',
'Train' : 'train',
'Tree' : 'vegetation',
'Truck_Bus' : 'truck',
'Tunnel' : 'tunnel',
'VegetationMisc' : 'vegetation',
'Void' : 'unlabeled',
'Wall' : 'building'
}

camvid_reduced_custom = {'Animal' : 'person',
'Archway' : 'building',
'Bicyclist' : 'person',
'Bridge' : 'building',
'Building' : 'building',
'Car' : 'car',
'CartLuggagePram' : 'car',
'Child' : 'person',
'Column_Pole' : 'pole',
'Fence' : 'fence',
'LaneMkgsDriv' : 'road mark',
'LaneMkgsNonDriv' : 'road mark',
'Misc_Text' : 'traffic sign',
'MotorcycleScooter' : 'bike',
'OtherMoving' : 'car',
'ParkingBlock' : 'fence',
'Pedestrian' : 'person',
'Road' : 'road',
'RoadShoulder' : 'sidewalk',
'Sidewalk' : 'sidewalk',
'SignSymbol' : 'traffic sign',
'Sky' : 'sky',
'SUVPickupTruck' : 'car',
'TrafficCone' : 'pole',
'TrafficLight' : 'traffic light',
'Train' : 'car',
'Tree' : 'vegetation',
'Truck_Bus' : 'car',
'Tunnel' : 'building',
'VegetationMisc' : 'vegetation',
'Void' : 'unlabeled',
'Wall' : 'building'
}

camvid_only_mark = {'Animal' : 'unlabeled',
'Archway' : 'unlabeled',
'Bicyclist' : 'unlabeled',
'Bridge' : 'unlabeled',
'Building' : 'unlabeled',
'Car' : 'unlabeled',
'CartLuggagePram' : 'unlabeled',
'Child' : 'unlabeled',
'Column_Pole' : 'unlabeled',
'Fence' : 'unlabeled',
'LaneMkgsDriv' : 'road mark',
'LaneMkgsNonDriv' : 'road mark',
'Misc_Text' : 'unlabeled',
'MotorcycleScooter' : 'unlabeled',
'OtherMoving' : 'unlabeled',
'ParkingBlock' : 'unlabeled',
'Pedestrian' : 'unlabeled',
'Road' : 'road',
'RoadShoulder' : 'unlabeled',
'Sidewalk' : 'unlabeled',
'SignSymbol' : 'unlabeled',
'Sky' : 'unlabeled',
'SUVPickupTruck' : 'unlabeled',
'TrafficCone' : 'unlabeled',
'TrafficLight' : 'unlabeled',
'Train' : 'unlabeled',
'Tree' : 'unlabeled',
'Truck_Bus' : 'unlabeled',
'Tunnel' : 'unlabeled',
'VegetationMisc' : 'unlabeled',
'Void' : 'unlabeled',
'Wall' : 'unlabeled'
}

vistas_map = { 'Bird' : 'alive',
 'Ground Animal' : 'alive',
 'Curb' : 'fence',
 'Fence' : 'fence',
 'Guard Rail' : 'fence',
 'Barrier' : 'fence',
 'Wall' : 'building',
 'Bike Lane' : 'road',
 'Crosswalk - Plain' : 'road mark',
 'Curb Cut' : 'fence',
 'Parking' : 'road',
 'Pedestrian Area' : 'sidewalk',
 'Rail Track' : 'road mark',
 'Road' : 'road',
 'Service Lane' : 'road',
 'Sidewalk' : 'sidewalk',
 'Bridge' : 'building',
 'Building' : 'building',
 'Tunnel' : 'building',
 'Person' : 'alive',
 'Bicyclist' : 'alive',
 'Motorcyclist' : 'alive',
 'Other Rider' : 'alive',
 'Lane Marking - Crosswalk' : 'road mark',
 'Lane Marking - General' : 'road mark',
 'Mountain' : 'building',
 'Sand' : 'sidewalk',
 'Sky' : 'sky',
 'Snow' : 'sidewalk',
 'Terrain' : 'sidewalk',
 'Vegetation' : 'vegetation',
 'Water' : 'sidewalk',
 'Banner' : 'road sign',
 'Bench' : 'building',
 'Bike Rack' : 'building',
 'Billboard' : 'road sign',
 'Catch Basin' : 'sidewalk',
 'CCTV Camera' : 'building',
 'Fire Hydrant' : 'pole',
 'Junction Box' : 'pole',
 'Mailbox' : 'pole',
 'Manhole' : 'unlabeled',
 'Phone Booth' : 'building',
 'Pothole' : 'road',
 'Street Light' : 'pole',
 'Pole' : 'pole',
 'Traffic Sign Frame' : 'traffic light',
 'Utility Pole' : 'Pole',
 'Traffic Light' : 'traffic light',
 'Traffic Sign (Back)' : 'traffic light',
 'Traffic Sign (Front)' : 'traffic light',
 'Trash Can' : 'cone',
 'Bicycle' : 'car',
 'Boat' : 'car',
 'Bus' : 'car',
 'Car' : 'car',
 'Caravan' : 'car',
 'Motorcycle' : 'car',
 'On Rails' : 'car',
 'Other Vehicle' : 'car',
 'Trailer' : 'car',
 'Truck' : 'car',
 'Wheeled Slow' : 'car',
 'Car Mount' : 'unlabeled',
 'Ego Vehicle' : 'car',
 'Unlabeled' : 'unlabeled'
}

vistas_map_only_mark = {
  'Bird' : 'unlabeled',
 'Ground Animal' : 'unlabeled',
 'Curb' : 'unlabeled',
 'Fence' : 'unlabeled',
 'Guard Rail' : 'unlabeled',
 'Barrier' : 'unlabeled',
 'Wall' : 'unlabeled',
 'Bike Lane' : 'road',
 'Crosswalk - Plain' : 'road',
 'Curb Cut' : 'unlabeled',
 'Parking' : 'road',
 'Pedestrian Area' : 'unlabeled',
 'Rail Track' : 'road',
 'Road' : 'road',
 'Service Lane' : 'road',
 'Sidewalk' : 'unlabeled',
 'Bridge' : 'unlabeled',
 'Building' : 'unlabeled',
 'Tunnel' : 'unlabeled',
 'Person' : 'unlabeled',
 'Bicyclist' : 'unlabeled',
 'Motorcyclist' : 'unlabeled',
 'Other Rider' : 'unlabeled',
 'Lane Marking - Crosswalk' : 'road',
 'Lane Marking - General' : 'road mark',
 'Mountain' : 'unlabeled',
 'Sand' : 'unlabeled',
 'Sky' : 'unlabeled',
 'Snow' : 'unlabeled',
 'Terrain' : 'unlabeled',
 'Vegetation' : 'unlabeled',
 'Water' : 'unlabeled',
 'Banner' : 'unlabeled',
 'Bench' : 'unlabeled',
 'Bike Rack' : 'unlabeled',
 'Billboard' : 'unlabeled',
 'Catch Basin' : 'unlabeled',
 'CCTV Camera' : 'unlabeled',
 'Fire Hydrant' : 'unlabeled',
 'Junction Box' : 'unlabeled',
 'Mailbox' : 'unlabeled',
 'Manhole' : 'unlabeled',
 'Phone Booth' : 'unlabeled',
 'Pothole' : 'road',
 'Street Light' : 'unlabeled',
 'Pole' : 'unlabeled',
 'Traffic Sign Frame' : 'unlabeled',
 'Utility Pole' : 'unlabeled',
 'Traffic Light' : 'unlabeled',
 'Traffic Sign (Back)' : 'unlabeled',
 'Traffic Sign (Front)' : 'unlabeled',
 'Trash Can' : 'unlabeled',
 'Bicycle' : 'unlabeled',
 'Boat' : 'unlabeled',
 'Bus' : 'unlabeled',
 'Car' : 'unlabeled',
 'Caravan' : 'unlabeled',
 'Motorcycle' : 'unlabeled',
 'On Rails' : 'unlabeled',
 'Other Vehicle' : 'unlabeled',
 'Trailer' : 'unlabeled',
 'Truck' : 'unlabeled',
 'Wheeled Slow' : 'unlabeled',
 'Car Mount' : 'unlabeled',
 'Ego Vehicle' : 'unlabeled',
 'Unlabeled' : 'unlabeled'
}

cityscape_cityscape = {
 'unlabeled' : 'unlabeled',
 'ego vehicle' : 'unlabeled',
 'rectification border' : 'road',
 'out of roi' : 'unlabeled',
 'static' : 'unlabeled',
 'dynamic' : 'unlabeled',
 'ground' : 'ground',
 'road' : 'road',
 'sidewalk' : 'sidewalk',
 'parking' : 'road',
 'rail track' : 'rail track',
 'building' : 'building',
 'wall' : 'wall',
 'fence' : 'fence',
 'guard rail' : 'fence',
 'bridge' : 'bridge',
 'tunnel' : 'tunnel',
 'pole' : 'pole',
 'polegroup' : 'pole',
 'traffic light' : 'traffic light',
 'traffic sign' : 'traffic sign',
 'vegetation' : 'vegetation',
 'terrain' : 'terrain',
 'sky' : 'unlabeled',
 'person' : 'person',
 'rider' : 'person',
 'car' : 'car',
 'truck' : 'car',
 'bus' : 'car',
 'caravan' : 'car',
 'trailer' : 'car',
 'train' : 'car',
 'motorcycle' : 'motorcycle',
 'bicycle' : 'motorcycle'
 }

apollo_cityscape = {
    'others' : 'unlabeled',
    'rover' : 'unlabeled',
    'sky' : 'unlabeled',
    'car' : 'car',
    'car_groups' : 'car',
    'motorbicycle' : 'motorcycle',
    'motorbicycle_group' : 'motorcycle',
    'bicycle' : 'motorcycle',
    'bicycle_group' : 'motorcycle',
    'person' : 'person',
    'person_group' : 'person',
    'rider' : 'person',
    'rider_group' : 'person',
    'truck' : 'car',
    'truck_group' : 'car',
    'bus' : 'car',
    'bus_group' : 'car',
    'tricycle' : 'motorcycle',
    'tricycle_group' : 'motorcycle',
    'road' : 'road',
    'siderwalk' : 'sidewalk',
    'traffic_cone' : 'pole',
    'road_pile' : 'pole',
    'fence' : 'fence',
    'traffic_light' : 'traffic light',
    'pole' : 'pole',
    'traffic_sign' : 'traffic sign',
    'wall' : 'wall',
    'dustbin' : 'pole',
    'billboard' : 'traffic sign',
    'building' : 'building',
    'bridge' : 'bridge',
    'tunnel' : 'tunnel',
    'overpass' : 'bridge',
    'vegatation' : 'vegetation',
    'unlabeled' : 'unlabeled'
}

vistas_cityscape = {
 'Bird' : 'person',
 'Ground Animal' : 'person',
 'Curb' : 'sidewalk',
 'Fence' : 'fence',
 'Guard Rail' : 'fence',
 'Barrier' : 'wall',
 'Wall' : 'wall',
 'Bike Lane' : 'sidewalk',
 'Crosswalk - Plain' : 'road',
 'Curb Cut' : 'sidewalk',
 'Parking' : 'road',
 'Pedestrian Area' : 'sidewalk',
 'Rail Track' : 'rail track',
 'Road' : 'road',
 'Service Lane' : 'road',
 'Sidewalk' : 'sidewalk',
 'Bridge' : 'bridge',
 'Building' : 'building',
 'Tunnel' : 'tunnel',
 'Person' : 'person',
 'Bicyclist' : 'person',
 'Motorcyclist' : 'person',
 'Other Rider' : 'person',
 'Lane Marking - Crosswalk' : 'road',
 'Lane Marking - General' : 'road',
 'Mountain' : 'terrain',
 'Sand' : 'ground',
 'Sky' : 'unlabeled',
 'Snow' : 'ground',
 'Terrain' : 'terrain',
 'Vegetation' : 'vegetation',
 'Water' : 'terrain',
 'Banner' : 'traffic sign',
 'Bench' : 'building',
 'Bike Rack' : 'unlabeled',
 'Billboard' : 'traffic sign',
 'Catch Basin' : 'unlabeled',
 'CCTV Camera' : 'unlabeled',
 'Fire Hydrant' : 'pole',
 'Junction Box' : 'pole',
 'Mailbox' : 'pole',
 'Manhole' : 'unlabeled',
 'Phone Booth' : 'building',
 'Pothole' : 'unlabeled',
 'Street Light' : 'pole',
 'Pole' : 'pole',
 'Traffic Sign Frame' : 'traffic sign',
 'Utility Pole' : 'pole',
 'Traffic Light' : 'traffic light',
 'Traffic Sign (Back)' : 'traffic sign',
 'Traffic Sign (Front)' : 'traffic sign',
 'Trash Can' : 'pole',
 'Bicycle' : 'motorcycle',
 'Boat' : 'car',
 'Bus' : 'car',
 'Car' : 'car',
 'Caravan' : 'car',
 'Motorcycle' : 'motorcycle',
 'On Rails' : 'car',
 'Other Vehicle' : 'car',
 'Trailer' : 'car',
 'Truck' : 'car',
 'Wheeled Slow' : 'motorcycle',
 'Car Mount' : 'unlabeled',
 'Ego Vehicle' : 'unlabeled',
 'Unlabeled' : 'unlabeled'
}

def convert_vistas_labels_to_txt(filename, outputname):
    with open(filename, 'r') as f:
        data = json.loads(f.read())

    output = ''

    for label in data['labels']:
        name = label['readable']
        color = label['color']
        output = output + str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' ' + name + '\n'

    output_template = '''vistas_map = {'''
    index = 0
    for label in data['labels']:
        name = label['readable']
        output_template = output_template + ''' '{0}' : '{1}', '''.format(name, index) + '\n'
        index = index + 1
    output_template = output_template + '\n}'
    print(output_template)

    with open(outputname, 'w') as f:
        f.write(output)

def convert_indeces_to_txt(filename, outputname):
    with open(filename, 'r') as f:
        data = json.loads(f.read())

    output = ''

    index = 0
    for label in data['labels']:
        name = label['readable']

        output = output + str(index) + ' ' + name + '\n'

        index = index + 1

    # output_template = '''vistas_map = {'''
    # index = 0
    # for label in data['labels']:
    #     name = label['readable']
    #     output_template = output_template + ''' '{0}' : '{1}', '''.format(name, index) + '\n'
    #     index = index + 1
    # output_template = output_template + '\n}'
    #print(output_template)

    with open(outputname, 'w') as f:
        f.write(output)

def load_index_map(filename):

    colormap = {}
    with open(filename, 'r') as file:
        for line in file:

            if len(line.strip()) < 3:
                continue

            line = line.replace('\t\t', ' ')
            line = line.replace('\t', ' ')
            items = line.split(' ')

            color = int(items[0])
            name = ' '.join(items[1 : ]).strip()
            colormap[name] = color

    print(colormap)
    return colormap

def load_color_map(filename):

    colormap = {}
    with open(filename, 'r') as file:
        for line in file:

            if len(line.strip()) < 9:
                continue

            line = line.replace('\t\t', ' ')
            line = line.replace('\t', ' ')
            items = line.split(' ')
            print(items)
            color = [int(item.strip()) for item in items[: 3]]
            name = ' '.join(items[3 : ]).strip()
            colormap[name] = color

    print(colormap)
    return colormap

def _process_index_to_index(filename, names_map, orig_map, dist_map, must_have_index = -1, must_have_percent = 0.0, size = ()):

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if len(size) == 2:
        img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)

    # The slowest operation is to find indeces. So it is multiprocessed
    colors_indeces = {}

    def get_indx(name):
        i = (img == orig_map[name])
        return name, i


    result = []
    for name in orig_map.keys():
        i = (img == orig_map[name])
        result.append([name, i])

    # pool = Pool(processes = 4, maxtasksperchild = 10)
    # result = pool.map(get_indx, orig_map.keys())
    # pool.close()
    # pool.clear()

    for res in result:
        name = res[0]
        idx = res[1]
        colors_indeces[name] = idx
    ######

    for orig_name, name in names_map.items():
        index = dist_map[name]

        i = colors_indeces[orig_name]
        img[i] = index

    if must_have_index >= 0:

        if must_have_percent > 0.0:

            percent = float(np.count_nonzero(img > 0)) / (img.shape[0] * img.shape[1])
            if percent < must_have_percent:
                return False, None


        else:
            if not must_have_index in img:
                return False, None

    return True, img

def convert_index_to_index(path_orig, indexmap_original, indexmap_dist, names_map, mask = ''):

    orig_map = load_color_map(colormap_original)
    dist_map = load_color_map(colormap_dist)

    files = navmii_utils.GetAllFilesListRecusive(path_orig, ['.jpeg', '.png', '.jpg'])

    filenames = list([f for f in files if mask in f])

    i = 0
    for filename in filenames:
        i = i + 1
        sys.stdout.flush()
        sys.stdout.write('\r>> Converting image %d/%d' % (i, len(filenames)))
        sys.stdout.flush()
        sucess, img = _process_convert_to_index(filename, names_map, orig_map, dist_map)
        if success:
            cv2.imwrite(filename, img)

def convert_colors_to_index(path_orig, colormap_original, colormap_dist, names_map, mask = ''):

    orig_map = load_color_map(colormap_original)
    dist_map = load_color_map(colormap_dist)

    files = navmii_utils.GetAllFilesListRecusive(path_orig, ['.jpeg', '.png', '.jpg', '.JPG', '.JPEG'])

    filenames = list([f for f in files if mask in f])

    i = 0
    for filename in filenames:
        i = i + 1
        sys.stdout.flush()
        sys.stdout.write('\r>> Converting image %d/%d' % (i, len(filenames)))
        sys.stdout.flush()
        sucess, img = _process_convert_to_index(filename, names_map, orig_map, dist_map,
                                                must_have_index = -1, must_have_percent = 0.0)
        if sucess:
            cv2.imwrite(filename, img)

def _process_convert_to_index(filename, names_map, orig_map, dist_map, must_have_index = -1,  must_have_percent = 0.0, size = ()):
    img = cv2.imread(filename)

    if len(size) == 2:
        img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #dist_img = np.zeros((img.shape[0], img.shape[1]))

    # The slowest operation is to find indeces. So it is multiprocessed
    colors_indeces = {}

    def get_indx(name):
        i = np.where((img == orig_map[name]).all(axis = 2))
        return name, i

    result = pool.map(get_indx, orig_map.keys())

    for res in result:
        name = res[0]
        idx = res[1]
        colors_indeces[name] = idx
    ######

    for orig_name, name in names_map.items():
        orig_color = orig_map[orig_name]
        color = dist_map[name]
        i = colors_indeces[orig_name]

        print(orig_name, len(i[0]), color)
        index = autovision_label_colours.index((color[0], color[1], color[2]))

        #img[np.where((img == orig_color).all(axis = 2))] = color
        img[i] = index

    if must_have_index >= 0:

        if must_have_percent > 0.0:

            percent = float(np.count_nonzero(img == must_have_index)) / (img.shape[0] * img.shape[1])
            if percent < must_have_percent:
                return False, None


        else:
            if not must_have_index in img:
                return False, None


    #cv2.imwrite(filename, dist_img)
    return True, img

import time
def convert_colors(path_orig, colormap_original, colormap_dist, names_map, mask = ''):

    orig_map = load_color_map(colormap_original)
    dist_map = load_color_map(colormap_dist)

    files = navmii_utils.GetAllFilesListRecusive(path_orig, ['.jpeg', '.png', '.jpg'])

    filenames = list([f for f in files if mask in f])

    i = 0
    for filename in filenames:
        i = i + 1
        sys.stdout.flush()
        sys.stdout.write('\r>> Converting image %d/%d' % (i, len(filenames)))
        sys.stdout.flush()
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for orig_name, name in names_map.items():
            orig_color = orig_map[orig_name]
            color = dist_map[name]
            img[np.where((img == orig_color).all(axis = 2))] = color

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img)

def merge_cityscape_into_dir_indeces(images_path, dist, names_map, orig_map, dist_map, o_filename):
    images = navmii_utils.GetAllFilesListRecusive(images_path, ['_leftImg8bit.png'])
    labels = navmii_utils.GetAllFilesListRecusive(images_path, ['_gtFine_labelIds.png'])

    orig_map = load_index_map(orig_map)
    dist_map = load_index_map(dist_map)

    list_out = ''
    i = 0
    uns = 0
    for img in images:

        i = i + 1
        sys.stdout.flush()
        sys.stdout.write('\r>> Copying image %d/%d' % (i, len(images)))
        sys.stdout.flush()

        pure_name = img[img.rfind('/') + 1: ]
        new_img_name = dist + '/' + pure_name

        mask_names = [label for label in labels if pure_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png') in label]

        if len(mask_names) != 1:
            print('WTF???', mask_names, pure_name)
            continue

        mask_name = mask_names[0]
        new_mask_name = dist + '/' + pure_name[: pure_name.rfind('.')] + '_L.png'

        t = time.time()
        success, im = _process_index_to_index(mask_name, names_map, orig_map,
                                              dist_map, must_have_index = 0,
                                              must_have_percent = 0.2, size = (1280, 1280))
        #print(time.time() - t)

        # No road mark on image therefore
        if not success:

            print('Unsuccess percent: ', (uns / float(i)) * 100.0)
            uns = uns + 1
            continue

        cv2.imwrite(new_mask_name, im)

        im = cv2.imread(img)
        im = cv2.resize(im, (1280, 1280))
        cv2.imwrite(new_img_name, im)
        del im

        list_out = list_out + new_img_name + ' ' + new_mask_name + '\n'

    with open(dist + '/../' + o_filename, 'w') as f:
        f.write(list_out)


def merge_apollo_into_dir_indeces(images_path, dist, names_map, orig_map, dist_map):
    images = navmii_utils.GetAllFilesListRecusive(images_path, ['.jpg'])
    labels = navmii_utils.GetAllFilesListRecusive(images_path, ['_bin.png'])

    orig_map = load_index_map(orig_map)
    dist_map = load_index_map(dist_map)

    list_out = ''
    i = 0
    uns = 0
    times = []
    for img in images:

        t = time.time()

        i = i + 1
        sys.stdout.flush()
        avg_time = 0
        if len(times):
            avg_time = (float(sum(times)) / len(times)) / 60.0
            avg_time = '{}'.format(int(avg_time * (len(images) - i)))
        sys.stdout.write('\r>> Copying image %d/%d, remains ~%s minutes' % (i, len(images), avg_time))
        sys.stdout.flush()

        pure_name = img[img.rfind('/') + 1: ]
        new_img_name = dist + '/' + pure_name

        mask_name = img.replace('.jpg', '_bin.png').replace('ColorImage', 'Label')

        if not os.path.exists(mask_name):
            mask_name = img.replace('.jpg', '.png').replace('ColorImage', 'Label')
            if not os.path.exists(mask_name):
                print('WTF???', img, mask_name)
                continue

        new_mask_name = dist + '/' + pure_name[: pure_name.rfind('.')] + '_L.png'

        success, im = _process_index_to_index(mask_name, names_map, orig_map,
                                              dist_map, must_have_index = 0,
                                              must_have_percent = 0.2, size = (1280, 1280))

        # No road mark on image therefore
        if not success:

            print('Unsuccess percent: ', (uns / float(i)) * 100.0)
            uns = uns + 1
            continue

        cv2.imwrite(new_mask_name, im)

        im = cv2.imread(img)
        im = cv2.resize(im, (1280, 1280))
        cv2.imwrite(new_img_name, im)

        list_out = list_out + new_img_name + ' ' + new_mask_name + '\n'
        times.append(time.time() - t)

    with open(dist + '/../valid_list.txt', 'w') as f:
        f.write(list_out)

def merge_vistas_into_dir_indeces(images_path, labels_path, dist, names_map, orig_map, dist_map, o_filename):

    images = navmii_utils.GetAllFilesListRecusive(images_path, ['.jpeg', '.png', '.jpg'])
    labels = navmii_utils.GetAllFilesListRecusive(labels_path, ['.jpeg', '.png', '.jpg'])

    orig_map = load_index_map(orig_map)
    dist_map = load_index_map(dist_map)

    list_out = ''
    i = 0
    uns = 0
    times = []
    for img in images:

        t = time.time()
        i = i + 1
        sys.stdout.flush()
        avg_time = 0
        if len(times):
            avg_time = (float(sum(times)) / len(times)) / 60.0
            avg_time = '{}'.format(int(avg_time * (len(images) - i)))
        sys.stdout.write('\r>> Copying image %d/%d, remains ~%s minutes' % (i, len(images), avg_time))
        sys.stdout.flush()

        pure_name = img[img.rfind('/') + 1: ]
        new_img_name = dist + '/' + pure_name

        mask_names = [label for label in labels if pure_name.replace('.jpg', '.png') in label]

        if len(mask_names) != 1:
            print('WTF???', mask_names, pure_name)
            continue

        mask_name = mask_names[0]
        new_mask_name = dist + '/' + pure_name[: pure_name.rfind('.')] + '_L.png'

        t = time.time()
        success, im = _process_index_to_index(mask_name, names_map, orig_map,
                                              dist_map, must_have_index = 2,
                                              must_have_percent = 0.0015, size = (1280, 1280))
        #print(time.time() - t)

        # No road mark on image therefore
        if not success:

            print('Unsuccess percent: ', (uns / float(i)) * 100.0)
            uns = uns + 1
            continue

        cv2.imwrite(new_mask_name, im)

        im = cv2.imread(img)
        im = cv2.resize(im, (1280, 1280))
        cv2.imwrite(new_img_name, im)
        times.append(time.time() - t)

        list_out = list_out + new_img_name + ' ' + new_mask_name + '\n'

    with open(dist + '/../' + o_filename, 'w') as f:
        f.write(list_out)

def merge_vistas_into_dir(images_path, labels_path, dist, names_map, orig_map, dist_map):
    images = navmii_utils.GetAllFilesListRecusive(images_path, ['.jpeg', '.png', '.jpg'])
    labels = navmii_utils.GetAllFilesListRecusive(labels_path, ['.jpeg', '.png', '.jpg'])

    orig_map = load_color_map(orig_map)
    dist_map = load_color_map(dist_map)

    list_out = ''
    i = 0
    for img in images:

        i = i + 1
        sys.stdout.flush()
        sys.stdout.write('\r>> Copying image %d/%d' % (i, len(images)))
        sys.stdout.flush()

        pure_name = img[img.rfind('/') + 1: ]
        new_img_name = dist + '/' + pure_name

        mask_names = [label for label in labels if pure_name.replace('.jpg', '.png') in label]

        if len(mask_names) != 1:
            print('WTF???', mask_names, pure_name)
            continue

        mask_name = mask_names[0]
        new_mask_name = dist + '/' + pure_name[: pure_name.rfind('.')] + '_L.png'

        t = time.time()
        success, im = _process_convert_to_index(mask_name, names_map, orig_map,
                                                dist_map, must_have_index = 4,
                                                size = (1280, 1280))
        print(time.time() - t)

        # No road mark on image therefore
        if not success:
            print('Unsuccess')
            continue

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        cv2.imwrite(new_mask_name, im)

        im = cv2.imread(img)
        im = cv2.resize(im, (1280, 1280))
        cv2.imwrite(new_img_name, im)

        list_out = list_out + new_img_name + ' ' + new_mask_name + '\n'

    with open(dist + '/../list.txt', 'w') as f:
        f.write(list_out)


def create_list(path, outfile, ext, postfix = '_L'):
    images = navmii_utils.GetAllFilesListRecusive(path, [ext])
    l_images = [im for im in images if postfix in im]
    r_images = [im for im in images if not postfix in im]

    output = ''
    for r_im in r_images:
        l_im = r_im.replace(ext, postfix + '.png')
        output = output + r_im + ' ' + l_im + '\n'

    with open(outfile, 'w') as f:
        f.write(output)

def check(path, postfix = '_L'):
    images = navmii_utils.GetAllFilesListRecusive(path, ['.jpeg', '.png', '.jpg'])

    for img in images:
        try:
            print(img)
            cv2.imread(img)
        except:
            os.remove(img)
            if '.jpg' in img:
                img = img.replace('.jpg', postfix + '.png')
            if '.jpeg' in img:
                img = img.replace('.jpg', postfix + '.png')
            os.remove(img)

def calc_mean(path, ignore_mask = '_L'):
    images = navmii_utils.GetAllFilesListRecusive(path, ['.jpeg', '.png', '.jpg'])

    files = [f for f in images if ignore_mask not in f]

    mean = np.array([0, 0, 0]) # BGR

    for img_path in files:
        img = cv2.imread(img_path)
        m = [img[:, :, i].mean() for i in range(img.shape[-1])]
        print(m)
        mean = mean + m
        print(mean)

    print('BGR mean: ', mean, len(files), np.array(mean) / len(files))

def resize_in_dir(dir, size, gt_mask = '_L'):

    files = navmii_utils.GetAllFilesListRecusive(dir, ['.jpeg', '.png', '.jpg'])

    for f in files:
        img = cv2.imread(f)
        if not gt_mask in f:
            img = cv2.resize(img, size)
        else:
            img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(f, img)

def convert_custom_to_index(path, colours_list, offset = 20):
    files = navmii_utils.GetAllFilesListRecusive(path, ['.jpeg', '.png', '.jpg'])

    masks = list([f for f in files if 'mask' in f])

    i = 0
    for mask in masks:
        mask_img = cv2.imread(mask)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        orig_mask = copy.deepcopy(mask_img)

        # if not isinstance(orig_img, np.ndarray):
        #     continue

        i = i + 1
        sys.stdout.flush()
        sys.stdout.write('\r>> Converting image %d/%d' % (i, len(files)))
        sys.stdout.flush()

        for color in colours_list:

            # upper_color = np.array(color) + offset
            # lower_color = np.array(color) - offset
            # print(upper_color, lower_color)
            # color_mask = cv2.inRange(orig_mask, lower_color, upper_color)
            # print(color_mask)
            # print(set(color_mask.flatten()))

            # mask_img[np.where(color_mask == [255])] = colours_list.index(color)

            #index = np.where(np.logical_and(condition_one, condition_two))
            # print('index', condition_one, color)
            #mask_img[index] = colours_list.index(color)
            mask_img[np.where((mask_img == color).all(axis = 2))] = colours_list.index(color)

        color_mask = cv2.inRange(mask_img, np.array([0, 0, 0]), np.array([len(colours_list) - 1] * 3))
        print(color_mask)
        mask_img[np.where(color_mask == 0)] = 0
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(mask, mask_img)

if __name__ == '__main__':
    # convert_indeces_to_txt('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/config.json',
    #                        '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/index_labels.txt')
    #merge_vistas_into_dir('/mnt/Data/Datasets/Segmentation/mapillary_3_class_big/training/images',
    #'/mnt/Data/Datasets/Segmentation/mapillary_3_class_big/training/labels',
    #'/mnt/Data/Datasets/Segmentation/mapillary_3_class_big/merged',
    #orig_map = '/mnt/Data/Datasets/Segmentation/mapillary_3_class_big/labels.txt',
    #dist_map = '/mnt/Data/Datasets/Segmentation/Cityscapes/color_map_road_mark.txt',
    #names_map = vistas_map_only_mark)
    #
    # check('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped')
    # check('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/validation')
    # check('/mnt/Data/Datasets/Segmentation/Cityscapes/remapped')
    # check('/mnt/Data/Datasets/Segmentation/Apollo/remapped')

    # path_orig, colormap_original, colormap_dist, names_map, mask = '')
    convert_colors_to_index('/mnt/Data/Datasets/Autovision/v0beta/testset', 
                            '/mnt/Data/Datasets/Autovision/v0beta/autovision_color_map.txt',
                            '/mnt/Data/Datasets/Autovision/v0beta/autovision_color_map.txt',
                            autovision_to_autovision, mask = '_mask0')

    create_list('/mnt/Data/Datasets/Autovision/v0beta/trainset',
              '/mnt/Data/Datasets/Autovision/v0beta/train.txt', ext = '.png', postfix = '_mask0')
    create_list('/mnt/Data/Datasets/Autovision/v0beta/testset',
              '/mnt/Data/Datasets/Autovision/v0beta/test.txt', ext = '.png', postfix = '_mask0')
              
    # merge_apollo_into_dir_indeces('/mnt/Data/Datasets/Segmentation/Apollo',
    #                               '/mnt/Data/Datasets/Segmentation/Apollo/remapped',
    #                               orig_map = '/mnt/Data/Datasets/Segmentation/Apollo/index_map.txt',
    #                               dist_map = '/mnt/Data/Datasets/Segmentation/Cityscapes/cityscape_color_index.txt',
    #                               names_map = apollo_cityscape)
    # merge_vistas_into_dir_indeces('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/validation/images',
    #                               '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/validation/instances',
    #                               '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped_valid',
    #                               orig_map = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/index_labels.txt',
    #                               dist_map = '/mnt/Data/Datasets/Segmentation/Cityscapes/cityscape_color_index.txt',
    #                               names_map = vistas_cityscape, o_filename = 'valid.txt')
    # merge_vistas_into_dir_indeces('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/training/images',
    #                               '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/training/instances',
    #                               '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped',
    #                               orig_map = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/index_labels.txt',
    #                               dist_map = '/mnt/Data/Datasets/Segmentation/Cityscapes/cityscape_color_index.txt',
    #                               names_map = vistas_cityscape, o_filename = 'train.txt')
    # merge_cityscape_into_dir_indeces('/mnt/Data/Datasets/Segmentation/Cityscapes/gtFine_trainvaltest/gtFine',
    #                                  '/mnt/Data/Datasets/Segmentation/Cityscapes/remapped',
    #                                  orig_map = '/mnt/Data/Datasets/Segmentation/Cityscapes/cityscape_color_index_original.txt',
    #                                  dist_map = '/mnt/Data/Datasets/Segmentation/Cityscapes/cityscape_color_index.txt',
    #                                  names_map = cityscape_cityscape, o_filename = 'train.txt')

    #convert_custom_to_index('/home/undead/segment/ld_out/test', label_only_mark)

    # convert_colors_to_index('/mnt/Data/Datasets/Segmentation/Camvid/dataset', '/mnt/Data/Datasets/Segmentation/Camvid/class_codes.txt',
    # '/mnt/Data/Datasets/Segmentation/Cityscapes/color_map_road_mark.txt', camvid_only_mark, mask = '_L')
    #resize_in_dir('/mnt/Data/Datasets/Segmentation/Camvid/dataset', (1280, 1280), '_L')


    #calc_mean('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/merged', '_L')

    # print(len(set(vistas_map_only_mark.values())))
    # convert_colors(path_orig = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/test',
    #                colormap_original = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/labels.txt',
    #                colormap_dist = '/mnt/Data/Datasets/Segmentation/Cityscapes/color_map_road_mark.txt',
    #                names_map = vistas_map_only_mark, mask = '_L')
    # convert_vistas_labels_to_txt('/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/config.json',
    #                             '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/labels.txt')
