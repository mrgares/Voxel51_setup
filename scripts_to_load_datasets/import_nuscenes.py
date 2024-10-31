# Description: Import nuscenes dataset to fiftyone dataset for the first time
# Author: Follow tutorial from https://voxel51.com/blog/nuscenes-dataset-navigating-the-road-ahead/
# Date: 28/oct/2024

import os
os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://fiftyone_server:27017" # Set the correct MongoDB URI for FiftyOne
import fiftyone as fo
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import  BoxVisibility
from nuscenes.utils.geometry_utils import box_in_image, view_points
import numpy as np
from PIL import Image
from nuscenes.utils.color_map import get_colormap
from nuscenes.lidarseg.lidarseg_utils import paint_points_label
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
import open3d as o3d
import os
import time
from tqdm import tqdm


DATASET_ROOT = "/datastore/nuScenes/"
BATCH_SIZE = 50

def load_lidar(lidar_token):
    #Grab and Generate Colormaps
    #    gt_from = "lidarseg"
    #    lidarseg_filename = DATASET_ROOT + nusc.get(gt_from, lidar_token)['filename']
    #    colormap = get_colormap()
    #    name2index = nusc.lidarseg_name2idx_mapping
    #    coloring = paint_points_label(lidarseg_filename,None,name2index,      colormap=colormap)
    filepath = DATASET_ROOT + nusc.get("sample_data", lidar_token)['filename']
    root, extension = os.path.splitext(filepath)
   
   # Check if the file is already in the right format
    if extension == ".pcd":
        return filepath

    #Load Point Cloud
    cloud = LidarPointCloud.from_file(filepath)
    pcd = o3d.geometry.PointCloud()
    points_np=cloud.points[:3,:].T.astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(points_np)
    #    colors = coloring[:,:3]
    #    colors.max()
    #    pcd.colors = o3d.utility.Vector3dVector(colors)
    #Save back Point Cloud
    o3d.io.write_point_cloud(root, pcd)
    #Return Filepath For New Point Cloud   
    return root

def lidar_sample(group, filepath, sensor, lidar_token):
    #Grab all detections
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,)
    sample = fo.Sample(filepath=filepath, group=group.element(sensor))
    detections = []
    for box in boxes:
        #Grab the variables needed to place a box in a point cloud           
        x, y, z = box.orientation.yaw_pitch_roll
        w, l, h = box.wlh.tolist()
        detection = fo.Detection(
                label=box.name,
                location=box.center.tolist(),
                rotation=[z, y, x],
                dimensions=[l,w,h]
                )
        detections.append(detection)
    #Add all of our new detections to the sample
    sample["ground_truth"] = fo.Detections(detections=detections)
    return sample

def camera_sample(group, filepath, sensor, token):
   #Initialize our sample
    sample = fo.Sample(filepath=filepath, group=group.element(sensor))
    #Load our boxes
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(token, box_vis_level=BoxVisibility.ANY)
    image = Image.open(data_path)
    width, height = image.size
    shape = (height,width)
    polylines = []
    for box in boxes:
        #Convert 3D corners to 2D corners relative to camera
        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
        front = [(corners[0][0]/width,corners[1][0]/height),
                (corners[0][1]/width,corners[1][1]/height),
                (corners[0][2]/width,corners[1][2]/height),
                (corners[0][3]/width,corners[1][3]/height),]
        back =  [(corners[0][4]/width,corners[1][4]/height),
                (corners[0][5]/width,corners[1][5]/height),
                (corners[0][6]/width,corners[1][6]/height),
                (corners[0][7]/width,corners[1][7]/height),]
        #Create new cuboid and add to list
        polylines.append(fo.Polyline.from_cuboid(front + back, label=box.name))
    #Update our sample with its new detections
    sample["cuboids"] = fo.Polylines(polylines=polylines)
    return sample

# Existing setup code remains the same...
if 'nuscenes' not in fo.list_datasets():
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_ROOT, verbose=True)

    # Create the main dataset and add a split field for train/val
    dataset = fo.Dataset(name='nuscenes', persistent=True, overwrite=True)
    dataset.add_group_field("group", default="CAM_FRONT")
    dataset.add_sample_field("split", fo.StringField)  

    print('Loading dataset...')
    
    groups = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", 
              "CAM_BACK_RIGHT", "LIDAR_TOP", "RADAR_FRONT", 
              "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", 
              "RADAR_BACK_RIGHT")
    
    # Get predefined splits
    splits = create_splits_scenes()
    train_scenes = splits["train"]
    val_scenes = splits["val"]
    
    
    batch_samples = []
    scene_len = len(nusc.scene)
    for scene_idx, scene in enumerate(tqdm(nusc.scene, desc="Processing scenes", unit="scene")):
        
        my_scene = scene
        token = my_scene['first_sample_token']
        my_sample = nusc.get('sample', token)
        scene_name = scene["name"]
        
        # Determine the split based on scene name
        if scene_name in train_scenes:
            split = "train"
        elif scene_name in val_scenes:
            split = "validation"
        else:
            continue  # Skip scenes not in train/val splits (e.g., test)

        while not my_sample["next"] == "":
            lidar_token = my_sample["data"]["LIDAR_TOP"]
            group = fo.Group()
            for sensor in groups:
                data = nusc.get('sample_data', my_sample['data'][sensor])
                filepath = DATASET_ROOT + data["filename"]
                
                if data["sensor_modality"] == "lidar":
                    filepath = load_lidar(lidar_token)
                    sample = lidar_sample(group, filepath, sensor, lidar_token)
                elif data["sensor_modality"] == "camera":
                    sample = camera_sample(group, filepath, sensor, my_sample['data'][sensor])
                else:
                    sample = fo.Sample(filepath=filepath, group=group.element(sensor))

                # Assign split label to each sample
                sample["split"] = split
                sample["scene_name"] = scene_name

                batch_samples.append(sample)                 

            token = my_sample["next"]
            my_sample = nusc.get('sample', token)

        # Add batch samples to dataset
        if scene_idx % BATCH_SIZE == 0:
            dataset.add_samples(batch_samples)
            batch_samples = []
        
    # Add any remaining samples
    if batch_samples:
        dataset.add_samples(batch_samples)

    # Filter dataset views for training and validation
    train_view = dataset.match({"split": "train"})
    val_view = dataset.match({"split": "validation"})
    
    print('Created dataset with %d samples' % len(dataset))
    print('%d train samples' % len(train_view))
    print('%d validation samples' % len(val_view))
else:
    dataset = fo.load_dataset("nuscenes")
    print('Loaded dataset with %d samples' % len(dataset))

    # Create filtered views for training and validation
    train_view = dataset.match({"split": "train"})
    val_view = dataset.match({"split": "validation"})
    
    print('Loaded dataset with %d samples' % len(dataset))
    print('%d train samples' % len(train_view))
    print('%d validation samples' % len(val_view))


print(dataset)
session = fo.launch_app(dataset)

# Optionally, launch separate sessions for train and validation views
# session_train = fo.launch_app(train_view)
# session_val = fo.launch_app(val_view)

# Keep the script alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")