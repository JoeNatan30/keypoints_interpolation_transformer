import gc
import ast
import tqdm
import time
import h5py
import glob
import torch
import pandas as pd
import numpy as np
from collections import Counter
import torch.utils.data as torch_data
from torch.utils.data import Dataset
import logging
import random

import augmentation

import cv2

def get_data_from_h5(path):
    hf = h5py.File(path, 'r')
    return hf

class bodyKeypointMap():
    def __init__(self, body_section, body_part):
        self.pose = [pos for pos, body in enumerate(body_section) if body == 'pose' or body == 'face']
        self.face = [pos for pos, body in enumerate(body_section) if body == 'face']
        self.leftHand = [pos for pos, body in enumerate(body_section) if body == 'leftHand']
        self.rightHand = [pos for pos, body in enumerate(body_section) if body == 'rightHand']

        self.body_section_dict = {body:pos for pos, body in enumerate(body_part)}

    def body_part_points(self):
        return self.pose, self.face, self.leftHand, self.rightHand

    def body_dict(self):
        return self.body_section_dict



####################################################################
# Function that helps to see keypoints in an image
####################################################################
def prepare_keypoints_image(keypoints,tag):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    cv2.imwrite(f'foo_{tag}.jpg', img)

##########################################################
# Process used to normalize the pose
##########################################################
def normalize_pose(data, body_dict):

    sequence_size = data.shape[0]
    valid_sequence = True

    last_starting_point, last_ending_point = None, None

    for sequence_index in range(sequence_size):

        # Prevent from even starting the analysis if some necessary elements are not present
        if (data[sequence_index][body_dict['pose_left_shoulder']][0] == 0.0 or data[sequence_index][body_dict['pose_right_shoulder']][0] == 0.0):
            if not last_starting_point:
                valid_sequence = False
                continue

            else:
                starting_point, ending_point = last_starting_point, last_ending_point
    
        else:

            # NOTE:
            #
            # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
            # this is meant for the distance between the very ends of one's shoulder, as literature studying body
            # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
            # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
            # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
            #
            # Please, review this if using other third-party pose estimation libraries.

            if data[sequence_index][body_dict['pose_left_shoulder']][0] != 0 and data[sequence_index][body_dict['pose_right_shoulder']][0] != 0:
                
                left_shoulder = data[sequence_index][body_dict['pose_left_shoulder']]
                right_shoulder = data[sequence_index][body_dict['pose_right_shoulder']]

                shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                       (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)

                mid_distance = (0.5,0.5)#(left_shoulder + right_shoulder)/2
                head_metric = shoulder_distance/2
            '''
            # use it if you have the neck keypoint
            else:
                neck = (data["neck_X"][sequence_index], data["neck_Y"][sequence_index])
                nose = (data["nose_X"][sequence_index], data["nose_Y"][sequence_index])
                neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
                head_metric = neck_nose_distance
            '''
            # Set the starting and ending point of the normalization bounding box
            starting_point = [mid_distance[0] - 3 * head_metric, data[sequence_index][body_dict['pose_right_eye']][1] - (head_metric / 2)]
            ending_point = [mid_distance[0] + 3 * head_metric, mid_distance[1] + 3.5 * head_metric]

            last_starting_point, last_ending_point = starting_point, ending_point

        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):
            
            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - ending_point[1]) / (starting_point[1] -
                                                                                    ending_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = 1 - normalized_y
            
    return data
################################################
# Function that normalize the hands (but also the face)
################################################
def normalize_hand(data, body_section_dict):
    """
    Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
    the definition from our paper.
    :param data: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                that particular joints coordinates
    :return: Dictionary with normalized skeletal data (following the same schema as input data)
    """

    sequence_size = data.shape[0]
    
    # Treat each element of the sequence (analyzed frame) individually
    for sequence_index in range(sequence_size):

        # Retrieve all of the X and Y values of the current frame
        landmarks_x_values = data[sequence_index][:, 0]
        landmarks_y_values = data[sequence_index][:, 1]

        # Prevent from even starting the analysis if some necessary elements are not present
        #if not landmarks_x_values or not landmarks_y_values:
        #    continue

        # Calculate the deltas
        width, height = max(landmarks_x_values) - min(landmarks_x_values), max(landmarks_y_values) - min(
            landmarks_y_values)
        if width > height:
            delta_x = 0.1 * width
            delta_y = delta_x + ((width - height) / 2)
        else:
            delta_y = 0.1 * height
            delta_x = delta_y + ((height - width) / 2)

        # Set the starting and ending point of the normalization bounding box
        starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
        ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):

            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                    starting_point[1] - ending_point[1]) == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - starting_point[1]) / (ending_point[1] -
                                                                                    starting_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = normalized_y

    return data

###################################################################################
# This function normalize the body and the hands separately
# body_section has the general body part name (ex: pose, face, leftHand, rightHand)
# body_part has the specific body part name (ex: pose_left_shoulder, face_right_mouth_down, etc)
###################################################################################
def normalize_pose_hands_function(data, body_parts_class):

    pose, face, leftHand, rightHand = body_parts_class.body_part_points()
    body_section_dict = body_parts_class.body_dict()

    assert len(pose) > 0 and len(leftHand) > 0 and len(rightHand) > 0 #and len(face) > 0

    prepare_keypoints_image(data[2][2][:,:],"before")

    for index_video in range(len(data)):
        data[index_video][:,pose+leftHand+rightHand,:] = normalize_pose(data[index_video][:,pose+leftHand+rightHand,:], body_section_dict)
        #data[index_video][:,face,:] = normalize_hand(data[index_video][:,face,:], body_section_dict)
        #data[index_video][:,leftHand,:] = normalize_hand(data[index_video][:,leftHand,:], body_section_dict)
        #data[index_video][:,rightHand,:] = normalize_hand(data[index_video][:,rightHand,:], body_section_dict)

    prepare_keypoints_image(data[2][2][:,:],"after")

    kp_bp_index = {'pose':pose,
                   'left_hand':leftHand,
                   'rigth_hand':rightHand}

    return data, kp_bp_index, body_section_dict


def get_dataset_from_hdf5(path,keypoints_model, landmarks_ref, keypoints_number):
    print('path                       :',path)
    print('keypoints_model            :',keypoints_model)
    print('landmarks_ref              :',landmarks_ref)

    # Prepare the data to process the dataset

    index_array_column = None #'mp_indexInArray', 'wp_indexInArray','op_indexInArray'

    print('Use keypoint model : ',keypoints_model) 
    if keypoints_model == 'openpose':
        index_array_column  = 'op_indexInArray'
    if keypoints_model == 'mediapipe':
        index_array_column  = 'mp_indexInArray'
    if keypoints_model == 'wholepose':
        index_array_column  = 'wp_indexInArray'
    print('use column for index keypoint :',index_array_column)

    assert not index_array_column is None

    # all the data from landmarks_ref
    df_keypoints = pd.read_csv(landmarks_ref, skiprows=1)

    # 29, 54 or 71 points
    if keypoints_number == 29:
        df_keypoints = df_keypoints[(df_keypoints['Selected 29']=='x' )& (df_keypoints['Key']!='wrist')]
    elif keypoints_number == 71:
        df_keypoints = df_keypoints[(df_keypoints['Selected 71']=='x' )& (df_keypoints['Key']!='wrist')]
    else:
        df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]

    logging.info(" using keypoints_number: "+str(keypoints_number))

    idx_keypoints = sorted(df_keypoints[index_array_column].astype(int).values)
    name_keypoints = df_keypoints['Key'].values
    section_keypoints = (df_keypoints['Section']+'_'+df_keypoints['Key']).values

    print('section_keypoints : ',len(section_keypoints),' -- uniques: ',len(set(section_keypoints)))
    print('name_keypoints    : ',len(name_keypoints),' -- uniques: ',len(set(name_keypoints)))
    print('idx_keypoints     : ',len(idx_keypoints),' -- uniques: ',len(set(idx_keypoints)))
    print('')
    print('section_keypoints used:')
    print(section_keypoints)

    # process the dataset (start)

    print('Reading dataset .. ')
    data = get_data_from_h5(path)

    print('Total size dataset : ',len(data.keys()))

    video_dataset  = []

    time.sleep(2)
    for index in tqdm.tqdm(list(data.keys())):

        data_video = np.array(data[index]['data'])
        data_label = np.array(data[index]['label']).item().decode('utf-8')



        # F x C x K  (frames, coords, keypoitns)
        n_frames, n_axis, n_keypoints = data_video.shape

        data_video = np.transpose(data_video, (0,2,1)) #transpose to n_frames, n_keypoints, n_axis 
        if index=='0':
            print('original size video : ',data_video.shape)
            print('filtering by keypoints idx .. ')
        data_video = data_video[:,idx_keypoints,:]

        if index=='0':
            print('filtered size video : ',data_video.shape)

        video_dataset.append(data_video)

    del data
    gc.collect()

    print('label encoding completed!')

    print('Reading dataset completed!')

    return video_dataset, df_keypoints['Section'], section_keypoints

def replace_points(data, timestep, hand, wrist):
    
    data[timestep,hand,0] = data[timestep,wrist,0]
    data[timestep,hand,1] = data[timestep,wrist,1]

    return data

def put_missing_values(video, body_parts_class):

    pose, _, leftHand, rightHand = body_parts_class.body_part_points()
    body_section_dict = body_parts_class.body_dict()

    missing_amount = random.randrange(1,video.shape[0])
    missing_samples = random.choices(range(video.shape[0]), k=missing_amount)

    for r, pos in enumerate(missing_samples):
        #if r == 0:
        #    prepare_keypoints_image(video[pos][:,:].numpy(),"omae")

        missing_hand_type = random.randrange(3)

        if missing_hand_type == 0:
            video = replace_points(video, pos, leftHand, body_section_dict['pose_left_wrist'])
        elif missing_hand_type == 1:
            video = replace_points(video, pos, rightHand, body_section_dict['pose_right_wrist'])
        else:
            video = replace_points(video, pos, leftHand, body_section_dict['pose_left_wrist'])
            video = replace_points(video, pos, rightHand, body_section_dict['pose_right_wrist'])
        #if r == 0:
        #    prepare_keypoints_image(video[pos][:,:].numpy(),"nani")

    return video, None

def put_missing_frames(video, hidden_dim):

    # Numbers of frames to create missing landmarks
    missing_amount = random.randrange(1, video.shape[0])

    # chose randomly the number of frames you desire
    missing_samples = random.choices(range(video.shape[0]), k=missing_amount)

    if hidden_dim == None:
        mask = torch.zeros([video.shape[0]])
    else:
        mask = torch.zeros([video.shape[0]])

    for r, pos in enumerate(missing_samples):

        video[pos] = torch.zeros(video[pos].shape)
        mask[pos] = 1

    return video, mask


def filter_bad_videos(video, body_section_dict):

    is_bad = False

    # More than # value (because we delete about 12 frames 6 at the beginning ant 6 at final)
    if len(video) < 20:
        is_bad = True
    else:
        video = video[8:-8,:,:]

    for pos in range(len(video)):
        if is_bad:
            return is_bad
        
        comp_r_one = video[pos][body_section_dict['pose_right_wrist']] == video[pos][body_section_dict['rightHand_thumb_tip']]
        comp_r_two = video[pos][body_section_dict['pose_right_wrist']] == video[pos][body_section_dict['rightHand_middle_finger_dip']]
        
        comp_l_one = video[pos][body_section_dict['pose_left_wrist']] == video[pos][body_section_dict['leftHand_thumb_tip']]
        comp_l_two = video[pos][body_section_dict['pose_left_wrist']] == video[pos][body_section_dict['leftHand_middle_finger_dip']]

        if comp_r_one.any() and comp_r_two.any():
            is_bad = True
        if comp_l_one.any() and comp_l_two.any():
            is_bad = True

    return False

def filter_videos(data, body_parts_class):

    body_section_dict = body_parts_class.body_dict()
    count = 0

    print("Filtering videos ...")
    # Reverse of the list, so we can delete directly from the tensor/list/numpy
    for pos in tqdm.tqdm(range(len(data)-1, -1, -1)):

        if filter_bad_videos(data[pos], body_section_dict):
            count += 1
            data.pop(pos)
    
    print(f"The filer deletes {count} videos")
    return data

def add_sos_eos(video, mask=None):

    _, Kp_size, coord_size = video.shape
    sos = torch.ones(1, Kp_size, coord_size)  # tensor de unos
    
    eos = torch.zeros(1, Kp_size, coord_size-1)  # tensor de mitad ceros y mitad unos
    eos = torch.cat((eos,sos[:,:,-1:].clone()), dim=2)
    
    video = torch.cat([sos, video, eos], dim=0)

    if mask!=None:
        mask = torch.cat([torch.zeros(1), mask, torch.zeros(1)], dim=0)  
        mask = mask.unsqueeze(0)

    return video, mask

def delete_last_sequence(video, mask):

    video = video[:-1,:,:]
    mask = mask[:,:-1]

    return video, mask

def create_chunks(video_list, cut_size=20):

    new_dataset = []

    for video_ind in range(len(video_list)):

        video = video_list[video_ind]

        video_len = len(video)

        times = video_len // cut_size
        rest = video_len % cut_size

        if times == 0:
            new_dataset.append(video)
            continue

        for chunk in range(times):
            new_dataset.append(video[cut_size*chunk:cut_size*(chunk+1),:,:])

        if rest > 0:
            new_dataset.append(video[-cut_size:,:,:])

    new_dataset = np.array(new_dataset)
    return new_dataset


class LSP_Dataset(Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str,keypoints_model:str,  transform=None, have_aumentation=True,
                 augmentations_prob=0.5, normalize=False,landmarks_ref= 'Mapeo landmarks librerias.csv',
                 keypoints_number = 54,
                 hidden_dim=None):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """
        print("*"*20)
        print("*"*20)
        print("*"*20)
        print('Use keypoint model : ',keypoints_model) 
        logging.info('Use keypoint model : '+str(keypoints_model))

        video_dataset, body_section, body_part = get_dataset_from_hdf5(path=dataset_filename,
                                                                        keypoints_model=keypoints_model,
                                                                        landmarks_ref=landmarks_ref,
                                                                        keypoints_number = keypoints_number)
        
        self.body_parts_class = bodyKeypointMap(body_section, body_part)
        
        # HAND AND POSE NORMALIZATION
        video_dataset, keypoint_body_part_index, body_section_dict = normalize_pose_hands_function(video_dataset, self.body_parts_class)

        viedo_dataset = filter_videos(video_dataset, self.body_parts_class)

        
        self.transform = transform

        self.hidden_dim = hidden_dim
        
        self.have_aumentation = have_aumentation
        print(keypoint_body_part_index, body_section_dict)
        self.augmentation = augmentation.augmentation(keypoint_body_part_index, body_section_dict)
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

        # CREATE CHUNKS
        viedo_dataset = create_chunks(viedo_dataset)
        self.data = video_dataset


    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        depth_map = torch.from_numpy(np.copy(self.data[idx]))

        # Apply potential augmentations
        if self.have_aumentation and random.random() < self.augmentations_prob:

            selected_aug = random.randrange(4)

            if selected_aug == 0:
                depth_map_a = self.augmentation.augment_rotate(depth_map, angle_range=(-15, 15))
                    
            if selected_aug == 1:
                depth_map_a = self.augmentation.augment_shear(depth_map, "perspective", squeeze_ratio=(-0.15, 0.15))
                    
            if selected_aug == 2:
                depth_map_a = self.augmentation.augment_shear(depth_map, "squeeze", squeeze_ratio=(-0.15, 0.15))

            if selected_aug == 3:
                depth_map_a = self.augmentation.augment_arm_joint_rotate(depth_map, 0.5, angle_range=(-15, 15))

        #depth_map = depth_map - 0.5
        if self.transform:
            depth_map = self.transform(depth_map)

        # Missing landmarks
        #depth_map_missing, mask = put_missing_values(depth_map.clone(), self.body_parts_class)
        
        # Missing frames
        depth_map_missing, mask = put_missing_frames(depth_map.clone(), self.hidden_dim)

        # add SOS in the data and mask
        depth_map_missing, mask = add_sos_eos(depth_map_missing, mask)
        depth_map, _ = add_sos_eos(depth_map)

        # shift to errase the last keypoint group
        depth_map_missing, mask = delete_last_sequence(depth_map_missing, mask)

        return depth_map_missing, depth_map, mask

    def __len__(self):
        return len(self.data)
