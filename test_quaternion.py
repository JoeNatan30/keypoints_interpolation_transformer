import dataloader
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from tqdm import tqdm

def replace_frame_with_zeros(inputs, mask):
    # Asegurarse de que 'inputs' y 'mask' tengan el mismo tamaño
    assert inputs.shape[0] == mask[0].shape[0], "Los tamaños de 'inputs' y 'mask' no coinciden"

    for pos, val in enumerate(mask[0]):
        if val.item() == 1:
            inputs[pos] = inputs[pos].zero_()

    return inputs

def translate_to_local(p, o_p2):
    return p + o_p2

def rotation_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cosTheta = np.dot(v1, v2)

    rotationAxis = np.cross(np.asarray([0., 0., 1.]), v1)
    #print("rotationAxis:",rotationAxis)
    if np.linalg.norm(rotationAxis) < 0.01:
        # special case when vectors are coincident, or very close to it
        rotationAxis = np.cross(np.asarray([1., 0., 0.]), v1)
    axis = rotationAxis # np.cross(v1, v2)
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    if (cosTheta < -1 + 0.001):
        # special case when vectors in opposite directions:
        # there is no "ideal" rotation between them
        # So we define the rotation to be 180 degrees about some arbitrary
        # perpendicular to both vectors
        angle = np.pi
    else:
        angle = np.arccos(cosTheta)

    return Quaternion(axis=axis, angle=angle)


def translate_p_2_to_origin(p1, p2, p3):
    o_p1 = p1 - p2
    o_p3 = p3 - p2
    return o_p1, o_p3

def create_interpolation_block(mask):
    
    i_limit = mask.shape[0]-1
    j_limit = mask.shape[0]-1
    
    i = 0
    j = 1
    
    saved_pos = []
    
    while(i < i_limit):
        
        # Encuentra el primer punto donde mask[i] es 1
        while i < i_limit and mask[i] == 0:
            i += 1
        
        if i == i_limit:
            break  # Se llegó al final de la máscara, salir del bucle
        
        j = i + 1  # Inicializar j en el siguiente índice
        
        # Encuentra el último punto donde mask[j] es 1
        while j < j_limit and mask[j] == 1:
            j += 1
            
        saved_pos.append((i, j))

        i = j

    return saved_pos

def quaternion_interpolation(data, mask, j_ori, j_tar):
    
    data_copy = data.clone().detach()
    interpolated_data = torch.empty_like(data_copy)
    
    time_block = create_interpolation_block(mask)
    print(time_block)
    print(data_copy.shape, mask.shape)
    
    for (time_ini, time_end) in time_block:
        for joint_tar, joint_ori in zip(j_tar, j_ori):
            
            p_ini_l1 = data_copy[time_ini][joint_ori]
            p_ini_l1 = torch.cat((p_ini_l1, torch.tensor([0.0])))
            p_end_l1 = data_copy[time_ini][joint_tar]
            p_end_l1 = torch.cat((p_end_l1, torch.tensor([0.0])))
            
            p_ini_l2 = data_copy[time_end][joint_ori]
            p_ini_l2 = torch.cat((p_ini_l2, torch.tensor([0.0])))
            p_end_l2 = data_copy[time_end][joint_tar]
            p_end_l2 = torch.cat((p_end_l2, torch.tensor([0.0])))
            
            interp_result = joints_intermediates(p_ini_l1, p_end_l1, p_ini_l2, p_end_l2, int(data_copy.shape[0]))
            
            
            for k in range(time_end - time_ini):
                interpolated_data[time_ini + k][joint_tar] = torch.tensor(interp_result[k,:2])
            
    ###
       
        
    '''
    times = np.arange(0, data.shape[0]-1)

    interpolated_data = []

    for time_ini, time_end in zip(times, times+1):

        tmp = []

        for joint_tar, joint_ori in zip(j_tar, j_ori):
        
            p_ini_l1 = data[time_ini][joint_ori]
            p_end_l1 = data[time_ini][joint_tar]
            
            p_ini_l2 = data[time_end][joint_ori]
            p_end_l2 = data[time_end][joint_tar]
            
            interp_result = joints_intermediates(p_ini_l1, p_end_l1, p_ini_l2, p_end_l2, int(61/data.shape[0]))
            tmp.append(interp_result[1:,:2])
            
        tmp = np.array(tmp)
        tmp = np.transpose(tmp, (1,2,0))
        
        interpolated_data.append(tmp)

    interpolated_data = np.array(interpolated_data)
    interpolated_data = interpolated_data.reshape((interpolated_data.shape[0]*interpolated_data.shape[1], 
                                                interpolated_data.shape[2],
                                                interpolated_data.shape[3]))
    '''

    return interpolated_data

def joints_intermediates(p_ini_l1, p_end_l1, p_ini_l2, p_end_l2, times=9):
    
    tempP = p_ini_l1 + p_ini_l2
    tempP = tempP / 2

    o_p1, o_p3 = translate_p_2_to_origin(p_end_l1, tempP, p_end_l2)

    v1 = o_p1 - np.zeros(3)
    v2 = o_p3 - np.zeros(3)

    q = rotation_between_vectors(v1, v2)
    q0 = Quaternion(axis=[1, 1, 1], degrees=0)

    q_interm = Quaternion.intermediates(q0, q, times, include_endpoints=True)

    lenV1 = np.linalg.norm(v1)
    lenV2 = np.linalg.norm(v2)

    v2_normalized = v2 / lenV2

    step = (lenV2 - lenV1) / (times + 1)

    prev = o_p1
    result = []

    for count, qi in enumerate(q_interm):
        v1_prime = qi.rotate(o_p1)
        v1_final = torch.tensor(v1_prime) + step * (count) * v2_normalized
        v1_final = translate_to_local(v1_final, tempP)

        result.append(v1_final)
        prev = v1_final

    #result.pop()

    return np.array(result)

def main():
    g = torch.Generator()
    criterion = MSELoss()

    val_set = dataloader.LSP_Dataset('validation.hdf5', have_aumentation=False, keypoints_model='mediapipe', is_random_missing=True)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    print("\n\n")
    loss_collector = []
    
    df_keypoints = pd.read_csv('points_54.csv')

    j_ori = df_keypoints.origin.values-1
    j_tar = df_keypoints.tarjet.values-1

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, mask = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()
        if mask is not None:
            mask = mask.squeeze(0).float()

        inputs = replace_frame_with_zeros(inputs, mask)

        # Adaptar la función de interpolación según tu necesidad
        prediction = quaternion_interpolation(inputs[1:, :, :], mask[0][1:], j_ori, j_tar)

        loss = criterion(prediction, sota[1:-1, :, :])
        loss_collector.append(loss)

    print(sum(loss_collector) / len(loss_collector))

if __name__ == "__main__":
    main()
