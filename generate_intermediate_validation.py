from torch.utils.data import DataLoader

import dataloader
import torch
from tqdm import tqdm
import h5py
import numpy as np

from utils import load_configuration

def generate_h5_metadata(h5_file, group_name):
    
    group = h5_file.create_group(group_name)
    
    group.create_dataset('x', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen='float32'), chunks=True)
    group.create_dataset('y', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen='float32'), chunks=True)
    group.create_dataset('x_mask', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen='int'), chunks=True)
    group.create_dataset('y_mask', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen='int'), chunks=True)
    group.create_dataset('length', shape=(0,), maxshape=(None,), dtype='int')
    group.create_dataset('shape', shape=(2,), maxshape=(None,), dtype='int')

    return group

def resize_dataset(dataset, value):
    dataset.resize((len(dataset) + 1,))
    dataset[-1] = value
 

def Generate_intermediate_h5():

    to_process = "PUCP_PSL_DGI305" #AEC #PUCP_PSL_DGI305 #AUTSL
    dataset_info = load_configuration("dataset_config")

    device = torch.device("cpu")

    g = torch.Generator()
    g.manual_seed(42)

    val_set = dataloader.LSP_Dataset(f'data/training--{to_process}.hdf5', have_aumentation=False, keypoints_model='mediapipe',is_random_missing=False)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    validation_h5_file = h5py.File(f'data/training_intermediate--{to_process}.hdf5', 'w')

    interVal_group = generate_h5_metadata(validation_h5_file, "intermediate")

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, mask = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()

        x = inputs.squeeze(0).to(device).float()[:-1,:,:]                   #  [Start] + [keypoints]
        x_no_sota = inputs.clone().squeeze(0).to(device).float()[1:,:,:]    #  [keypoints] + [end]
        
        y = sota.squeeze(0).to(device).float() #  [keypoints] + [end]
        
        mask = mask.squeeze(0).to(device).float()

        x_mask = mask[:-1].clone().detach() #  [Start] + [keypoints]
        y_mask = mask[1:].clone().detach()  #  [keypoints] + [end] 

        
 
        mask_expanded = x_mask.bool()[:, None, None].expand(-1, inputs.size(1), inputs.size(2)).to(device)
        x = torch.where(mask_expanded, torch.zeros_like(x).float().to(device), x)


        print(x.shape, y.shape, x_mask.shape, y_mask.shape)
        resize_dataset(interVal_group['x'], x.flatten())
        resize_dataset(interVal_group['y'], y.flatten())
        resize_dataset(interVal_group['x_mask'], x_mask)
        resize_dataset(interVal_group['y_mask'], y_mask)
        resize_dataset(interVal_group['length'], x.shape[0])
    
    interVal_group['shape'][:] = x.shape[1:]


    validation_h5_file.close()

def read_intermediate_h5():

    to_process = "PUCP_PSL_DGI305"

    data = h5py.File(f'data/training_intermediate--{to_process}.hdf5', 'r')


    group = data['intermediate']

    _x = group['x']
    _y = group['y']
    _x_mask = group['x_mask']
    _y_mask = group['y_mask']
    _length = group['length']
    _shape = group['shape']
    _shape = np.array(_shape)
    
    print("N° of data:",_x.shape)

    x = [np.transpose(np.array(value).reshape(length, _shape[0], _shape[1]), (0,2,1)) for value, length in zip(_x, _length)]
    y = [np.transpose(np.array(value).reshape(length, _shape[0], _shape[1]), (0,2,1)) for value, length in zip(_y, _length)]
    x_mask = [np.array(value).reshape(length) for value, length in zip(_x_mask, _length)]
    y_mask = [np.array(value).reshape(length) for value, length in zip(_y_mask, _length)]

    for idx in range(len(x)):
        print(x[idx].shape, y[idx].shape, x_mask[idx].shape, y_mask[idx].shape)

   
# Generate_intermediate_h5()
read_intermediate_h5()