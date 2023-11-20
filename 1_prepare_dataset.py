import os
import pandas as pd
import numpy as np
import h5py
from utils import load_configuration

def read_csv_file(csv_file):
    return pd.read_csv(csv_file, encoding='utf-8')

def shuffle_and_split_data(video_paths, seed_value=42, split_ratio=0.8):
    np.random.seed(seed_value)
    shuffled_paths = video_paths.sample(frac=1, random_state=seed_value).reset_index(drop=True)
    split_index = int(split_ratio * len(shuffled_paths))
    train_data = shuffled_paths[:split_index]
    val_data = shuffled_paths[split_index:]
    
    return train_data, val_data

def generate_h5_metadata(h5_file, group_name):
    
    group = h5_file.create_group(group_name)
    
    data = group.create_dataset('data', shape=(0, ), maxshape=(None,), dtype=h5py.special_dtype(vlen='float32'), chunks=True)
    length = group.create_dataset('length', shape=(0,), maxshape=(None,), dtype='int')
    label = group.create_dataset('label', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    videoName = group.create_dataset('video_name', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    shape = group.create_dataset('shape', shape=(2,), maxshape=(None,), dtype='int')

    return group

def resize_dataset(dataset, value):
    dataset.resize((len(dataset) + 1,))
    dataset[-1] = value

def fill_h5(group, missing):
    
    _shape = None
    
    for m_data, m_name, m_label in zip(missing['data'], missing['Video Name'], missing['Class']):
        _data = m_data
        _name = m_name
        _label = m_label
        _shape = _data.shape[1:]
        _length = _data.shape[0]
        resized_data = _data.flatten()
        
        resize_dataset(group['data'], resized_data)
        resize_dataset(group['length'], _length)
        resize_dataset(group['label'], _label)
        resize_dataset(group['video_name'], _name)
    
    group['shape'][:] = _shape


def got_h5_data(path_list, info_dict, idx_keypoints):
    
    path_list['data'] = None

    for group_name in info_dict:
        
        group = info_dict[group_name]

        #group['data']
        #group['label']
        #group['video_name']
        
        g_name = group['video_name'][...].item().decode('utf-8')
        
        index = path_list.index[path_list['Video Name'].astype(str).isin([g_name])].tolist()

        if index != []:
            path_list.at[index[0], 'data'] = group['data'][:,:,idx_keypoints]
    
    return path_list

def main():
    
    to_process = "PUCP_PSL_DGI305"
    
    dataset_info = load_configuration("dataset_config")#pd.read_csv('./dataset_info.csv', encoding='utf-8')
    
    df_keypoints = pd.read_csv('Mapeo landmarks librerias.csv', skiprows=1)
    df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]
    idx_keypoints = sorted(df_keypoints['mp_indexInArray'].astype(int).values)
    
    missing_train = pd.DataFrame()
    missing_valid = pd.DataFrame()
    
    for dataset, config  in dataset_info.items():
        
        if dataset != to_process and to_process != "all":
            continue
        
        config = {k: v for k, v in config.items()}


        h5_path = config.get('hdf5_file', None)
        csv_path = config.get('csv_file', None)
        
        print(h5_path)

        # leer h5
        h5_file = h5py.File(h5_path, 'r')
        
        # obtener lista
        no_missing = read_csv_file(csv_path)
        
        # obtener datos h5
        data_h5 = got_h5_data(no_missing, h5_file, idx_keypoints)
        
        # separar data
        train_data, valid_data = shuffle_and_split_data(data_h5)
        
        # acumular datos
        missing_train = pd.concat([missing_train, train_data], ignore_index=True)
        missing_valid = pd.concat([missing_valid, valid_data], ignore_index=True)

        # cerrar h5
        h5_file.close()
    
    print(missing_train)
    # crear nuevo h5 file

    train_h5_file = h5py.File(f'data/training--{to_process}.hdf5', 'w')
    valid_h5_file = h5py.File(f'data/validation--{to_process}.hdf5', 'w')
    
    train_group = generate_h5_metadata(train_h5_file, 'no_missing')
    valid_group = generate_h5_metadata(valid_h5_file, 'no_missing')
    
    fill_h5(train_group,missing_train)
    fill_h5(valid_group,missing_valid)
    
    train_h5_file.close()
    valid_h5_file.close()

    
if __name__ == "__main__":
    main()
