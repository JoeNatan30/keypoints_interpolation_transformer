
import dataloader
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


np.random.seed(42)
pd.np.random.seed(42)
torch.manual_seed(42)


def replace_frame_with_zeros(inputs, mask):
    # Asegurarse de que 'inputs' y 'mask' tengan el mismo tamaño
    assert inputs.shape[0] == mask[0].shape[0], "Los tamaños de 'inputs' y 'mask' no coinciden"

    for pos, val in enumerate(mask[0]):
        if val.item() == 1:
            inputs[pos] = inputs[pos].zero_()

    return inputs
    

def cubic_interpolation(data):
    data_copy = data.clone().detach().permute(1, 2, 0)
    interpolated_data = torch.empty_like(data_copy)
    
    for kp_pos in np.arange(0, data_copy.shape[0]):
        
        x = data_copy[kp_pos][0]
        y = data_copy[kp_pos][1]

        df = pd.DataFrame({'x': x, 'y': y})
        df['time'] = np.arange(len(df))
        
        df['x'] = df['x'].replace(0, np.nan).interpolate(method='cubic', limit_direction='both')
        df['y'] = df['y'].replace(0, np.nan).interpolate(method='cubic', limit_direction='both')

        interpolated_data[kp_pos][0] = torch.from_numpy(np.nan_to_num(df['x'].values))
        interpolated_data[kp_pos][1] = torch.from_numpy(np.nan_to_num(df['y'].values))

    return interpolated_data.permute(2, 0, 1)


def main():

    g = torch.Generator()
    criterion = MSELoss()

    val_set = dataloader.LSP_Dataset('validation.hdf5', have_aumentation=False, keypoints_model='mediapipe',is_random_missing=True)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    print("\n\n")
    loss_collector = []

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, mask = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()
        if mask!=None:
            mask = mask.squeeze(0).float()
        #print(mask)
        inputs = replace_frame_with_zeros(inputs, mask)
        
        prediction = cubic_interpolation(inputs[1:,:,:])

        loss = criterion(prediction, sota[1:-1,:,:])
        #print("pred:",prediction[0])
        #print("sota:",sota[1:,:,:][0])
        loss_collector.append(loss)


    #print(sum(loss_collector)/len(loss_collector))
    # Crear un histograma
    plt.hist(loss_collector, bins=24, edgecolor='black', color='skyblue', alpha=0.7)  # Ajusta alpha para transparencia
    # Agregar líneas de cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Histogram of loss - Cubic')
    plt.xlabel('Loss')
    plt.ylabel('Frequence')
    plt.savefig('cubic_histogram.jpg')
    plt.legend(['Datos'])
    
main()