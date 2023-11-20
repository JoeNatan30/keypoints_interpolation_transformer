
import dataloader
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_configuration

from euclidean_loss import EuclideanLoss


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
        df['x'] = df['x'].replace(0, np.nan).interpolate(method='cubic', limit_direction='both', limit_area='inside')
        df['y'] = df['y'].replace(0, np.nan).interpolate(method='cubic', limit_direction='both', limit_area='inside')
        
        interpolated_data[kp_pos][0] = torch.from_numpy(np.nan_to_num(df['x'].values))
        interpolated_data[kp_pos][1] = torch.from_numpy(np.nan_to_num(df['y'].values))

    return interpolated_data.permute(2, 0, 1)


def main():
    
    to_process = "AUTSL" #AEC #PUCP_PSL_DGI305 #AUTSL
    dataset_info = load_configuration("dataset_config")

    g = torch.Generator()
    g.manual_seed(42)
    #criterion = MSELoss()
    criterion = EuclideanLoss()

    val_set = dataloader.LSP_Dataset(f'data/validation--{to_process}.hdf5', have_aumentation=False, keypoints_model='mediapipe',is_random_missing=False)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    print("\n\n")
    loss_collector = []
    loss_original = []

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, _ = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()
        #if mask!=None:
        #    mask = mask.squeeze(0).float()
        #print(mask)
        #inputs = replace_frame_with_zeros(inputs, mask)
        original_loss = criterion(inputs[1:,:,:], sota[1:-1,:,:])
        
        prediction = cubic_interpolation(inputs[1:,:,:])

        loss = criterion(prediction, sota[1:-1,:,:])

        #print("pred:",prediction[0])
        #print("sota:",sota[1:,:,:][0])
        loss_collector.append(loss)
        loss_original.append(original_loss)


    # Crear un histograma conjunto para comparar las distribuciones
    plt.figure(figsize=(12, 8))

    # Definir rangos de bins para ambos conjuntos de datos
    bins = np.histogram_bin_edges(np.concatenate([loss_original, loss_collector]), bins=24)
    #plt.style.use('seaborn-darkgrid')
    # Dibujar histogramas con bordes y colores específicos
    plt.hist(loss_original, bins=bins, alpha=0.7, label='Original Loss', color='skyblue', edgecolor='black')
    plt.hist(loss_collector, bins=bins, alpha=0.7, label='Interpolation Loss', color='orange', edgecolor='black')

    # Agregar líneas de cuadrícula y cambiar el estilo de la cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Agregar líneas verticales para resaltar la media y cuartiles
    plt.axvline(x=np.mean(loss_original), color='blue', linestyle='dashed', linewidth=3, label='Mean Original Loss')
    plt.axvline(x=np.mean(loss_collector), color='red', linestyle='dashed', linewidth=3, label='Mean Interpolation Loss')

    # Resaltar los cuartiles con líneas adicionales
    plt.axvline(x=np.percentile(loss_original, 25), color='blue', linestyle='dashed', linewidth=1, label='Q1 Original Loss')
    plt.axvline(x=np.percentile(loss_collector, 25), color='red', linestyle='dashed', linewidth=1, label='Q1 Interpolation Loss')

    plt.axvline(x=np.percentile(loss_original, 75), color='blue', linestyle='dashed', linewidth=1, label='Q3 Original Loss')
    plt.axvline(x=np.percentile(loss_collector, 75), color='red', linestyle='dashed', linewidth=1, label='Q3 Interpolation Loss')

    # Cambiar el estilo de la leyenda para mayor claridad
    plt.legend(loc='upper right', fontsize='small')

    # Añadir un título al eje y para indicar que es la frecuencia acumulativa
    plt.ylabel('Cumulative Frequency', fontsize=14)

    # Mejorar la legibilidad y el diseño general
    plt.title('Histogram of Loss - Cubic Interpolation', fontsize=18)
    plt.xlabel('Loss', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()  # Ajustar el diseño automáticamente para evitar superposiciones


    plt.savefig(f'cubic_histogram_euclidean_{to_process}.jpg')
    #plt.legend(['Datos'])
    



main()