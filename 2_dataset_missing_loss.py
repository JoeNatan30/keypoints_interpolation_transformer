
import dataloader
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_configuration


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


def main():
    
    to_process = "PUCP_PSL_DGI305" #AEC #PUCP_PSL_DGI305 #AUTSL
    dataset_info = load_configuration("dataset_config")

    g = torch.Generator()
    criterion = MSELoss()

    val_set = dataloader.LSP_Dataset(f'data/validation--{to_process}.hdf5', have_aumentation=False, keypoints_model='mediapipe',is_random_missing=False)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    print("\n\n")
    loss_collector = []

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, _ = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()

        loss = criterion(inputs[1:,:,:], sota[1:-1,:,:])

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
    plt.savefig(f'results/initial_loss_{to_process}.jpg')
    plt.legend(['Datos'])

    
main()