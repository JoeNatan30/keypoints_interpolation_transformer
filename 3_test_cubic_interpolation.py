
import dataloader
from torch.utils.data import DataLoader
from torch.nn import MSELoss, SmoothL1Loss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_configuration

from scipy.stats import f_oneway, ttest_ind, tukey_hsd
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
    

def cubic_interpolation(data, mask):
    data_copy = data.clone().detach().permute(1, 2, 0)
    interpolated_data = torch.empty_like(data_copy)
    #print(data_copy.shape, mask.shape)
    for _pos, _value in enumerate(mask[0]):
 
        if _value == 1:
            #print(data_copy.shape, _pos)
            data_copy[:,:,_pos] = torch.zeros(data_copy[:,:,_pos].shape)
    
    for kp_pos in range(data_copy.shape[0]):
        x = data_copy[kp_pos][0]
        y = data_copy[kp_pos][1]

        df = pd.DataFrame({'x': x, 'y': y})
        df['time'] = np.arange(len(df))

        # Interpola incluso si los primeros valores son NaN
        df['x'] = df['x'].replace(0, np.nan).interpolate(method='cubicspline', limit_direction='both', limit_area=None)
        df['y'] = df['y'].replace(0, np.nan).interpolate(method='cubicspline', limit_direction='both', limit_area=None)
        # df['x'] = df['x'].replace(0, np.nan).interpolate(method='cubicspline', limit_direction='backward', limit_area='outside')
        # df['y'] = df['y'].replace(0, np.nan).interpolate(method='cubicspline', limit_direction='backward', limit_area='outside')

        interpolated_data[kp_pos][0] = torch.from_numpy(np.nan_to_num(df['x'].values))
        interpolated_data[kp_pos][1] = torch.from_numpy(np.nan_to_num(df['y'].values))
    
    return interpolated_data.permute(2, 0, 1)


def main():
    
    to_process = "AEC" #AEC #PUCP_PSL_DGI305 #AUTSL
    dataset_info = load_configuration("dataset_config")

    g = torch.Generator()
    g.manual_seed(42)
    #criterion = MSELoss()
    #criterion = SmoothL1Loss(beta=5.0)
    criterion = EuclideanLoss()

    val_set = dataloader.LSP_Dataset(f'data/validation--{to_process}.hdf5', have_aumentation=False, keypoints_model='mediapipe',is_random_missing=False)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    print("\n\n")
    loss_collector_acum = []
    loss_baseline_acum = []

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, mask = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()
        #if mask!=None:
        #    mask = mask.squeeze(0).float()
        #print(mask)
        #inputs = replace_frame_with_zeros(inputs, mask)
        baseline_loss = criterion(inputs, sota)
        #print(mask)
        prediction = cubic_interpolation(inputs, mask)

        loss = criterion(prediction, sota)

        #print("pred:",prediction[0])
        #print("sota:",sota[1:,:,:][0])
        loss_collector_acum.append(loss)
        loss_baseline_acum.append(baseline_loss)


    # Crear un histograma conjunto para comparar las distribuciones
    plt.figure(figsize=(12, 8))

    # Definir rangos de bins para ambos conjuntos de datos
    bins = np.histogram_bin_edges(np.concatenate([loss_baseline_acum, loss_collector_acum]), bins=24)
    #plt.style.use('seaborn-darkgrid')
    # Dibujar histogramas con bordes y colores específicos
    plt.hist(loss_baseline_acum, bins=bins, alpha=0.7, label='Baseline Loss', color='skyblue', edgecolor='black')
    plt.hist(loss_collector_acum, bins=bins, alpha=0.7, label='Cubic I. Loss', color='orange', edgecolor='black')

    # Agregar líneas de cuadrícula y cambiar el estilo de la cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Agregar líneas verticales para resaltar la media
    #plt.axvline(x=np.mean(loss_baseline_acum), color='blue', linestyle='dashed', linewidth=2, label='Mean Original Loss')
    #plt.axvline(x=np.mean(loss_collector_acum), color='red', linestyle='dashed', linewidth=2, label='Mean Interpolation Loss')
    # Agregar líneas verticales para resaltar la mediana
    plt.axvline(x=np.median(loss_baseline_acum), color='blue', linestyle='dashed', linewidth=3, label='Median Baseline Loss')
    plt.axvline(x=np.median(loss_collector_acum), color='orange', linestyle='dashed', linewidth=3, label='Median Interpolation Loss')

    # Resaltar los cuartiles con líneas adicionales
    #plt.axvline(x=np.percentile(loss_baseline_acum, 25), color='blue', linestyle='dashed', linewidth=1, label='Q1 Original Loss')
    #plt.axvline(x=np.percentile(loss_collector_acum, 25), color='red', linestyle='dashed', linewidth=1, label='Q1 Interpolation Loss')
    #plt.axvline(x=np.percentile(loss_baseline_acum, 75), color='blue', linestyle='dashed', linewidth=1, label='Q3 Original Loss')
    #plt.axvline(x=np.percentile(loss_collector_acum, 75), color='red', linestyle='dashed', linewidth=1, label='Q3 Interpolation Loss')

    # Cambiar el estilo de la leyenda para mayor claridad
    plt.legend(loc='upper right', fontsize='small')

    # Añadir un título al eje y para indicar que es la frecuencia acumulativa
    plt.ylabel('Cumulative Frequency', fontsize=14)

    # Mejorar la legibilidad y el diseño general
    plt.title('Histogram of Loss - Cubic Interpolation', fontsize=18)
    plt.xlabel('Loss', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()  # Ajustar el diseño automáticamente para evitar superposiciones


    plt.savefig(f'results/cubic_histogram_freq_{to_process}.jpg')
    #plt.legend(['Datos'])
    


    all_losses = [loss_baseline_acum, loss_collector_acum]
    medians = [np.median(loss) for loss in all_losses]
    labels = ['Baseline', 'Cubic I.'] 

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.violinplot(all_losses, showmedians=True)
    ax.plot(np.arange(1, len(labels) + 1), medians, marker='o', linestyle='None', color='blue', label='median')

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Personaliza el título y las etiquetas de los ejes
    plt.title('Loss Comparison: Cubic Interpolation vs. Baseline', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Agrega una leyenda para la línea de la mediana
    plt.legend()
    
    plt.savefig(f"results/cubic_histogram_{to_process}.jpg")
    
    
    # Realiza el análisis de varianza (ANOVA)
    f_stat, p_value = f_oneway(*all_losses)
    
    # Imprime los resultados
    print(f"F-statistic: {f_stat}, p-value: {p_value}")

    # Compara con un nivel de significancia (por ejemplo, 0.05)
    if p_value < 0.05:
        print("Hay diferencias significativas entre al menos dos grupos.")
    else:
        print("No hay diferencias significativas entre los grupos.")
    
    # Realiza la prueba de Tukey como prueba post hoc
    tukey_results = tukey_hsd(*all_losses)
    print(tukey_results)
    
    # Realiza la prueba t de Student
    t_stat, p_value = ttest_ind(*all_losses)
    print(f"T-statistic: {t_stat}, p-value: {p_value}")


main()