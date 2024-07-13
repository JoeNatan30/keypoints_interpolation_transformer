from torch.utils.data import DataLoader

import dataloader
import torch
from tqdm import tqdm

from utils import load_configuration

def main():

    to_process = "AUTSL" #AEC #PUCP_PSL_DGI305 #AUTSL
    dataset_info = load_configuration("dataset_config")

    g = torch.Generator()
    g.manual_seed(42)

    val_set = dataloader.LSP_Dataset(f'data/validation--{to_process}.hdf5', have_aumentation=False, keypoints_model='mediapipe',is_random_missing=False)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)

    for i, data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Procesando datos")):
        inputs, sota, mask = data
        inputs = inputs.squeeze(0).float()
        sota = sota.squeeze(0).float()

        print(sota.shape)



main()