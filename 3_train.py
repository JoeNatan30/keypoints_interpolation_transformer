
from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
#from spoter.gaussian_noise import GaussianNoise
import matplotlib.pyplot as plt
import os
import wandb
import model
import dataloader
import augmentation

from euclidean_loss import EuclideanLoss

import parseMain
import numpy as np
import pandas as pd

from utils import prepare_keypoints_image, get_edges_index


np.random.seed(42)
pd.np.random.seed(42)
torch.manual_seed(42)


CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "fill_missings_transformer"
ENTITY = "joenatan30" #joenatan30
TAG = ["paper"]

#os.environ["WANDB_API_KEY"] = "c16c54799944a6127132bcb81b2fb9ebcb4fe5db"

connections = np.moveaxis(np.array(get_edges_index('54')), 0, 1)

def lr_lambda(current_step, lr, optim):
    
    #if current_step <= 50:
    #    lr_rate = current_step/50000  # Función lineal
    #else:
    #    lr_rate = (0.00005/current_step) ** 0.5  # Función de raíz cuadrada inversa

    lr_rate = lr

    print(f'[{current_step}], Lr_rate: {lr_rate}')
    optim.param_groups[0]['lr'] = lr_rate

    return optim

def sent_histogram(loss_original, loss_collector, to_process):
    
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

    # Mejorar la legibilidad y el diseño general
    plt.title('Histogram of Loss - Cubic Interpolation', fontsize=18)
    plt.xlabel('Loss', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()  # Ajustar el diseño automáticamente para evitar superposiciones

    plt.savefig(f'IA_histogram_{to_process}.jpg')
    wandb.log({"IA_histogram": [wandb.Image(f"IA_histogram_{to_process}.jpg", caption="histogram - Interpolaion IA")]})


def sent_test_result(model, inputs, mask, device):

    tgt_mask = model.get_tgt_mask(len(inputs)).to(device)

    pred = model(inputs, inputs, decoder_mask=mask, tgt_mask=tgt_mask)

    pred_images = prepare_keypoints_image(pred[0], connections, 0, "Test")
    for _rel_pos in range(1, len(inputs)):
        pred_images = np.concatenate((pred_images, prepare_keypoints_image(pred[_rel_pos], connections, _rel_pos)), axis=1)

    images = wandb.Image(pred_images, caption="Validation")
    wandb.log({"examples of test": images})

def sent_validation_result(inputs, prediction, sota, connections):

    # add input
    input_images = prepare_keypoints_image(inputs[0], connections, -1, "Input")
    for _rel_pos in range(1, len(inputs)):
        input_images = np.concatenate((input_images, prepare_keypoints_image(inputs[_rel_pos], connections, _rel_pos-1)), axis=1)

    # add output
    prediction_images = prepare_keypoints_image(prediction[0], connections, 0, "Prediction")
    for _rel_pos in range(1, len(prediction)):
        prediction_images = np.concatenate((prediction_images, prepare_keypoints_image(prediction[_rel_pos], connections, _rel_pos)), axis=1)

    # add sota
    sota_images = prepare_keypoints_image(sota[0], connections, 0, "Sota")
    for _rel_pos in range(1, len(sota)):
        sota_images = np.concatenate((sota_images, prepare_keypoints_image(sota[_rel_pos],  connections, _rel_pos)), axis=1)

    output = np.concatenate((input_images, prediction_images, sota_images), axis=0)
    images = wandb.Image(output, caption="Validation")
    wandb.log({"examples_validation epoch": images})


    ## TEST
    '''
    # send Test
    _, Kp_size, coord_size = inputs.shape

    #          Y_recursive starts as <SOS>
    y_recursive = torch.ones(1, inputs.shape[1], inputs.shape[2]).to(device) # SOS
    #          We save the first (empty) frame
    test_images = prepare_keypoints_image(y_recursive[0], connections, -1, "Test")

    eos = torch.zeros(1, Kp_size, coord_size-1).to(device)  # tensor de mitad ceros y mitad unos
    eos = torch.cat((eos,y_recursive[:,:,-1:].clone()), dim=2)
    
    for _rel_pos in range(1, len(inputs)+5):

        tgt_mask = model.get_tgt_mask(len(y_recursive)).to(device)

        pred = model(inputs, y_recursive, tgt_mask=tgt_mask)

        #append keypoints
        if len(pred) == 1:
            y_recursive = torch.cat((y_recursive, pred), dim=0)
        else:
            y_recursive = torch.cat((y_recursive, pred[-1:]), dim=0)

        #append image
        test_images = np.concatenate((test_images, prepare_keypoints_image(y_recursive[_rel_pos], connections, _rel_pos-1)), axis=1)

        next_item_check = pred == eos
        if next_item_check.all():
            break

    images = wandb.Image(test_images, caption="Test Output")
    wandb.log({"examples of test": images})
    '''

def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0

    data_length = len(dataloader)

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data

        inputs = inputs.squeeze(0).to(device).float()
        sota = sota.squeeze(0).to(device).float()

        if mask!=None:
            mask = mask.squeeze(0).to(device).float()
        
        
        # Use Batch
        if len(inputs.shape) != 3: # is 4 if you are using batch
            tgt_mask = model.get_tgt_mask(inputs.shape[1]).to(device)
            prediction = model(inputs, sota[:,:-1,:,:], coder_mask=mask, tgt_mask=tgt_mask)
            loss = criterion(prediction, sota[:,1:,:,:])
        # No use Batch
        else:
            tgt_mask = model.get_tgt_mask(inputs.shape[0]).to(device)
            prediction = model(inputs, sota[:-1,:,:], coder_mask=mask, tgt_mask=tgt_mask)
            loss = criterion(prediction, sota[1:,:,:])

        loss = loss.float()    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

    return running_loss/data_length

def eval_epoch(model, dataloader, criterion, body_parts_class, dataset_name, device):

    model.eval()
    running_loss = 0.0

    data_length = len(dataloader)
    
    loss_collector = []
    loss_original = []

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data

        inputs = inputs.squeeze(0).to(device).float()
        sota = sota.squeeze(0).to(device).float()

        if mask!=None:
            mask = mask.squeeze(0).to(device).float()

        # Use Batch
        #if len(inputs.shape) != 3: # is 4 if you are using batch
        #    tgt_mask = model.get_tgt_mask(inputs.shape[1]).to(device)
        #    prediction = model(inputs, sota[:,:-1,:,:], coder_mask=mask, tgt_mask=tgt_mask)
        #    loss = criterion(prediction, sota[:,1:,:,:])
        # No use Batch
        #else:
        tgt_mask = model.get_tgt_mask(inputs.shape[0]).to(device)
        prediction = model(inputs, sota[:-1,:,:], coder_mask=mask, tgt_mask=tgt_mask)
        loss = criterion(prediction[:-1,:,:], sota[1:-1,:,:])
        original_loss = criterion(inputs[1:,:,:], sota[1:-1,:,:])
        loss_collector.append(loss.clone().detach().cpu().numpy())
        loss_original.append(original_loss.clone().detach().cpu().numpy())

        loss = loss.float()    
        running_loss += loss

        # Print in WandB
        if i == 1:
            # Use Batch
            if len(inputs.shape) != 3:
                batch_pos = 1
                sent_validation_result(inputs[batch_pos], prediction[batch_pos], sota[batch_pos,1:,:,:], connections)
                sent_test_result(model, inputs[batch_pos], mask[batch_pos], device)
            # No use Batch
            else:
                sent_validation_result(inputs, prediction, sota[1:,:,:], connections)
                sent_test_result(model, inputs, mask, device)
    
    
    sent_histogram(loss_original, loss_collector, dataset_name)
                

    return running_loss/data_length

def train(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    g = torch.Generator()
    g.manual_seed(42)

    #transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = dataloader.LSP_Dataset(args.training_set_path,
                                       have_aumentation=True,
                                       keypoints_model='mediapipe',
                                       is_random_missing=False)

    body_parts_class = train_set.body_parts_class

    val_set = dataloader.LSP_Dataset(args.validation_set_path,
                                        keypoints_model='mediapipe',
                                        have_aumentation=False,
                                        is_random_missing=False)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1, generator=g)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, generator=g)
    
    keysecom_model = model.KeypointCompleter(input_size=54*2, hidden_dim=128, num_layers=6)
    wandb.watch(keysecom_model)
    criterion = EuclideanLoss()#MSELoss()
    optimizer = Adam(keysecom_model.parameters(), lr=0.0001)

    keysecom_model.train(True)
    keysecom_model.to(device)

    epoch_start = 0

    losses = []

    # TRAINING AND EVALUATION

    for epoch in range(epoch_start, args.epochs):

        optimizer = lr_lambda(epoch, args.lr, optimizer)
        print("epoch:",epoch)
        train_loss = train_epoch(keysecom_model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(keysecom_model, val_loader, criterion, body_parts_class,
                              val_set.dataset_name, device)
        print("train loss:", train_loss)
        print("eval loss:", val_loss)

        wandb.log({
            'train_loss': train_loss,
            'val_loss':val_loss,
            'epoch': epoch
        })

        losses.append(train_loss.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[parseMain.get_default_args()], add_help=False)
    args = parser.parse_args()

    run = wandb.init(project=PROJECT_WANDB,
                     entity=ENTITY,
                     config=args,
                     name=args.experiment_name,
                     job_type="model-training",
                     tags=TAG)

    config = wandb.config

    train(args)