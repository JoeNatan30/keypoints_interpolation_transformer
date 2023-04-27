
from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
#from spoter.gaussian_noise import GaussianNoise

import os
import wandb
import model
import dataloader
import augmentation

import parseMain
import numpy as np

from utils import prepare_keypoints_image, get_edges_index

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "fill_missings_transformer"
ENTITY = "joenatan30" #joenatan30
TAG = ["paper"]

os.environ["WANDB_API_KEY"] = "8b69e5a1943f75b652c694fbe3875c3216e3fbe6"

connections = np.moveaxis(np.array(get_edges_index('54')), 0, 1)



def lr_lambda(current_step, optim):
    
    #if current_step <= 50:
    #    lr_rate = current_step/50000  # Función lineal
    #else:
    #    lr_rate = (0.00005/current_step) ** 0.5  # Función de raíz cuadrada inversa

    lr_rate = 0.000005

    print(f'[{current_step}], Lr_rate: {lr_rate}')
    optim.param_groups[0]['lr'] = lr_rate

    return optim

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    data_length = len(dataloader)

    for i, data in enumerate(dataloader):
        inputs, trueOut, mask = data

        inputs = inputs.squeeze(0).to(device).float()
        trueOut = trueOut.squeeze(0).to(device).float()

        if mask!=None:
            mask = mask.squeeze(0).to(device).float()
        
        tgt_mask = model.get_tgt_mask(inputs.shape[0]).to(device)
        outputs = model(inputs, trueOut[:-1,:,:], mask, tgt_mask)

        loss = criterion(outputs, trueOut[1:,:,:])
        loss = loss.float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

    return running_loss/data_length

def eval_epoch(model, dataloader, criterion, body_parts_class, device):

    model.eval()
    running_loss = 0.0

    data_length = len(dataloader)

    for i, data in enumerate(dataloader):

        inputs, trueOut, mask = data

        inputs = inputs.squeeze(0).to(device).float()
        trueOut = trueOut.squeeze(0).to(device).float()


        tgt_mask = model.get_tgt_mask(inputs.shape[0]).to(device)
        outputs = model(inputs, trueOut[:-1,:,:], tgt_mask=tgt_mask)

        loss = criterion(outputs, trueOut[1:,:,:])
        loss = loss.float()
        running_loss += loss

        # Print in WandB
        if i == 1:
            # send validation output
            trueOut = trueOut[1:,:,:]
            outputs = outputs[:,:,:]
            inputs = inputs[:,:,:]

            # add input
            input_images = prepare_keypoints_image(inputs[0], connections, -1, "input")
            for _rel_pos in range(1, len(inputs)):
                input_images = np.concatenate((input_images, prepare_keypoints_image(inputs[_rel_pos], connections, _rel_pos-1)), axis=1)

            # add output
            output_images = prepare_keypoints_image(outputs[0], connections, 0, "prediction")
            for _rel_pos in range(1, len(outputs)):
                output_images = np.concatenate((output_images, prepare_keypoints_image(outputs[_rel_pos], connections, _rel_pos)), axis=1)
        
            # add trueOut
            trueOut_images = prepare_keypoints_image(trueOut[0], connections, 0, "trueOut")
            for _rel_pos in range(1, len(trueOut)):
                trueOut_images = np.concatenate((trueOut_images, prepare_keypoints_image(trueOut[_rel_pos],  connections, _rel_pos)), axis=1)

            output = np.concatenate((input_images, output_images, trueOut_images), axis=0)
            images = wandb.Image(output, caption="Validation")
            wandb.log({"examples_validation epoch": images})

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

    return running_loss/data_length

def train(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    #transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = dataloader.LSP_Dataset(args.training_set_path, 
                                       have_aumentation=True, 
                                       keypoints_model='mediapipe',
                                       hidden_dim = 128)

    body_parts_class = train_set.body_parts_class

    val_set = dataloader.LSP_Dataset(args.validation_set_path, 
                                       have_aumentation=False, 
                                       keypoints_model='mediapipe')

    train_loader = DataLoader(train_set, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, shuffle=False, generator=g)
    
    keysecom_model = model.KeypointCompleter(input_size=54*2, hidden_dim=128, num_layers=6)
    wandb.watch(keysecom_model)
    criterion = MSELoss()
    optimizer = Adam(keysecom_model.parameters(), lr=0.0001)

    keysecom_model.train(True)
    keysecom_model.to(device)

    epoch_start = 0

    losses = []

    # TRAINING AND EVALUATION

    for epoch in range(epoch_start, args.epochs):

        optimizer = lr_lambda(epoch, optimizer)
        print("epoch:",epoch)
        train_loss = train_epoch(keysecom_model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(keysecom_model, val_loader, criterion, body_parts_class, device)
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