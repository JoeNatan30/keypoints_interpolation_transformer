

import argparse
import os
import parseMain

#from spoter.gaussian_noise import GaussianNoise


import numpy as np
import pandas as pd
import wandb
import model
import dataloader
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, tukey_hsd

from euclidean_loss import EuclideanLoss
from utils import prepare_keypoints_image, get_edges_index


np.random.seed(42)
#pd.np.random.seed(42)
torch.manual_seed(42)


CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "fill_missings_transformer"
ENTITY = "joenatan30" #joenatan30
TAG = ["cycle"]

connections = np.moveaxis(np.array(get_edges_index('54')), 0, 1)


loss_baseline_acum = []
loss_cubic_acum = []

def lr_lambda(current_step, lr, optim):
    
    #if current_step <= 50:
    #    lr_rate = current_step/50000  # Función lineal
    #else:
    #    lr_rate = (0.00005/current_step) ** 0.5  # Función de raíz cuadrada inversa
    lr_rate = lr[current_step]

    for param_group in optim.param_groups:
        param_group['lr'] = lr_rate

    return optim

def cubic_interpolation(data, mask):
    data_copy = data.clone().detach().permute(1, 2, 0).cpu()
    interpolated_data = torch.empty_like(data_copy)
    
    for _pos, _val in enumerate(mask):
 
        if _val == 1:
            #print(data_copy.shape, _pos)
            data_copy[:,:,_pos] = torch.zeros(data_copy[:,:,_pos].shape)

    for kp_pos in np.arange(0, data_copy.shape[0]):
        
        x = data_copy[kp_pos][0]
        y = data_copy[kp_pos][1]

        df = pd.DataFrame({'x': x, 'y': y})
        df['time'] = np.arange(len(df))

        df['x'] = df['x'].replace(0, np.nan).interpolate(method='cubicspline', limit_direction='both', limit_area=None)
        df['y'] = df['y'].replace(0, np.nan).interpolate(method='cubicspline', limit_direction='both', limit_area=None)
        
        interpolated_data[kp_pos][0] = torch.from_numpy(np.nan_to_num(df['x'].values))
        interpolated_data[kp_pos][1] = torch.from_numpy(np.nan_to_num(df['y'].values))

    return interpolated_data.permute(2, 0, 1)

def sent_histogram(loss_baseline_acum, loss_collector_acum, loss_cubic_acum, to_process, epoch, bins=24, figsize=(12, 8)):

    all_losses = [loss_baseline_acum, loss_collector_acum, loss_cubic_acum]
    medians = [np.median(loss) for loss in all_losses]
    labels = ['Baseline', 'AI', "Cubicspline"] 

    fig, ax = plt.subplots(figsize=(8, 6))

    # Crea los violines
    violins = ax.violinplot(all_losses, showmedians=True)
    colors = ['steelblue', 'brown', 'orange']  # Cambia los colores según tus preferencias

    for i, violin in enumerate(violins['bodies']):
        violin.set_facecolor(colors[i])
        violin.set_edgecolor('black')
        violin.set_alpha(0.7)

    # Agrega etiquetas a los violines
    for i, label in enumerate(labels, start=1):
        violins['bodies'][i - 1].set_label(label)

    # Agrega puntos para representar las medianas
    #ax.plot(np.arange(1, len(labels) + 1), medians, marker='o', linestyle='None', color='blue', label='median')

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Personaliza el título y las etiquetas de los ejes
    plt.title('Loss Comparison: Cubic Interpolation vs. Baseline', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Agrega una leyenda para la línea de la mediana
    plt.legend()
    
    plt.savefig(f'results/IA_histogram_{to_process}.jpg')
    wandb.log({"IA_histogram": [wandb.Image(f"results/IA_histogram_{to_process}.jpg", caption="histogram - Interpolation IA")]}, step=epoch)

    # ### ### ### ### ###

    # Realiza el análisis de varianza (ANOVA)
    f_stat, p_value = f_oneway(*all_losses)
    
    # Imprime los resultados
    print(f"F-statistic: {f_stat}, p-value: {p_value}")

    # Compara con un nivel de significancia (por ejemplo, 0.05)
    if p_value < 0.05:
        print("Hay diferencias significativas entre al menos dos grupos.")
    else:
        print("No hay diferencias significativas entre los grupos.")
    
    print("\n0) Baseline")
    print("1) IA")
    print("2) Cubicspline\n")
    # Realiza la prueba de Tukey como prueba post hoc
    tukey_results = tukey_hsd(*all_losses)
    print(tukey_results)

def sent_test_result(model, inputs, mask, device):

    src_mask = model.get_src_mask(mask, len(inputs)).to(device)

    pred = model(inputs, inputs, decoder_mask=mask, src_mask=src_mask)

    pred_images = prepare_keypoints_image(pred[0], connections, 0, "Test")
    for _rel_pos in range(1, len(inputs)):
        pred_images = np.concatenate((pred_images, prepare_keypoints_image(pred[_rel_pos], connections, _rel_pos)), axis=1)

    images = wandb.Image(pred_images, caption="Validation")
    wandb.log({"examples of test": images})

def sent_validation_result(inputs, prediction, sota, connections, epoch):

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
    wandb.log({"examples_validation epoch": images}, step=epoch)

def train_epoch(model, first_model, dataloader, criterion, optimizer, device):

    model.train()
    loss_collector_acum = []

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data
        
        x = inputs.squeeze(0).to(device).float()[:-1,:,:]
        x_no_sota = inputs.squeeze(0).to(device).float()[1:,:,:]
        
        y = sota.squeeze(0).to(device).float()
        
        mask = mask.squeeze(0).to(device).float()
        x_mask = mask[:-1].clone().detach()
        y_mask = mask[1:].clone().detach()
        
        x_no_missing_mask = 1 - x_mask
        y_no_missing_mask = 1 - y_mask
        
        src_mask = first_model.get_mask(x_mask, x.shape[0], "all").to(device)
        tgt_mask = first_model.get_mask(y_mask, y.shape[0], "all").to(device)

        src_mask_first = first_model.get_mask(x_mask, x.shape[0], "repeat-inc").to(device)
        tgt_mask_first = first_model.get_mask(y_mask, y.shape[0], "repeat-inc").to(device)

        pred = first_model(x, x_no_sota,
                           src_pad_mask=x_mask.unsqueeze(0),
                           tgt_pad_mask=y_mask.unsqueeze(0),
                           src_mask=src_mask_first,
                           tgt_mask=tgt_mask_first)

        pred = model(x_no_sota, pred,
                     src_pad_mask=torch.ones_like(x_mask.unsqueeze(0)),
                     tgt_pad_mask=torch.ones_like(y_mask.unsqueeze(0)),
                     src_mask=src_mask,
                     tgt_mask=tgt_mask)

        #print(pred.shape, y_mask.unsqueeze(1).unsqueeze(2).shape, y.shape, y_no_missing_mask.unsqueeze(1).unsqueeze(2).shape, "<--")
        #pred = pred * y_mask.unsqueeze(1).unsqueeze(2) + y * y_no_missing_mask.unsqueeze(1).unsqueeze(2)
        loss = criterion(pred, y)

        loss = loss.float() # + endocer_loss.float() * 0.5# + decoder_loss.float() * 0.5
        loss_collector_acum.append(loss.clone().detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_collector_acum

def eval_epoch(model, first_model, dataloader, criterion, epoch, device):

    model.eval()
    #running_loss = 0.0
    
    #data_length = len(dataloader)
    loss_collector_acum = []

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data
        
        x = inputs.squeeze(0).to(device).float()[:-1,:,:]
        x_no_sota = inputs.squeeze(0).to(device).float()[1:,:,:]
        
        y = sota.squeeze(0).to(device).float()

        mask = mask.squeeze(0).to(device).float()
        x_mask = mask[:-1].clone().detach()
        y_mask = mask[1:].clone().detach()
        
        x_no_missing_mask = 1 - x_mask
        y_no_missing_mask = 1 - y_mask

        src_mask = first_model.get_mask(x_mask, x.shape[0], "all").to(device)
        tgt_mask = first_model.get_mask(y_mask, y.shape[0], "all").to(device)

        src_mask_first = first_model.get_mask(x_mask, x.shape[0], "repeat-inc").to(device)
        tgt_mask_first = first_model.get_mask(y_mask, y.shape[0], "repeat-inc").to(device)

        pred = first_model(x, x_no_sota,
                           src_pad_mask=x_mask.unsqueeze(0),
                           tgt_pad_mask=y_mask.unsqueeze(0),
                           src_mask=src_mask_first,
                           tgt_mask=tgt_mask_first)

        pred = model(x_no_sota, pred,
                     src_pad_mask=torch.ones_like(x_mask.unsqueeze(0)),
                     tgt_pad_mask=torch.ones_like(y_mask.unsqueeze(0)),
                     src_mask=src_mask,
                     tgt_mask=tgt_mask)
                        
        pred = pred * y_mask.unsqueeze(1).unsqueeze(2) + y * y_no_missing_mask.unsqueeze(1).unsqueeze(2)
        
        loss = criterion(pred, y)
        loss_collector_acum.append(loss.clone().detach().cpu().numpy())
        
        if epoch == 0:
            baseline_loss = criterion(x_no_sota, y)
            loss_baseline_acum.append(baseline_loss.clone().detach().cpu().numpy())

            cubic = cubic_interpolation(x_no_sota, y_mask)
            cubic_loss = criterion(cubic, y.clone().detach().cpu())
            loss_cubic_acum.append(cubic_loss)

        if i == 1:
 
            valid_results_variables = {
                'inputs': x * x_no_missing_mask.unsqueeze(1).unsqueeze(2),
                'prediction': pred,
                'sota': y,
                'connections': connections,
                'epoch': epoch
            }

    return loss_collector_acum, valid_results_variables

def train(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    g = torch.Generator()
    g.manual_seed(42)

    train_set = dataloader.LSP_Dataset(args.training_set_path,
                                       have_aumentation=True,
                                       is_train=True,
                                       keypoints_model='mediapipe',
                                       is_random_missing=False)

    body_parts_class = train_set.body_parts_class

    val_set = dataloader.LSP_Dataset(args.validation_set_path,
                                        keypoints_model='mediapipe',
                                        is_train=False,
                                        have_aumentation=False,
                                        is_random_missing=False)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1,  generator=g)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1,  generator=g)


    checkpoint = torch.load(f'model_checkpoint/{args.upload_model}')
    
    keysecom_model = model.KeypointCompleter(input_size=checkpoint['input_size'], 
                                             hidden_dim=checkpoint['hidden_dim'], 
                                             num_layers=checkpoint['num_layers'],
                                             num_heads=checkpoint['num_heads'])

    
    keysecom_model_cycle = model.KeypointCompleterCycle(input_size=54*2, 
                                             hidden_dim=args.hidden_dim, 
                                             num_layers=args.num_layers,
                                             num_heads=args.num_heads)
    wandb.watch(keysecom_model_cycle)
    model_path = f'./model_checkpoint/{wandb.run.name}.pth'

    criterion = MSELoss()#EuclideanLoss()#MSELoss()
    criterion_validation = EuclideanLoss()
    optimizer = Adam(keysecom_model_cycle.parameters(), lr=args.lr)
    # optimizer = RMSprop(keysecom_model_cycle.parameters(), lr=args.lr)

    keysecom_model.load_state_dict(checkpoint['model_state_dict'])

    keysecom_model.train(False)
    keysecom_model.to(device)

    keysecom_model_cycle.train(True)
    keysecom_model_cycle.to(device)

    best_model_state_dict = None
    epoch_start = 0
    #losses = []
    patience_loss = 0
    min_loss = float('inf')
    
    # learning rate scheduler | This function doesn't work for continue learning at certain point
    initial_lr = args.lr
    final_lr = args.lr/5
    lr_values = np.linspace(initial_lr, final_lr, num=args.epochs)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # TRAINING AND EVALUATION
    for epoch in range(epoch_start, args.epochs):
        print(f'|=> Epoch: [{epoch}], Lr_rate: {optimizer.param_groups[0]["lr"]}')

        optimizer = lr_lambda(epoch, lr_values, optimizer)
        
        print("patinece:", patience_loss)

        train_loss_acum = train_epoch(keysecom_model_cycle, keysecom_model, train_loader, criterion, optimizer, device)
        val_loss_acum, valid_result_first_epoch = eval_epoch(keysecom_model_cycle, keysecom_model, val_loader, criterion_validation, epoch, device)
        
        train_loss = np.mean(train_loss_acum)
        val_loss = np.mean(val_loss_acum)
        
        print("train loss:", train_loss)
        print("eval loss:", val_loss)

        patience_loss += 1
        
        if val_loss < min_loss:
            min_loss = val_loss
            sent_histogram(loss_baseline_acum, val_loss_acum, loss_cubic_acum, val_set.dataset_name, epoch)  
            sent_validation_result(valid_result_first_epoch['inputs'],
                                   valid_result_first_epoch['prediction'],
                                   valid_result_first_epoch['sota'],
                                   valid_result_first_epoch['connections'],
                                   valid_result_first_epoch['epoch'])
            patience_loss = 0
            #best_model_state_dict = keysecom_model_cycle.state_dict()
            
            torch.save({
                'model_state_dict': keysecom_model_cycle.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_size':54*2, 
                'hidden_dim':args.hidden_dim, 
                'num_layers':args.num_layers,
                'num_heads':args.num_heads,
                'loss': min_loss  
            }, model_path)
           
            artifact = wandb.Artifact(name=f'model_{wandb.run.name}', type='model')
            artifact.add_file(model_path)

            wandb.log_artifact(artifact)
        
        wandb.log({
            'train_loss': train_loss,
            'val_loss':val_loss,
            'epoch': epoch,
            'minimun_loss': min_loss,
        })
        
        if patience_loss >= args.patience:
            print("Max patience set to ", args.patience)

            break
        #keysecom_model_cycle.load_state_dict(best_model_state_dict)
        #optimizer = torch.optim.Adam(keysecom_model_cycle.parameters(), lr=lr_values[epoch])
        #scheduler.step()
        # losses.append(train_loss.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[parseMain.get_default_args()], add_help=False)
    args = parser.parse_args()

    run = wandb.init(project=PROJECT_WANDB,
                     entity=ENTITY,
                     config=args,
                     name=args.experiment_name,
                     #mode="offline",
                     job_type="model-training",
                     tags=TAG,
                     save_code=True)

    run.notes = args.notes 
    config = wandb.config

    wandb.run.log_code(".")
    
    train(args)