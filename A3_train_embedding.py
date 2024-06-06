import argparse
import os
import parseMain
import random

#from spoter.gaussian_noise import GaussianNoise


import numpy as np # type: ignore
import pandas as pd # type: ignore
import wandb
import model
import dataloader
import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torch.nn import MSELoss # type: ignore
from torch.optim import Adam # type: ignore


from euclidean_loss import EuclideanLoss
from utils import sent_histogram, sent_validation_result, sent_test_result, get_edges_index


np.random.seed(42)
#pd.np.random.seed(42)
torch.manual_seed(42)


CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "fill_missings_transformer_cycle"
ENTITY = "joenatan30" #joenatan30
TAG = ["Embedding"]
EMBEDDING_MODEL_NAME = "embedding_256_sparkling-glade-35"

connections = np.moveaxis(np.array(get_edges_index('54')), 0, 1)

loss_baseline_acum = []
loss_cubic_acum = []

def lr_lambda(lr, optim):

    for param_group in optim.param_groups:
        param_group['lr'] = lr

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



def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    loss_collector_acum = []

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data

        x = inputs.squeeze(0).to(device).float()[:-1,:,:]
        x_mask = mask[:,:-1].clone().detach().squeeze(0)

        if random.random() >= 0.05:
            mask_expanded = x_mask.bool()[:, None, None].expand(-1, inputs.size(2), inputs.size(3)).to(device)
            zeros_matrix = torch.zeros_like(x).float().to(device)
            x = torch.where(mask_expanded, zeros_matrix, x)

        y = x

        pred = model(x)

        loss = criterion(pred, y)

        loss = loss.float()
        loss_collector_acum.append(loss.clone().detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_collector_acum

def eval_epoch(model, dataloader, criterion, epoch, device):

    model.eval()

    loss_collector_acum = []

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data
        
        x = inputs.squeeze(0).to(device).float()[:-1,:,:]
        x_mask = mask[:,:-1].clone().detach().squeeze(0)

        mask_expanded = x_mask.bool()[:, None, None].expand(-1, inputs.size(2), inputs.size(3)).to(device)
        zeros_matrix = torch.zeros_like(x).float().to(device)
        x = torch.where(mask_expanded, zeros_matrix, x)

        y = x

        pred = model(x)
        
        loss = criterion(pred, y)
        loss_collector_acum.append(loss.clone().detach().cpu().numpy())
        
        if epoch == 0:
            baseline_loss = criterion(x, y)
            loss_baseline_acum.append(baseline_loss.clone().detach().cpu().numpy())

            cubic = cubic_interpolation(x, x_mask)
            cubic_loss = criterion(cubic, y.clone().detach().cpu())
            loss_cubic_acum.append(cubic_loss)

        # Print in WandB
        if i == 1:
            valid_results_variables = {
                'inputs': x ,
                'prediction': pred,
                'sota': y,
                'connections': connections,
                'epoch': epoch
            }
            #sent_test_result(model, inputs, mask, device)

    return loss_collector_acum, valid_results_variables

def train(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    g = torch.Generator()
    g.manual_seed(42)

    #transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = dataloader.LSP_Dataset(args.training_set_path,
                                       have_aumentation=True,
                                       is_train=True,
                                       keypoints_model='mediapipe',
                                       is_random_missing=False)

    val_set = dataloader.LSP_Dataset(args.validation_set_path,
                                        keypoints_model='mediapipe',
                                        is_train=False,
                                        have_aumentation=False,
                                        is_random_missing=False)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=1,  generator=g)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1,  generator=g)
    
    embedding_model = model.Embedding(input_size=54*2, 
                                          hidden_dim=args.hidden_dim)

    criterion = EuclideanLoss() #EuclideanLoss() #MSELoss()
    criterion_validation = EuclideanLoss()
    optimizer = Adam(embedding_model.parameters(), lr=args.lr)
    # optimizer = RMSprop(embedding_model.parameters(), lr=args.lr)

    if args.upload_model:
        checkpoint = torch.load(f'model_checkpoint/{EMBEDDING_MODEL_NAME}.pth', map_location=torch.device('cpu'))
        embedding_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if torch.cuda.is_available():
            embedding_model = embedding_model.to(device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    wandb.watch(embedding_model)
    model_path = f'./model_checkpoint/embedding_{args.hidden_dim}_{wandb.run.name}.pth'

    embedding_model.train(True)
    embedding_model.to(device)


    epoch_start = 0
    #losses = []
    patience_loss = 0
    min_loss = float('inf')
    
    # learning rate scheduler | This function doesn't work for continue learning at certain point
    initial_lr = args.lr

    optimizer = lr_lambda(initial_lr, optimizer)

    tmp_max_patience = 20
    rounds = 1

    # TRAINING AND EVALUATION
    for epoch in range(epoch_start, args.epochs):
        print(f'|=> Epoch: [{epoch}], Lr_rate: {optimizer.param_groups[0]["lr"]}')
        
        print("patinece:", patience_loss)

        train_loss_acum = train_epoch(embedding_model, train_loader, criterion, optimizer, device)
        val_loss_acum, valid_result_first_epoch = eval_epoch(embedding_model, val_loader, criterion_validation, epoch, device)
        
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
            #best_model_state_dict = embedding_model.state_dict()
            
            torch.save({
                'model_state_dict': embedding_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_size':54*2, 
                'hidden_dim':args.hidden_dim, 
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

        if epoch == 80:
            initial_lr = initial_lr/10
            optimizer = lr_lambda(initial_lr, optimizer)
        
        if patience_loss >= tmp_max_patience:
            if rounds <= 0:  # equal to 0
                break
            tmp_max_patience = args.patience

            rounds = rounds - 1

            print("Max patience set to ", args.patience)

            

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