

import argparse
import os
import parseMain

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

from euclidean_loss import EuclideanLoss, EuclideanDistanceLoss
from utils import sent_histogram, sent_validation_result, sent_test_result, get_edges_index


np.random.seed(42)
#pd.np.random.seed(42)
torch.manual_seed(42)


CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "fill_missings_transformer"
ENTITY = "joenatan30" #joenatan30
TAG = ["trained_embedding"]

EMBEDDING_MODEL_NAME = "embedding_256_ethereal-hill-40"
GENERAL_MODEL_NAME = "vivid-hill-800"#"dashing-star-794"#"polished-sky-789"

#os.environ["WANDB_API_KEY"] = "c16c54799944a6127132bcb81b2fb9ebcb4fe5db"

connections = np.moveaxis(np.array(get_edges_index('54')), 0, 1)


loss_baseline_acum = []
loss_cubic_acum = []

def lr_lambda(current_step, lr, optim):
    
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



def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    loss_collector_acum = []

    for i, data in enumerate(dataloader):

        inputs, sota, mask = data
        
        x = inputs.squeeze(0).to(device).float()[:-1,:,:]
        x_no_sota = inputs.clone().squeeze(0).to(device).float()[1:,:,:]
        
        y = sota.squeeze(0).to(device).float()
        
        mask = mask.squeeze(0).to(device).float()
        x_mask = mask[:-1].clone().detach()
        y_mask = mask[1:].clone().detach()
        
        # x_no_missing_mask = 1 - x_mask
        # y_no_missing_mask = 1 - y_mask

        src_mask = model.get_mask(x_mask, x.shape[0], "repeat-inc").to(device)
        tgt_mask = model.get_mask(y_mask, y.shape[0], "repeat-inc").to(device)

        mask_expanded = x_mask.bool()[:, None, None].expand(-1, inputs.size(2), inputs.size(3)).to(device)
        x = torch.where(mask_expanded, torch.zeros_like(x).float().to(device), x)

        pred = model(x, x_no_sota,
                     src_pad_mask=x_mask.unsqueeze(0),
                     tgt_pad_mask=y_mask.unsqueeze(0),
                     src_mask=src_mask,
                     tgt_mask=tgt_mask)

        loss = criterion(pred, y)

        loss = loss.float() # + endocer_loss.float() * 0.5# + decoder_loss.float() * 0.5
        loss_collector_acum.append(loss.clone().detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_collector_acum

def eval_epoch(model, dataloader, criterion, epoch, device):

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
        
        src_mask = model.get_mask(x_mask, x.shape[0], "repeat-inc").to(device)
        tgt_mask = model.get_mask(y_mask, y.shape[0], "repeat-inc").to(device)

        mask_expanded = x_mask.bool()[:, None, None].expand(-1, inputs.size(2), inputs.size(3)).to(device)
        x = torch.where(mask_expanded, torch.zeros_like(x).float().to(device), x)

        pred = model(x, x_no_sota,
                     src_pad_mask=x_mask.unsqueeze(0),
                     tgt_pad_mask=y_mask.unsqueeze(0),
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
    
    keysecom_model = model.KeypointCompleter(input_size=54*2, 
                                             hidden_dim=args.hidden_dim, 
                                             num_layers=args.num_layers,
                                             num_heads=args.num_heads)
    
    optimizer = Adam(keysecom_model.parameters(), lr=args.lr)   
    
    if args.upload_general_model:
        checkpoint = torch.load(f'model_checkpoint/{GENERAL_MODEL_NAME}.pth', map_location=torch.device(device))
        general_model = checkpoint['model_state_dict']

        keysecom_model.load_state_dict(general_model)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    if args.upload_embedding_model:
        checkpoint = torch.load(f'model_checkpoint/{EMBEDDING_MODEL_NAME}.pth', map_location=torch.device(device))
        embedding_model = checkpoint['model_state_dict']

        keysecom_model.input_embedding.weight.data = embedding_model['input_embedding.weight']
        keysecom_model.input_embedding.bias.data = embedding_model['input_embedding.bias']

        keysecom_model.filled_embedding.weight.data = embedding_model['input_embedding.weight']
        keysecom_model.filled_embedding.bias.data = embedding_model['input_embedding.bias']

        keysecom_model.fc_final.weight.data = embedding_model['output_embedding.weight']
        keysecom_model.fc_final.bias.data = embedding_model['output_embedding.bias']

        for param in keysecom_model.input_embedding.parameters():
            param.requires_grad = False
        for param in keysecom_model.filled_embedding.parameters():
            param.requires_grad = False
        for param in keysecom_model.fc_final.parameters():
            param.requires_grad = False  

    wandb.watch(keysecom_model)
    model_path = f'./model_checkpoint/{wandb.run.name}.pth'

    criterion = EuclideanLoss()#EuclideanLoss()#MSELoss()
    criterion_validation = EuclideanDistanceLoss()

    # optimizer = RMSprop(keysecom_model.parameters(), lr=args.lr)
    keysecom_model.train(True)
    keysecom_model.to(device)


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

        train_loss_acum = train_epoch(keysecom_model, train_loader, criterion, optimizer, device)
        val_loss_acum, valid_result_first_epoch = eval_epoch(keysecom_model, val_loader, criterion_validation, epoch, device)
        
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
            #best_model_state_dict = keysecom_model.state_dict()
            
            torch.save({
                'model_state_dict': keysecom_model.state_dict(),
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
        #keysecom_model.load_state_dict(best_model_state_dict)
        #optimizer = torch.optim.Adam(keysecom_model.parameters(), lr=lr_values[epoch])
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
    
    #files_list = ["3_train.py", "dataloader.py", "model.py", "parseMain.py"]
    #for file in files_list:
    #    wandb.run.log_code(f"./{file}")
    wandb.run.log_code(".")
    
    
    train(args)