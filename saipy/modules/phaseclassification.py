import random
import os,sys
sys.path.insert(0, '..')
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
from .pytorchtools import EarlyStopping

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.y)
    
def train(args,device,Train_Loader,Valid_Loader,criterion,optimizer,scheduler):
    
    if not os.path.exists(args.model_save_path):
        # Create a new directory because it does not exist
        os.makedirs(args.model_save_path)
        print("The directory is created!")
    model = torch.jit.load('../saipy/saved_models/saved_model.pt', map_location = 'cpu')
    model = model.to(device)
    
    train_losses = []
    avg_train_losses = []

    valid_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience= args.patience, verbose=args.verbose, path =    args.model_save_path)
    
    for epoch in tqdm(range(1, args.epochs+1)):

        model.train()
        
        for step, (data, target) in enumerate(Train_Loader):
            
            data = data.to(device)
            target = target.to(device)
            
            if data.shape[1]!=3:
                data = data.permute(0,2,1)
                
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, target)
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
        # validate the model #
        model.eval() 
        for data, target in Valid_Loader:
            data = data.to(device)
            if data.shape[1]!=3:
                data = data.permute(0,2,1)
            target =  target.to(device)
            
            output = model(data.float())
            loss = criterion(output, target)
            
            valid_losses.append(loss.item())
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(args.epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')
        
        print(print_msg)
        
        if scheduler is not None:  
            scheduler.step()
            
        train_losses = []
        valid_losses = []
        
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'checkpoint.pt')))
    return  model, avg_train_losses, avg_valid_losses

def test(args, device, model, Test_Loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(args.num_classes))
    class_total = list(0. for i in range(args.num_classes))
    
    model = model.to(device)
    model.eval()
    out = []
    pred_prob = []
    for data, target in Test_Loader:
        data = data.to(device)
        if data.shape[1]!=3:
            data = data.permute(0,2,1)
        target =  target.to(device)
        
        if len(target.data) != args.batch_size:
            break
            
        output = model(data.float())
        
        loss = criterion(output, target)
        
        test_loss += loss.item()*data.size(0)
        pred_prob.append(torch.softmax(output, dim=1).detach().cpu().numpy())
        
        _, pred = torch.max(output, 1)
        out.append(pred.cpu())

        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(args.batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(Test_Loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(args.num_classes):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\n Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    out = torch.cat(out, dim=0)
    out = out.detach().cpu().numpy()
    return out, pred_prob


