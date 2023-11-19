
from torch.utils.data import DataLoader
from model import Classifier, lstmClassifier, transformerencoderClassifer
from dataset import LibriDataset
from util import preprocess_data, same_seeds
from dataloader import get_train_val_dataloader

import gc
import json
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    # import config
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_epoch = self.config['num_epoch']




    def train(self):

        # create model, define a loss function, and optimizer
        # model = lstmClassifier(input_dim=self.config['input_dim'] * self.config['concat_nframes'], 
        #                    hidden_layers=self.config['hidden_layers'], 
        #                    hidden_dim=self.config['hidden_dim']).to(self.device)
        model = transformerencoderClassifer(input_dim=self.config['input_dim'], 
                           hidden_layers=self.config['hidden_layers'], 
                           hidden_dim=self.config['hidden_dim'],
                           nhead=self.config['nhead'],
                           concat_nframes=self.config['concat_nframes']).to(self.device)
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        # training
        train_loader, val_loader = get_train_val_dataloader(batch_size=self.config['batch_size'], 
                                                            concat_nframes=self.config['concat_nframes'], 
                                                            train_ratio=self.config['train_ratio'], 
                                                            seed=self.config['seed'])


        best_acc = 0.0
        for epoch in range(self.config['num_epoch']):
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            
            # training
            model.train() # set the model to training mode
            for i, batch in enumerate(tqdm(train_loader)):
                features, labels = batch
                features = features.reshape(features.shape[0],
                                            self.config['concat_nframes'],
                                            self.config['input_dim'])
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad() 
                
                outputs = model(features) 
                
                loss = criterion(outputs, labels)
                loss.backward() 
                optimizer.step() 
                
                _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                train_acc += (train_pred.detach() == labels.detach()).sum().item()
                train_loss += loss.item()
            
            # validation
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.reshape(features.shape[0],
                                            self.config['concat_nframes'],
                                            self.config['input_dim'])
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(features)
                    
                    loss = criterion(outputs, labels) 
                    
                    _, val_pred = torch.max(outputs, 1) 
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += loss.item()

            print(f'[{epoch+1:03d}/{self.num_epoch:03d}] Train Acc: {train_acc/len(train_loader.dataset):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_loader.dataset):3.5f} loss: {val_loss/len(val_loader):3.5f}')

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), self.config['model_path'])
                print(f'saving model with acc {best_acc/len(val_loader.dataset):.5f}')

        

        del train_loader, val_loader
        gc.collect()

if __name__ == "__main__":
    # import config from config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.train()