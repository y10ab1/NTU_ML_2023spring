from torch.utils.data import DataLoader
from util import preprocess_data, same_seeds
from dataset import LibriDataset
import torch
import gc

def get_train_val_dataloader(batch_size=512, concat_nframes=3, train_ratio=0.8, seed=1213):

    same_seeds(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # preprocess data
    train_X, train_y = preprocess_data(split='train', 
                                        feat_dir='libriphone/feat', 
                                        phone_path='libriphone', 
                                        concat_nframes=concat_nframes, 
                                        train_ratio=train_ratio)
    val_X, val_y = preprocess_data(split='val', 
                                   feat_dir='libriphone/feat', 
                                   phone_path='libriphone', 
                                   concat_nframes=concat_nframes, 
                                   train_ratio=train_ratio)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect()

    # get dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader