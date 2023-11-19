from tqdm import tqdm
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from model import Classifier, lstmClassifier, transformerencoderClassifer
from dataset import LibriDataset
from util import preprocess_data, same_seeds

class Tester:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test(self):
        # load data
        test_X = preprocess_data(split='test', feat_dir='libriphone/feat', phone_path='libriphone', concat_nframes=self.config['concat_nframes'])
        test_set = LibriDataset(test_X, None)
        test_loader = DataLoader(test_set, batch_size=self.config['batch_size'] , shuffle=False)

        # load model
        model = transformerencoderClassifer(input_dim=self.config['input_dim'], 
                           hidden_layers=self.config['hidden_layers'], 
                           hidden_dim=self.config['hidden_dim'],
                           nhead=self.config['nhead'],
                           concat_nframes=self.config['concat_nframes']).to(self.device)
        model.load_state_dict(torch.load(self.config['model_path']))

        pred = np.array([], dtype=np.int32)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                features = batch
                features = features.to(self.device)
                features = features.reshape(features.shape[0],
                                            self.config['concat_nframes'],
                                            self.config['input_dim'])

                outputs = model(features)

                _, predicted = torch.max(outputs.data, 1)
                pred = np.concatenate((pred, predicted.cpu().numpy()))

        return pred


def main():
    # load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    # set seed
    same_seeds(config['seed'])

    # test
    tester = Tester(config)
    pred = tester.test()

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))

if __name__ == "__main__":
    main()
