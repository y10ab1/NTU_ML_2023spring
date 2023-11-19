import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class lstmClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, bidirectional=True, dropout=0.2):
        super(lstmClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True, bidirectional=True, dropout=dropout, bias=True)
        if bidirectional:
            hidden_dim *= 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        
        return x


class transformerencoderClassifer(nn.Module):  
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, bidirectional=True, nhead=8, concat_nframes=41):
        super(transformerencoderClassifer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=hidden_layers)

        self.fc = nn.Sequential(
            nn.Linear(concat_nframes*input_dim, output_dim),
            # nn.ReLU(),
        )

    def forward(self, x):
        
        x = self.transformer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x