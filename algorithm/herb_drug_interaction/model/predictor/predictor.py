import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, in_feats, out_feats, config):
        super().__init__()
        self.dropout = nn.Dropout(config['predictor_dropout'])
        self.linear1 = nn.Linear(in_feats, config['predictor_hidden_feats'])
        self.activation = nn.GELU()
        self.batch_normal = nn.BatchNorm1d(config['predictor_hidden_feats'])
        self.linear2 = nn.Linear(config['predictor_hidden_feats'], out_feats)
    
    def forward(self, features):
        emb = self.dropout(features)
        emb = self.batch_normal(self.activation(self.linear1(emb)))
        emb = self.linear2(emb)
        return emb