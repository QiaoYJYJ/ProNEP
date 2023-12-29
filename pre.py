import sys

import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from dataloader import pretrained_embedding
from models import DrugBANCustom
import torch.nn.functional as F
import argparse
from Bio import SeqIO
import pandas as pd
from torch.utils.data import Dataset, DataLoader



max_length_NLR = 1279
max_length_eff = 1759
class ProteinDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p1 = pretrained_embedding(self.df.iloc[idx, 0], max_length_NLR)
        p2 = pretrained_embedding(self.df.iloc[idx, 1], max_length_eff)
        return p1, p2

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for p1, p2 in dataloader:
            _, _, score, _ = model(p1, p2, mode="eval")
            predictions.append(score)
    return predictions


def load_data(file1, file2):
    p1 = [str(record.seq) for record in SeqIO.parse(file1, "fasta")]
    p2 = [str(record.seq) for record in SeqIO.parse(file2, "fasta")]
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    return df


def main(file1, file2):
    df = load_data(file1, file2)
    df.to_csv('pairs.csv', index=False)

    dataset = ProteinDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    configs = {
        "PROTEIN": {
            "EMBEDDING_DIM": 6165,
            "NUM_FILTERS": [128, 128, 128],
            "KERNEL_SIZE": [3, 5, 7],
            "PADDING": True
        },
        "DECODER": {
            "IN_DIM": 256,
            "HIDDEN_DIM": 512,
            "OUT_DIM": 128,
            "BINARY": 1
        },
        "BCN": {
            "HEADS": 2
        }
    }

    model = DrugBANCustom(**configs)
    state_dict = torch.load('result/best_model_epoch_89.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    predictions = predict(model, dataloader)
    for batch_predictions in predictions:
        for logits in batch_predictions:
            probabilities = torch.sigmoid(logits)
            print(probabilities)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein Dataset')
    parser.add_argument('--file1', type=str, help='Path to p1.fasta')
    parser.add_argument('--file2', type=str, help='Path to p2.fasta')
    args = parser.parse_args()

    main(args.file1, args.file2)