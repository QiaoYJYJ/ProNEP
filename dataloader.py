import torch.utils.data as data
import torch
from pro_embedding import embed_sequence
from prose.models.multitask import ProSEMT

max_length = 1795

model = ProSEMT.load_pretrained()
model.eval()
def pretrained_embedding(v, max_length):
    z = embed_sequence(model, v)
    padding = z.size()[0]
    if padding < max_length:
        padding_zeros = torch.zeros((max_length - z.size()[0]), 6165)
        weight = torch.vstack((z, padding_zeros))
    else:
        weight = z[:max_length, :]
    return weight


class CNEDATA(data.Dataset):
    def __init__(self, list_IDs, df, max_length_NLR, max_length_eff):
        self.list_IDs = list_IDs
        self.df = df
        self.max_length_NLR = max_length_NLR
        self.max_length_eff = max_length_eff

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['Protein1']
        v_d = pretrained_embedding(v_d, self.max_length_eff)
        v_p = self.df.iloc[index]['Protein2']
        v_p = pretrained_embedding(v_p, self.max_length_NLR)
        y = self.df.iloc[index]["label"]
        y = torch.Tensor([y])
        return v_d, v_p, y


