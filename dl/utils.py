from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # reshape the features to have 1 channel and 41 sequence length
        x = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        y = torch.FloatTensor([self.targets[idx]])
        return {
            "x" : x,
            "y" : y
            }
            
            
        
       