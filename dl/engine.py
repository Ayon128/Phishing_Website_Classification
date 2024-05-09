import torch
import torch.nn as nn
class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.BCELoss()

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device).float()
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)  # Add sequence dimension
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        correct = 0.0
        total_examples = 0
        with torch.no_grad():
            for data in data_loader:
                inputs = data['x'].to(self.device)
                targets = data['y'].to(self.device).float()
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(1)  # Add sequence dimension
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
                final_loss += loss.item()
                pred = (logits > 0.5).float()
                correct += (pred == targets).float().sum().item()
                total_examples += targets.size(0)
        return final_loss / len(data_loader), correct / total_examples