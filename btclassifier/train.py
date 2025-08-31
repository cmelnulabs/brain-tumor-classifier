# Training loop and early stopping
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm

def get_class_weights(train_dataset, device):
    train_targets = [label for _, label in train_dataset.samples]
    class_sample_count = Counter(train_targets)
    weights = [1.0 / class_sample_count[i] for i in range(len(train_dataset.classes))]
    return torch.FloatTensor(weights).to(device)

def train_one_epoch(model, loader, criterion, optimizer, device, show_progress: bool = True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    iterator = tqdm(loader, desc="Train", leave=False) if show_progress else loader
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size
        if show_progress:
            iterator.set_postfix(loss=loss.item())
    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
