# Evaluation logic
import torch
from tqdm import tqdm

def evaluate_model(model, loader, criterion, device, show_progress: bool = True):
    model.eval()
    correct = 0
    total = 0
    losses = []
    iterator = tqdm(loader, desc="Eval", leave=False) if show_progress else loader
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = images.size(0)
            losses.append(loss.item() * batch_size)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size
            if show_progress:
                iterator.set_postfix(loss=loss.item())
    avg_loss = sum(losses) / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
