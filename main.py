# Entrypoint for training and evaluation
import torch
from tqdm import tqdm
from pathlib import Path
import kagglehub
from btclassifier.config import *
from btclassifier.data import get_transforms, load_datasets, count_images_per_class
from btclassifier.model import build_model
from btclassifier.train import get_class_weights, train_one_epoch
from btclassifier.evaluate import evaluate_model


def main():
    # Step 1: Download dataset from Kaggle (restored inline)
    base_dir = kagglehub.dataset_download(DATASET_KAGGLE)
    base_path = Path(base_dir)
    train_dir = base_path / "Training"
    test_dir = base_path / "Testing"

    transform = get_transforms()
    train_dataset, test_dataset = load_datasets(train_dir, test_dir, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Detected classes:", train_dataset.classes)
    print("\nTraining set image counts per class:")
    count_images_per_class(train_dataset)
    print("\nTest set image counts per class:")
    count_images_per_class(test_dataset)
    print("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(train_dataset.classes)).to(device)
    print("\nModel architecture:\n", model, "\n")

    class_weights = get_class_weights(train_dataset, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_test_accuracy = 0.0

    epoch_iter = tqdm(range(NUM_EPOCHS), desc="Epochs")
    for epoch in epoch_iter:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, show_progress=True)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, show_progress=True)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        scheduler.step(test_loss)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model updated and saved with test accuracy: {best_test_accuracy:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs).")
                break

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
