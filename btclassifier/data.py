# Data loading and transforms
from pathlib import Path
from collections import Counter
from torchvision import transforms, datasets

def get_transforms():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_datasets(train_dir, test_dir, transform):
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
    return train_dataset, test_dataset

def count_images_per_class(dataset):
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    for idx, class_name in enumerate(dataset.classes):
        print(f"{class_name}: {class_counts.get(idx, 0)} images")
