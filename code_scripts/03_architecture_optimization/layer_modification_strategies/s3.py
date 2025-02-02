import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
import timm
from torch.optim.lr_scheduler import StepLR
from timm.models.vision_transformer import PatchEmbed, Block

# Directories and files
img_dir = '../rafdb/Image/aligned/'
label_file = '../rafdb/raf_labels.csv'

# Dataset class
class RAFDBDataset(Dataset):
    def __init__(self, img_dir, label_df, transform=None):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx]['image_name']
        label = self.label_df.iloc[idx]['emotion'] - 1  # Convert to 0-based indexing
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# Data preprocessing
advanced_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

general_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labels with additional metadata and exclude 'gender = 2'
label_df = pd.read_csv(label_file)[['image_name', 'emotion', 'gender', 'race', 'age']]
label_df = label_df[label_df['gender'] != 2]  # Exclude rows where gender is 2

# Precompute metadata for fast lookup
metadata_dict = label_df.set_index('image_name').to_dict('index')

# Model setup
def initialize_model(num_classes=7):
    # Load the base ViT model
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)

    # Extract the layers
    total_layers = len(model.blocks)  # Total number of layers in the Transformer Encoder
    num_front = 2                     # Number of layers to retain from the front
    num_middle = 2                    # Number of layers to retain from the middle
    num_end = 2                       # Number of layers to retain from the end

    # Define the indices for the middle layers
    middle_start = (total_layers - num_front - num_end) // 2
    middle_end = middle_start + num_middle

    # Combine the selected layers
    selected_blocks = (
        model.blocks[:num_front] +        # First 2 layers
        model.blocks[middle_start:middle_end] +  # Middle 2 layers
        model.blocks[-num_end:]          # Last 2 layers
    )
    model.blocks = nn.Sequential(*selected_blocks)

    return model


# Training function
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Prediction collection function
def collect_predictions(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_ground_truths = []
    image_names = []

    with torch.no_grad():
        for images, labels, img_names in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_ground_truths.extend(labels.cpu().numpy())
            image_names.extend(img_names)

    return all_ground_truths, all_predictions, image_names

# Wrapper function to collect predictions with metadata
def collect_predictions_with_metadata(dataset, model, device, transform):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return collect_predictions(model, loader, device)


num_runs = 10
final_results = {
    'image_name': [],
    'ground_truth': [],
    'gender': [],
    'race': [],
    'age': [],
    'dataset_split': [],  # To record if the sample belongs to train/val/test
}

# Main loop for experiments
for run in range(num_runs):
    print(f"Starting run {run + 1}/{num_runs}")

    # Initialize datasets and dataloaders
    all_dataset = RAFDBDataset(img_dir, label_df, general_transform)  # Full dataset for metadata

    train_dataset = RAFDBDataset(
        img_dir, label_df[label_df['image_name'].str.startswith('train')], advanced_transform)
    test_dataset = RAFDBDataset(
        img_dir, label_df[label_df['image_name'].str.startswith('test')], general_transform)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Ensure consistent train/val split
    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = initialize_model(num_classes=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

    # Train model
    for epoch in range(20):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

    # Collect predictions for train/val/test
    for dataset_split, dataset in zip(
        ['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]
    ):
        gt, pred, imgs = collect_predictions_with_metadata(dataset, model, device, general_transform)

        # On the first run, collect metadata
        if run == 0:
            for img_name, ground_truth in zip(imgs, gt):
                metadata = metadata_dict[img_name]
                final_results['image_name'].append(img_name)
                final_results['ground_truth'].append(ground_truth)
                final_results['gender'].append(metadata['gender'])
                final_results['race'].append(metadata['race'])
                final_results['age'].append(metadata['age'])
                final_results['dataset_split'].append(dataset_split)

        # Add predictions for the current run
        column_name = f'predicted_label_{run + 1}'
        if column_name not in final_results:
            final_results[column_name] = []
        final_results[column_name].extend(pred)

# Check for consistency in results
expected_length = len(final_results['image_name'])
for key, value in final_results.items():
    if len(value) != expected_length:
        raise ValueError(f"Column {key} has inconsistent length: {len(value)}. Expected: {expected_length}.")

# Convert to DataFrame and save
final_results_df = pd.DataFrame(final_results)
final_results_df.to_csv('./s3.csv', index=False)

print("Predictions with metadata saved to './s3.csv'")
