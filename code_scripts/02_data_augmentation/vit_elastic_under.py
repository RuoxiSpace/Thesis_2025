

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
import timm
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Directories and files
img_dir = './rafdb/Image/aligned/'
label_file = './rafdb/raf_labels.csv'

# Dataset class with conditional transformation
class RAFDBDataset(Dataset):
    def __init__(self, img_dir, label_df, transform=None, advanced_transform=None, alpha=50, sigma=6):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform
        self.advanced_transform = advanced_transform
        self.alpha = alpha  # Set default alpha
        self.sigma = sigma  # Set default sigma

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        img_name = row['image_name']
        label = row['emotion'] - 1  # Convert to 0-based indexing
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Check if the sample is part of the minority group
        if self._is_minority_group(row) and self.advanced_transform:
            image = self.advanced_transform(image, self.alpha, self.sigma)
        elif self.transform:
            image = self.transform(image)

        return image, label, img_name

    def _is_minority_group(self, row):
        return (
            (row['gender'] == 0 and row['emotion'] in [1, 2, 5]) or
            (row['race'] == 1 and row['emotion'] in [1, 2, 5]) or
            (row['age'] in [0, 4] and row['emotion'] in [1, 2, 5])
        )

# Elastic Transformation Function
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)  # Set a new seed for randomness
    image_np = np.array(image)
    if len(image_np.shape) == 3:  # RGB image
        channels = []
        for c in range(image_np.shape[2]):
            channel = image_np[..., c]
            dx = gaussian_filter((random_state.rand(*channel.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*channel.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            x, y = np.meshgrid(np.arange(channel.shape[0]), np.arange(channel.shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            transformed_channel = map_coordinates(channel, indices, order=1, mode='reflect').reshape(channel.shape)
            channels.append(transformed_channel)
        transformed_image = np.stack(channels, axis=-1)
    else:  # Grayscale
        dx = gaussian_filter((random_state.rand(*image_np.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*image_np.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(image_np.shape[0]), np.arange(image_np.shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        transformed_image = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(image_np.shape)

    # Convert the numpy array back to a PIL Image
    transformed_image = Image.fromarray(transformed_image.astype(np.uint8))

    # Apply general transformations (resize, convert to tensor, normalize)
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transformed_image = transform_pipeline(transformed_image)

    return transformed_image

# General transformation
general_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load labels with additional metadata and exclude 'gender = 2'
label_df = pd.read_csv(label_file)[['image_name', 'emotion', 'gender', 'race', 'age']]
label_df = label_df[label_df['gender'] != 2]  # Exclude rows where gender is 2

# Precompute metadata for fast lookup
metadata_dict = label_df.set_index('image_name').to_dict('index')

# Model setup
def initialize_model(num_classes=7):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
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
def collect_predictions_with_metadata(dataset, model, device):
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
        img_dir, 
        label_df[label_df['image_name'].str.startswith('train')],
        transform=general_transform, 
        advanced_transform=lambda image, alpha=50, sigma=6: elastic_transform(image, alpha, sigma)
    )
    test_dataset = RAFDBDataset(
        img_dir, 
        label_df[label_df['image_name'].str.startswith('test')], 
        transform=general_transform
    )
    
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
        gt, pred, imgs = collect_predictions_with_metadata(dataset, model, device)

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
final_results_df.to_csv('./vit_elastic_under.csv', index=False)

print("Predictions with metadata for all datasets saved to './vit_elastic_under.csv'")
