import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
import timm
from torch.optim.lr_scheduler import StepLR

# Directories and files
img_dir = './rafdb/Image/aligned/'
label_file = './rafdb/raf_labels.csv'

# Dataset class
class RAFDBDataset(Dataset):
    def __init__(self, img_dir, label_df, transform=None):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        img_name = row['image_name']
        label = torch.tensor(row['emotion'] - 1, dtype=torch.long)  # Convert label to 0-based indexing as Tensor
        img_path = os.path.join(self.img_dir, img_name)

        # Load and preprocess the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Ensure the image is a 3D tensor [Channels, Height, Width]
        if len(image.shape) != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}, expected [C, H, W]")

        # Determine if the sample is from a minority group
        is_minority = self._is_minority_group(row)

        return image, label, img_name, is_minority

    def _is_minority_group(self, row):
        return (
            (row['gender'] == 0 and row['emotion'] in [1, 2, 5]) or  # Male + Fear, Disgust, Anger
            (row['race'] == 1 and row['emotion'] in [1, 2, 5]) or    # African-American + Fear, Disgust, Anger
            (row['age'] in [0, 4] and row['emotion'] in [1, 2, 5])   # Age 0-3 or 70+ + Fear, Disgust, Anger
        )


# Custom Vision Transformer with dynamic attention adjustment
class CustomViT(nn.Module):
    def __init__(self, num_classes=7, alpha=0.2):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.model.blocks = self.model.blocks[:6]  # Keep only the first 6 layers
        self.alpha = alpha  # Weight adjustment factor for minority groups

    def forward(self, x, is_minority=None):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)

        for blk in self.model.blocks:
            x = blk(x)

            # Apply dynamic weight adjustment for minority groups
            if is_minority is not None and is_minority.sum() > 0:
                w_g = 1 + self.alpha
                x[is_minority, 0, :] *= w_g  # Increase CLS token weight for minority samples

        x = self.model.norm(x)
        return self.model.head(x[:, 0])  # Use the CLS token for classification


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


# Training function
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels, _, is_minority in train_loader:
        images, labels = images.to(device), labels.to(device)
        is_minority = is_minority.clone().detach().to(device) if isinstance(is_minority, torch.Tensor) else torch.tensor(is_minority, dtype=torch.bool, device=device)
        optimizer.zero_grad()
        outputs = model(images, is_minority)
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
        for images, labels, img_names, is_minority in data_loader:
            images, labels = images.to(device), labels.to(device)
            is_minority = is_minority.clone().detach().to(device) if isinstance(is_minority, torch.Tensor) else torch.tensor(is_minority, dtype=torch.bool, device=device)
            outputs = model(images, is_minority)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_ground_truths.extend(labels.cpu().numpy())
            image_names.extend(img_names)

    return all_ground_truths, all_predictions, image_names


# Main training and evaluation loop
num_runs = 10
final_results = {
    'image_name': [],
    'ground_truth': [],
    'gender': [],
    'race': [],
    'age': [],
    'dataset_split': [],  # To record if the sample belongs to train/val/test
}

for run in range(num_runs):
    print(f"Starting run {run + 1}/{num_runs}")

    # Initialize datasets and dataloaders
    all_dataset = RAFDBDataset(img_dir, label_df, general_transform)
    train_dataset = RAFDBDataset(
        img_dir, label_df[label_df['image_name'].str.startswith('train')], advanced_transform)
    test_dataset = RAFDBDataset(
        img_dir, label_df[label_df['image_name'].str.startswith('test')], general_transform)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = CustomViT(num_classes=7, alpha=0.2)
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
        gt, pred, imgs = collect_predictions(model, DataLoader(dataset, batch_size=32), device)

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
final_results_df.to_csv('./a1.csv', index=False)

print("Predictions with metadata saved to './a1.csv'")
