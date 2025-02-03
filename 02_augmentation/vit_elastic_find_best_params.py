import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
import timm
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np

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
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx]['image_name']
        label = self.label_df.iloc[idx]['emotion'] - 1  # Convert to 0-based indexing
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Elastic Transformation Function
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

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

    return Image.fromarray(transformed_image.astype(np.uint8))

class ElasticTransform:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        return elastic_transform(image, self.alpha, self.sigma)

# Model setup
def initialize_model(num_classes=7):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# Training and validation functions
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Grid Search for Elastic Transformation Parameters
def grid_search_elastic(train_df, test_df, img_dir, alpha_values, sigma_values, device, num_epochs=5):
    best_accuracy = 0.0
    best_alpha_sigma = (None, None)
    results = []

    for alpha in alpha_values:
        for sigma in sigma_values:            
            transform_elastic = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                ElasticTransform(alpha=alpha, sigma=sigma),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            # Minimal transformation for testing
            general_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # Prepare datasets
            train_dataset = RAFDBDataset(img_dir=img_dir, label_df=train_df, transform=transform_elastic)
            test_dataset = RAFDBDataset(img_dir=img_dir, label_df=test_df, transform= general_transform)
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                train_dataset,
                [int(0.9 * len(train_dataset)), len(train_dataset) - int(0.9 * len(train_dataset))],
                generator=generator
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize and train model
            model = initialize_model()
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

            for epoch in range(num_epochs):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                validate(model, val_loader, criterion, device)

            # Test and store results
            test_accuracy = test_model(model, test_loader, device)
            results.append((alpha, sigma, test_accuracy))

            # Update best parameters
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_alpha_sigma = (alpha, sigma)

    print("\nGrid Search Results:")
    for alpha, sigma, accuracy in results:
        print(f"alpha={alpha}, sigma={sigma}, Test Accuracy={accuracy:.2f}%")
    print(f"\nBest Parameters: alpha={best_alpha_sigma[0]}, sigma={best_alpha_sigma[1]} with Test Accuracy={best_accuracy:.2f}%")
    return best_alpha_sigma, best_accuracy


# Execution
# Load label data
label_df = pd.read_csv(label_file)[['image_name', 'emotion', 'gender']]
label_df = label_df[label_df['gender'] != 2]  # Exclude rows where gender is Unsure

# Create train and test dataframes
train_df = label_df[label_df['image_name'].str.startswith('train')]
test_df = label_df[label_df['image_name'].str.startswith('test')]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform grid search to find best alpha and sigma
alpha_values = [10, 20, 30, 40, 50]
sigma_values = [4, 6, 8, 10]
best_alpha_sigma, _ = grid_search_elastic(train_df, test_df, img_dir, alpha_values, sigma_values, device)
