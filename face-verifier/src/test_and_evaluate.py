# PyTorch library (for model creation, training and inference)
import torch
# Timm library (for loading architectures pertaining to image classification)
import timm
# JSON support library
import json
# OS library
import os
# Random library
import random
# Argument parser library
import argparse

# Provides a base Module class to build the neural net on
import torch.nn as nn
# Provides various functions for convolution, pooling, attention and non-linear activation operations
import torch.nn.functional as F
# Matplotlib library (for visualization/plotting purposes)
import matplotlib.pyplot as plt
# NumPy library
import numpy as np
# For TTA (Test-Time Augmentation capabilities)
import ttach as tta

# For loading the dataset
from torch.utils.data import DataLoader, Dataset
# For automatic mixed precision (AMP) training/inference on supported hardware (e.g., NVIDIA GPUs)
from torch.amp import autocast
# For applying transforms on the input image
from torchvision import transforms
# For loading datasets organized by class folders (used by ImageFolder)
from torchvision.datasets import ImageFolder
# tqdm (provides an ergonomic progress bar during training, validation and inference)
from tqdm import tqdm
# scikit-learn metrics (for evaluation purposes)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, classification_report
)
# A dictionary that assigns a default value for non-existent keys
from collections import defaultdict
# For generating combinations of samples from lists
from itertools import combinations

# Path to the pretrained Chinmukh.ai Face Verifier model
model_dir = './pretrained'

# Test set directory
test_dir = './Comys_Hackathon5/Task_B/test'

# Initial seed for the random number generator (RNG)
seed = 42

# Sets the RNG seed for deterministic results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Used to load the dataset
class FacecomVerificationDataset(Dataset):
    # Wraps `ImageFolder` to provide optional transform support and class access
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir)
        self.transform = transform
    # Gets the number of items in the dataset
    def __len__(self):
        return len(self.data)
    # Gets a specific item in the dataset at a specified index
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return idx, image, label
    # Returns a list of the classes in the dataset
    @property
    def classes(self):
        return self.data.classes

# Model that takes an input image and embeds it into a vector of a given dimension
class DistortionFaceNet(nn.Module):
    # Initializes the ResNet-50 backbone for face embedding
    def __init__(self, embedding_dimensions=512):
        # Instantiate parameters from the base class with default values
        super().__init__()
        # Define the base model to train on
        self.base_model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        # Define the embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(self.base_model.num_features, embedding_dimensions),
            nn.Dropout(0.2),
            nn.LayerNorm(embedding_dimensions)
        )
        # Weight initialization of the linear layer using Xavier uniform
        nn.init.xavier_uniform_(self.embedding[0].weight)
        # Initializing bias of the linear layer to zero
        nn.init.zeros_(self.embedding[0].bias)
    # Forward pass
    def forward(self, x):
        # Run the input through the base model and calculate the embedding vector
        return F.normalize(self.embedding(self.base_model(x)))

# Converts embeddings
class WrappedDistortionFaceNet(nn.Module):
    # Initialization
    def __init__(self, model):
        super().__init__()
        self.model = model
    # Forward pass (normalize embeddings after merge)
    def forward(self, x):
        embeddings = self.model(x)
        return F.normalize(embeddings, dim=1)

if __name__ == '__main__':
    # Setting the RNG with a fixed, deterministic seed for reproducibility
    set_seed(seed)

    # Fetch the model's training data
    try:
        with open(os.path.join(model_dir, 'face_verifier_training_data.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Missing \'face_verifier_training_data.json\' in model\'s root directory.')

    # Fetch the image size from the model's training data
    image_size = data['image_size']
    # Fetch the batch size from the model's training data
    batch_size = data['batch_size']
    # Fetch the embedding dimensions from the model's training data
    embedding_dimensions = data['embedding_dimensions']
    # Fetch the best cosine threshold from the model's training data
    cosine_threshold = data['cosine_threshold']

    # Initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-tta', action='store_true', help='Use TTA (Test-Time Augmentation)')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU for evaluation/testing')
    args = parser.parse_args()

    # Device that uses GPU power for inference (if available/applicable)
    device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f'Running on GPU: {torch.cuda.get_device_name(device)}')
    else:
        print('Running on CPU')
    print(f'TTA: {'enabled' if args.use_tta else 'disabled'}')

    # Instantiate the model
    model = DistortionFaceNet(embedding_dimensions)
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'face_verifier_weights.pth'), map_location=device), strict=True)
    # Send the model to the GPU (if available) for faster calculations
    model.to(device)

    # Preprocessing (resize + crop + normalize) for the test set
    test_transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])       # Normalize the data
    ])

    # Import the test data from the dataset and apply the required transformations
    test_dataset = FacecomVerificationDataset(test_dir, test_transform)

    # Data loader for the test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == 'cuda',
        persistent_workers=True
    )

    # A dictionary that maps an identity (label) to indices of each of their images
    identity_to_idxs = defaultdict(list)
    for idx, _, label in test_dataset:
        identity_to_idxs[label].append(idx)
    all_labels = list(identity_to_idxs.keys())

    # Generate positive pairs from the dataset
    positive_pairs = []
    for label, idxs in identity_to_idxs.items():
        if len(idxs) >= 2:
            positive_pairs.extend(combinations(idxs, 2))
    # Generate negative pairs from the dataset
    negative_pairs = []
    for _ in range(len(positive_pairs)):
        label1, label2 = random.sample(all_labels, 2)
        idx1 = random.choice(identity_to_idxs[label1])
        idx2 = random.choice(identity_to_idxs[label2])
        negative_pairs.append((idx1, idx2))

    # TTA preprocessing for the test set
    tta_transforms = tta.Compose([
        tta.HorizontalFlip()
    ])

    # Set the model up for evaluation
    model.eval()

    # Instantiate the inference model
    inference_model = (
        tta.ClassificationTTAWrapper(WrappedDistortionFaceNet(model), tta_transforms, merge_mode='mean') if args.use_tta
        else WrappedDistortionFaceNet(model)
    )

    # Set the inference model up for evaluation
    inference_model.eval()

    test_labels, test_scores = [], []
    embeddings_dict = {}
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for idxs, inputs, labels in tqdm(test_loader, desc='Test loop'):
            # Send the batch inputs and outputs to the GPU for faster computation
            inputs, labels = inputs.to(device), labels.to(device)
            # Automatically casts operations to mixed precision (float16 and float32) for performance and memory efficiency
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                # Compute embeddings for the inputs by passing them through the model
                embeddings = inference_model(inputs)
            # Convert embeddings to float32
            embeddings = embeddings.float()
            # Record the batch embeddings in a dictionary for future evaluation purposes
            for idx, embedding in zip(idxs, embeddings):
                embeddings_dict[idx.item()] = embedding.cpu()

    # Compute the cosine similarity scores for positive pairs
    for idx1, idx2 in positive_pairs:
        embedding1 = embeddings_dict[idx1].unsqueeze(0).float()
        embedding2 = embeddings_dict[idx2].unsqueeze(0).float()
        score = F.cosine_similarity(embedding1, embedding2).item()
        test_labels.append(1)
        test_scores.append(score)
    # Compute the cosine similarity scores for negative pairs
    for idx1, idx2 in negative_pairs:
        embedding1 = embeddings_dict[idx1].unsqueeze(0).float()
        embedding2 = embeddings_dict[idx2].unsqueeze(0).float()
        score = F.cosine_similarity(embedding1, embedding2).item()
        test_labels.append(0)
        test_scores.append(score)

    # Compute the predictions by comparing the scores with the best cosine threshold
    test_preds = (np.array(test_scores) >= cosine_threshold).astype(int)

    # Compute test metrics: accuracy, precision, recall and F1-score
    accuracy  = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    recall    = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    f1        = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    auc       = roc_auc_score(test_labels, test_scores)

    # Print the results
    print(f'Accuracy:              {accuracy * 100:.2f}%')
    print(f'Precision (macro-avg): {precision * 100:.2f}%')
    print(f'Recall (macro-avg):    {recall * 100:.2f}%')
    print(f'F1-score (macro-avg):  {f1 * 100:.2f}%')
    print(f'ROC-AUC score:         {auc * 100:.2f}%')

    # Define the class names as same or different since this is a Face Verification model
    class_names = ['Different Identity', 'Same Identity']

    # Display the classification report for the model on the test set
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Revert to the default theme for displaying the confusion matrix
    plt.rcdefaults()

    # Display the confusion matrix on the test set
    disp = ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, display_labels=class_names, cmap='Blues')
    disp.figure_.canvas.manager.set_window_title('Face Verifier - Confusion Matrix on the Test Set')
    plt.tight_layout()
    plt.show()
