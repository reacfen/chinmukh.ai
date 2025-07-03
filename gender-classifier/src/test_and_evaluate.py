# PyTorch library (for creating the model)
import torch
# Timm library (for loading architectures pertaining to image classification)
import timm
# JSON support library
import json
# OS library
import os
# Argument parser library
import argparse

# Provides a base Module class to build the neural net on
import torch.nn as nn
# Matplotlib library (for visualization/plotting purposes)
import matplotlib.pyplot as plt
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
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, classification_report
)

# Path to the pretrained Chinmukh.ai Gender Classifier model
model_dir = './pretrained'

# Test set directory
test_dir = './Comys_Hackathon5/Task_A/test'

# Used to load the dataset
class FacecomGenderDataset(Dataset):
    # Wraps `ImageFolder` to provide optional transform support and class access
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    # Gets the number of items in the dataset
    def __len__(self):
        return len(self.data)
    # Gets a specific item in the dataset at a specified index
    def __getitem__(self, idx):
        return self.data[idx]
    # Returns a list of the classes in the dataset
    @property
    def classes(self):
        return self.data.classes

# Model that takes an input image and classifies it as male or female
class GenderNet(nn.Module):
    # Initializes the EfficientNet-b1 backbone for binary classification
    def __init__(self):
        # Instantiate parameters from the base class with default values
        super().__init__()
        # Define the base model to train on
        self.base_model = timm.create_model('efficientnet_b1', pretrained=False)
        # Replace the classifier head with a custom linear layer for binary classification
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.base_model.classifier.in_features, 1)
        )
    # Forward pass
    def forward(self, x):
        # Return the output logits from the model
        return self.base_model(x)

# Convert shape (B,) or (B,1) → (B,2) for compatibility with ttach
class WrappedGenderNet(nn.Module):
    # Initialization
    def __init__(self, model):
        # Instantiate parameters from the base class with default values
        super().__init__()
        self.model = model
    # Forward pass
    def forward(self, x):
        logits = self.model(x)
        logits = logits.view(-1)  # shape (B,)
        return torch.stack([1 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=1)

if __name__ == '__main__':
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
    model = GenderNet()
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'gender_classifier_weights.pth'), map_location=device), strict=True)
    # Send the model to the GPU (if available) for faster calculations
    model.to(device)

    # Fetch the model's training data
    try:
        with open(os.path.join(model_dir, 'gender_classifier_training_data.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Missing \'gender_classifier_training_data.json\' in model\'s root directory.')

    # Fetch the image size from the model's training data
    image_size = data['image_size']
    # Fetch the class names from the model's training data
    class_names = data['class_names']
    # Fetch the batch size from the model's training data
    batch_size = data['batch_size']
    # Fetch the validation sigmoid threshold from the model's training data
    valid_sigmoid_threshold = data['valid_sigmoid_threshold']

    # Preprocessing (resize + crop + normalize) for the test set
    test_transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])       # Normalize the data
    ])

    # Import the test data from the dataset and apply the required transformations
    test_dataset = FacecomGenderDataset(test_dir, test_transform)

    # Data loader for the test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == 'cuda',
        persistent_workers=True
    )

    # TTA preprocessing for the test set
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply([0.95, 1.2]),
        tta.Add([0.25])
    ])

    # Set the model up for evaluation
    model.eval()

    # Instantiate the inference model
    inference_model = (
        tta.ClassificationTTAWrapper(WrappedGenderNet(model), tta_transforms, merge_mode='mean') if args.use_tta
        else WrappedGenderNet(model)
    )

    # Set the inference model up for evaluation
    inference_model.eval()

    test_labels, test_preds = [], []
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Test loop'):
            # Send the batch inputs and labels to the GPU for faster computation
            inputs, labels = inputs.to(device), labels.to(device)
            # Automatically casts operations to mixed precision (float16 and float32) for performance and memory efficiency
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                # Calculate the logits using the model with the given inputs
                logits = inference_model(inputs)
            # Record the predictions
            probs = logits[:, 1]
            preds = (probs > valid_sigmoid_threshold).long()
            test_labels.extend(labels.cpu().tolist())
            test_preds.extend(preds.cpu().tolist())

    assert logits.shape[-1] == len(class_names), 'Expected single output logit for binary classification'

    # Compute test metrics: accuracy, precision, recall and F1-score
    accuracy  = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall    = recall_score(test_labels, test_preds, zero_division=0)
    f1        = f1_score(test_labels, test_preds, zero_division=0)

    # Print the results
    print(f'Accuracy:  {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall:    {recall * 100:.2f}%')
    print(f'F1-score:  {f1 * 100:.2f}%')

    # Check whether the classes of the test dataset matches with those of the model
    assert test_dataset.classes == class_names
    assert len(class_names) == 2, 'Mismatch between model and class labels in the test set'

    # Display the classification report for the model on the test set
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Revert to the default theme for displaying the confusion matrix
    plt.rcdefaults()

    # Display the confusion matrix on the test set
    disp = ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, display_labels=class_names, cmap='Blues')
    disp.figure_.canvas.manager.set_window_title('Gender Classifier - Confusion Matrix on the Test Set')
    plt.tight_layout()
    plt.show()
