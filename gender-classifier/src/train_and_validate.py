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
# Provides a bunch of optimizers to use during training
import torch.optim as optim
# Provides various functions for convolution, pooling, attention and non-linear activation operations
import torch.nn.functional as F
# Pandas library
import pandas as pd
# NumPy library
import numpy as np

# For loading the dataset and for a weighted sampler to balance each batch in an imbalanced dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# For automatic mixed precision (AMP) training/inference on supported hardware (e.g., NVIDIA GPUs)
from torch.amp import autocast, GradScaler
# For applying transforms on the input image
from torchvision import transforms
# For loading the dataset as an image folder with the necessary labels
from torchvision.datasets import ImageFolder
# For Exponential Moving Average (EMA)
from torch_ema import ExponentialMovingAverage
# tqdm (provides an ergonomic progress bar during training, validation and inference)
from tqdm import tqdm
# torchmetrics (for evaluation during training and validation)
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
# For stand-alone F1-score calculation
from torchmetrics.functional import accuracy, precision, recall, f1_score

# Path to the pretrained Chinmukh.ai Gender Classifier model (Comment this out if you want to train the model from scratch)
model_dir = './pretrained'
# Path to the output directory where the model will be saved after training completes
model_out_dir = './output'

# Training set directory
train_dir = './Comys_Hackathon5/Task_A/train'
# Validation set directory
valid_dir = './Comys_Hackathon5/Task_A/val'

# Specify the number of epochs/iterations for learning
num_epochs = 50
# Initial learning rate of the model
learning_rate = 5e-5
# Weight decay
weight_decay = 5e-4
# Layer decay
layer_decay = 0.8
# Size of each batch of image tensors
batch_size = 32

# Fixed size of each image
image_size = (224, 224)

# Early stopping patience
patience = 6

# Scalar or tensor of shape [num_classes], class weighting
focal_alpha = None
# Focusing parameter
focal_gamma = 2
# Reduction type of FocalLoss: can be either of 'mean', 'sum', or 'none'
focal_reduction_type = 'mean'

# A fixed sigmoid threshold used during training that doesn't change across epochs
train_sigmoid_threshold = 0.5

# Minimum improvement in F1-score required to consider a model as better
min_delta = 0.001

# Initial seed for the random number generator (RNG)
seed = 42

# Relative error due to rounding in floating-point arithmetic
epsilon = 1e-6

# Sets the RNG seed for deterministic results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# Dataset class for stratified data augmentation
class StratifiedAugmentationDataset(Dataset):
    # Wraps an existing `ImageFolder` dataset
    def __init__(self, raw_data, transforms_by_class):
        self.data = raw_data
        self.transforms_by_class = transforms_by_class
    # Gets the number of items in the dataset
    def __len__(self):
        return len(self.data)
    # Gets a specific item in the dataset at a specified index after apply class-specific transformations to it
    def __getitem__(self, idx):
        image, label = self.data[idx]
        transform = self.transforms_by_class[label]
        image = transform(image)
        return image, label
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
        self.base_model = timm.create_model('efficientnet_b1', pretrained=True)
        # Replace the classifier head with a custom linear layer for binary classification
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.base_model.classifier.in_features, 1)
        )
    # Forward pass
    def forward(self, x):
        # Return the output logits from the model
        return self.base_model(x)

# Binary focal loss emphasizes hard-to-classify samples, effective for handling class imbalance
class BinaryFocalLoss(nn.Module):
    # Define the focal loss with the necessary parameters
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        # Instantiate parameters from the base class with default values
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    # Forward loss calculation
    def forward(self, inputs, targets):
        # Clamp inputs to prevent potential precision errors
        inputs = torch.clamp(inputs, min=-20, max=20)
        # Ensure targets are float tensors of shape (N, 1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        # Calculate BCE (Binary Cross Entropy) loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce_loss = torch.clamp(bce_loss, min=1e-8, max=100)
        # Compute p_t using sigmoid of inputs/logits
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = torch.clamp(p_t, min=1e-8, max=1)
        if self.alpha is not None:
            # Handle class weighting (alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            # No class weighting (fallback to an alpha of 1)
            alpha_t = 1
        # Calculate the focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        # Apply the reduction method
        if self.reduction == 'mean':
            # Average the per-sample losses
            return focal_loss.mean()
        elif self.reduction == 'sum':
            # Sums up the per-sample losses
            return focal_loss.sum()
        elif self.reduction == 'none':
            # Returns the per-sample losses
            return focal_loss
        else:
            raise ValueError(f'Invalid reduction type specified: {self.reduction}')

if __name__ == '__main__':
    # Setting the RNG with a fixed, deterministic seed for reproducibility
    set_seed(seed)

    # Augmentations for the female class
    female_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1), ratio=(0.75, 1.33)),        # Vary aspect and crop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05),
                                scale=(0.95, 1.05), shear=5,
                                interpolation=transforms.InterpolationMode.BILINEAR,
                                fill=[int(0.485*255), int(0.456*255), int(0.406*255)]),
        transforms.RandomRotation(degrees=30),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),                                                               # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),                                     # Normalize the data
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.3), ratio=(0.4, 3.8),
                                 value='random')
    ])

    # Augmentations for the male class
    male_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1), ratio=(0.75, 1.33)),        # Vary aspect and crop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02),
                                scale=(0.98, 1.02), shear=3,
                                fill=[int(0.485*255), int(0.456*255), int(0.406*255)]),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),                                                               # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),                                     # Normalize the data
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.4, 3.3),
                                 value='random')
    ])

    # Import the raw training data from the training directory
    raw_data = ImageFolder(train_dir)

    # Dynamically generate the transformation by class dictionary from the classes' names
    transforms_by_class = {}
    for class_name, idx in raw_data.class_to_idx.items():
        if class_name == 'female':
            transforms_by_class[idx] = female_transform
        else:
            transforms_by_class[idx] = male_transform

    # Preprocessing (resize + crop + normalize) for the validation set
    valid_transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])       # Normalize the data
    ])

    # Wraps an ImageFolder dataset to apply class-specific transforms (used for stratified augmentation)
    train_dataset = StratifiedAugmentationDataset(raw_data, transforms_by_class)
    # Import the validation data from the dataset and apply the required transformations
    valid_dataset = FacecomGenderDataset(valid_dir, valid_transform)

    # Extract integer class labels (targets) from the raw ImageFolder dataset
    targets = [sample[1] for sample in train_dataset.data.imgs]
    # Fetch the class counts for each class
    class_counts = np.bincount(targets)
    # Compute inverse class frequency weights (higher weight for underrepresented classes)
    class_weights = 1 / class_counts
    # Assign sample-level weights based on their class (for use in WeightedRandomSampler)
    sample_weights = torch.tensor([class_weights[t] for t in targets], dtype=torch.float32)

    # Instantiate the weighted sampler with the calculated class weights
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU for training/validation')
    args = parser.parse_args()

    # Device that uses GPU power for training/validation (if available/applicable)
    device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f'Running on GPU: {torch.cuda.get_device_name(device)}')
    else:
        print('Running on CPU')

    # Data loader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == 'cuda',
        persistent_workers=True
    )
    # Data loader for the validation set
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == 'cuda',
        persistent_workers=True
    )

    # Instantiate the model
    model = GenderNet()
    # Send the model to the GPU (if available) for faster calculations during training/validation
    model.to(device)

    print('Class distribution in the training set:', {k: int(v) for k, v in zip(train_dataset.classes, class_counts)})

    if focal_alpha is not None:
        # Pre-defined alpha tensor
        alpha = torch.tensor(focal_alpha, dtype=torch.float32).to(device)
    else:
        # Calculate the alpha tensor dynamically using frequencies of each class in the training set
        inv_freq = 1 / class_counts
        alpha = inv_freq / inv_freq.sum()
        alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

    print('Using focal loss with alpha weights:', [round(a, 4) for a in alpha.cpu().tolist()])

    # Layer groups of EfficientNet-b1 ordered from input → output
    layer_groups = [
        model.base_model.conv_stem,
        model.base_model.bn1,
        *model.base_model.blocks,        # `model.blocks` is an nn.Sequential of 16 layers
        model.base_model.conv_head,
        model.base_model.bn2
    ]
    num_blocks = len(layer_groups)

    # Parameter for each layer to pass to the optimizer
    param_groups = []

    # Each earlier group gets a fraction of the learning rate of the next (specified by the `layer_decay` parameter)
    for i, block in enumerate(layer_groups):
        param_groups.append({
            'params': block.parameters(),
            'lr': learning_rate * (layer_decay ** (num_blocks - i - 1)),      # Decay from output → input
            'weight_decay': weight_decay
        })

    # Add head/classifier with the full learning rate
    param_groups.append({
        'params': model.base_model.classifier.parameters(),
        'lr': learning_rate,
        'weight_decay': weight_decay
    })

    # Define the loss function (Focal loss per sample)
    criterion = BinaryFocalLoss(alpha, focal_gamma, focal_reduction_type)
    # Specify the optimizer to use during training (with regularization using weight decay to prevent overfitting)
    optimizer = optim.AdamW(param_groups)
    # The One-cycle learning rate scheduler helps the model converge faster and prevents it from getting stuck in a bad minima
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group['lr'] for group in param_groups],
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,                                       # 10% warm-up, 90% cooldown
        anneal_strategy='cos',
        div_factor=10,                                       # Start LR = 1e-3 / 25
        final_div_factor=1e4                                 # Final LR = 1e-3 / 1e4
    )
    # Gradient scaler (Prevents underflow of gradients in float16 by scaling up the loss before the backward pass)
    scaler = GradScaler(enabled=device.type == 'cuda')
    # Exponential Moving Average for improved generalization
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    # Define classification metrics to use during training and validation
    train_accuracy_metric = Accuracy(task='binary')
    train_precision_metric = Precision(task='binary', average='weighted')
    train_recall_metric = Recall(task='binary', average='weighted')
    train_f1_metric = F1Score(task='binary', average='weighted')

    # Records the training and validation losses respectively
    train_losses, valid_losses = [], []

    # Records the training and validation accuracies respectively
    train_accs, valid_accs = [], []

    # Records the training and validation precisions respectively
    train_precs, valid_precs = [], []

    # Records the training and validation recalls respectively
    train_recalls, valid_recalls = [], []

    # Records the training and validation F1-scores respectively
    train_f1s, valid_f1s = [], []

    # Records the validation sigmoid threshold values
    valid_sigmoid_thresholds = []

    # Records the learning rates over each epoch
    lrs = []

    # Keeps track of the best validation F1-score across epochs
    best_valid_f1 = -float('inf')

    # Keeps track of the best validation loss across epochs
    best_valid_loss = float('inf')

    # Keeps track of the validation sigmoid threshold across epochs
    valid_sigmoid_threshold = 0

    # Keeps track of epochs with no improvement
    counter = 0

    # Ensure the model's output directory exists, otherwise create it
    os.makedirs(model_out_dir, exist_ok=True)

    # Starting/initial epoch
    start_epoch = 0

    # Find an existing checkpoint (if any)
    for path in [
        os.path.join(model_out_dir, 'gender_classifier_checkpoint.pth'),
        os.path.join(model_dir, 'gender_classifier_checkpoint.pth')
    ]:
        if os.path.exists(path):
            checkpoint_path = path
            break
    else:
        checkpoint_path = None

    # If a checkpoint exists, load model and optimizer states to resume training
    if checkpoint_path:
        # Load training state from existing checkpoint (if found)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if device.type == 'cuda' and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        valid_losses = checkpoint.get('valid_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        valid_accs = checkpoint.get('valid_accs', [])
        train_precs = checkpoint.get('train_precs', [])
        valid_precs = checkpoint.get('valid_precs', [])
        train_recalls = checkpoint.get('train_recalls', [])
        valid_recalls = checkpoint.get('valid_recalls', [])
        train_f1s = checkpoint.get('train_f1s', [])
        valid_f1s = checkpoint.get('valid_f1s', [])
        valid_sigmoid_thresholds = checkpoint.get('valid_sigmoid_thresholds', [])
        lrs = checkpoint.get('lrs', [])
        if 'best_valid_f1' in checkpoint:
            best_valid_f1 = checkpoint['best_valid_f1']
        if 'best_valid_loss' in checkpoint:
            best_valid_loss = checkpoint['best_valid_loss']
        if 'valid_sigmoid_threshold' in checkpoint:
            valid_sigmoid_threshold = checkpoint['valid_sigmoid_threshold']

    for epoch in range(start_epoch, num_epochs):
        # Display the current epoch
        print(f'\n[Epoch: {epoch + 1}/{num_epochs}]')

        # Training phase
        model.train()
        running_loss = 0
        for inputs, labels in tqdm(train_loader, desc='Training loop'):
            # Send the batch inputs and labels to the GPU for faster computation
            inputs, labels = inputs.to(device), labels.to(device)
            # Clears/zeroes out the gradients of all the model parameters that the optimizer is tracking
            optimizer.zero_grad()
            # Automatically casts operations to mixed precision (float16 and float32) for performance and memory efficiency
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                # Calculate the logits using the model with the given inputs
                logits = model(inputs)
                # Turn output logits into FP32
                logits = logits.float()
                # Compute the focal loss by passing in the output logits
                loss = criterion(logits, labels)
            # Backward pass with gradient scaling and update
            scaler.scale(loss).backward()
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients to prevent exploding gradients, improving training stability (if applicable)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # Apply the optimizer step using the scaled gradients
            scaler.step(optimizer)
            # Adjusts the scale factor dynamically based on gradient overflow
            scaler.update()
            # Update the Exponential Moving Average
            ema.update()
            # Step the learning rate scheduler
            scheduler.step()
            # Compute the running loss for the batch
            running_loss += loss.detach().item() * inputs.size(0)
            # Disable gradient computation during evaluation
            with torch.no_grad():
                # Record the predictions
                probs = torch.sigmoid(logits)
                preds = (probs > train_sigmoid_threshold).long()
                preds_cpu, labels_cpu = preds.view(-1).detach().cpu(), labels.detach().cpu()
                train_accuracy_metric.update(preds_cpu, labels_cpu)
                train_precision_metric.update(preds_cpu, labels_cpu)
                train_recall_metric.update(preds_cpu, labels_cpu)
                train_f1_metric.update(preds_cpu, labels_cpu)

        # Record the training loss for the current epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Compute training metrics: accuracy, precision, recall and F1-score
        train_acc       = train_accuracy_metric.compute().item()
        train_precision = train_precision_metric.compute().item()
        train_recall    = train_recall_metric.compute().item()
        train_f1        = train_f1_metric.compute().item()

        # Reset train metric states
        train_accuracy_metric.reset()
        train_precision_metric.reset()
        train_recall_metric.reset()
        train_f1_metric.reset()

        # Record the training accuracy for the current epoch
        train_accs.append(train_acc)

        # Record the training precision for the current epoch
        train_precs.append(train_precision)

        # Record the training recall for the current epoch
        train_recalls.append(train_recall)

        # Record the training F1-score for the current epoch
        train_f1s.append(train_f1)

        # Temporarily swap in the EMA weights for evaluation/validation
        with ema.average_parameters():
            # Validation phase
            model.eval()
            valid_labels, valid_probs = [], []
            running_loss = 0
            # Disable gradient computation during evaluation
            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader, desc='Validation loop'):
                    # Send the batch inputs and labels to the GPU for faster computation
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Automatically casts operations to mixed precision (float16 and float32) for performance and memory efficiency
                    with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        # Calculate the logits using the model with the given inputs
                        logits = model(inputs)
                        # Turn output logits into FP32
                        logits = logits.float()
                        # Compute the focal loss by passing in the output logits
                        loss = criterion(logits, labels)
                    # Compute the running loss for the batch
                    running_loss += loss.detach().item() * inputs.size(0)
                    # Record the probabilities for deferred prediction
                    probs = torch.sigmoid(logits)
                    valid_labels.append(labels.detach().cpu())
                    valid_probs.append(probs.view(-1).detach().cpu())

            # Record the validation loss for the current epoch
            valid_loss = running_loss / len(valid_loader.dataset)
            valid_losses.append(valid_loss)

            # Convert the probabilities and labels to tensors
            valid_probs = torch.cat(valid_probs)
            valid_labels = torch.cat(valid_labels)

            # Sweep over a dynamic threshold to find the best sigmoid threshold for the current batch
            best_f1 = -float('inf')
            best_sigmoid_threshold = 0
            for threshold in torch.arange(0, 1.01, 0.01):
                preds = (valid_probs >= threshold).long()
                f1 = f1_score(preds, valid_labels, task='binary', average='weighted')
                if f1 > best_f1:
                    best_f1 = f1.item()
                    best_sigmoid_threshold = threshold.item()

            # Calculate the validation predictions
            valid_preds = (valid_probs >= best_sigmoid_threshold).long()

            # Compute validation metrics: accuracy, precision, recall and F1-score
            valid_acc       = accuracy(valid_preds, valid_labels, task='binary').item()
            valid_precision = precision(valid_preds, valid_labels, task='binary', average='weighted').item()
            valid_recall    = recall(valid_preds, valid_labels, task='binary', average='weighted').item()
            valid_f1        = best_f1

        # Record the validation accuracy for the current epoch
        valid_accs.append(valid_acc)

        # Record the validation precision for the current epoch
        valid_precs.append(valid_precision)

        # Record the validation recall for the current epoch
        valid_recalls.append(valid_recall)

        # Record the validation F1-score for the current epoch
        valid_f1s.append(valid_f1)

        # Keeps track of the model's improvement across epochs
        if valid_f1 - best_valid_f1 > min_delta or (abs(valid_f1 - best_valid_f1) < epsilon and valid_loss < best_valid_loss):
            if abs(valid_f1 - best_valid_f1) < epsilon:
                print(f'✅ Validation loss improved at tied F1-score: {best_valid_loss:.4f} → {valid_loss:.4f}, saving model...')
            else:
                print(f'✅ Validation F1-score improved: {best_valid_f1:.4f} → {valid_f1:.4f}, saving model...')
                best_valid_f1 = valid_f1
            best_valid_loss = valid_loss
            valid_sigmoid_threshold = best_sigmoid_threshold
            counter = 0
            model.float()
            # Temporarily swap in the EMA weights for saving
            with ema.average_parameters():
                # Save the current best model
                torch.save(model.state_dict(), os.path.join(model_out_dir, 'gender_classifier_weights.pth'))
                # Export model using TorchScript for efficient and portable deployment
                model.eval()
                try:
                    scripted_model = torch.jit.script(model)
                    scripted_model.save(os.path.join(model_out_dir, 'gender_classifier_scripted.pt'))
                except Exception as e:
                    print(f'⚠️ TorchScript export failed: {e}')
            # Save the current state as a checkpoint from where training can be resumed later if needed
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_accs': train_accs,
                'valid_accs': valid_accs,
                'train_precs': train_precs,
                'valid_precs': valid_precs,
                'train_recalls': train_recalls,
                'valid_recalls': valid_recalls,
                'train_f1s': train_f1s,
                'valid_f1s': valid_f1s,
                'valid_sigmoid_thresholds': valid_sigmoid_thresholds,
                'lrs': lrs,
                'best_valid_f1': best_valid_f1,
                'best_valid_loss': best_valid_loss,
                'valid_sigmoid_threshold': valid_sigmoid_threshold
            }
            if scaler.is_enabled():
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_data, os.path.join(model_out_dir, 'gender_classifier_checkpoint.pth'))
        else:
            counter += 1
            if abs(valid_f1 - best_valid_f1) < epsilon:
                print(f'⏸️ Validation loss did not improve at tied F1-score for {counter}/{patience} consecutive epochs: {best_valid_loss:.4f} → {valid_loss:.4f}')
            else:
                print(f'⏸️ Validation F1-score did not improve by {min_delta} for {counter}/{patience} consecutive epochs: {best_valid_f1:.4f} → {valid_f1:.4f}')

        # Record the current learning rate of the classifier head
        current_lr = scheduler.get_last_lr()[-1]
        lrs.append(current_lr)

        # Record the sigmoid threshold for the current epoch
        valid_sigmoid_thresholds.append(valid_sigmoid_threshold)

        # Display the current learning rate of the classifier head
        print(f'Current Classifier Learning Rate: {current_lr:.6f}')

        # Display the current best sigmoid threshold
        print(f'Current Validation Sigmoid Threshold: {valid_sigmoid_threshold:.2f}')

        # Display the current state of the epoch
        train_loss_str = f'{train_loss:.4f}'
        valid_loss_str = f'{valid_loss:.4f}'
        loss_header_size = max(len(train_loss_str), len(valid_loss_str))
        loss_header = 'Loss'.center(loss_header_size, ' ')
        train_loss_col = train_loss_str.rjust(loss_header_size, ' ')
        valid_loss_col = valid_loss_str.rjust(loss_header_size, ' ')
        print(f'   Phase   | {loss_header} | Accuracy | Precision | Recall | F1-score')
        print('-----------+-' + '-' * loss_header_size + '-+----------+-----------+--------+----------')
        print(f'Training   | {train_loss_col} |   {train_acc:.4f} |    {train_precision:.4f} | {train_recall:.4f} |   {train_f1:.4f}')
        print(f'Validation | {valid_loss_col} |   {valid_acc:.4f} |    {valid_precision:.4f} | {valid_recall:.4f} |   {valid_f1:.4f}')

        if counter >= patience:
            # Stop training if no improvement in validation F1 for 'patience' epochs
            print('🛑 Early stopping triggered! Learning has stopped.')
            break

    # Save training history and metadata to JSON (for analysis or debugging)
    training_data = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs,
        'train_precs': train_precs,
        'valid_precs': valid_precs,
        'train_recalls': train_recalls,
        'valid_recalls': valid_recalls,
        'train_f1s': train_f1s,
        'valid_f1s': valid_f1s,
        'valid_sigmoid_thresholds': valid_sigmoid_thresholds,
        'lrs': lrs,
        'best_valid_f1': best_valid_f1,
        'best_valid_loss': best_valid_loss,
        'valid_sigmoid_threshold': valid_sigmoid_threshold,
        'class_weights': alpha.cpu().tolist(),
        'batch_size': batch_size,
        'image_size': image_size,
        'class_names': train_dataset.classes
    }
    with open(os.path.join(model_out_dir, 'gender_classifier_training_data.json'), 'w') as f:
        json.dump(training_data, f)

    # Save training and validation metrics per epoch to CSV for further analysis or visualization
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs,
        'train_precs': train_precs,
        'valid_precs': valid_precs,
        'train_recalls': train_recalls,
        'valid_recalls': valid_recalls,
        'train_f1s': train_f1s,
        'valid_f1s': valid_f1s,
        'valid_sigmoid_thresholds': valid_sigmoid_thresholds,
        'lrs': lrs
    })
    metrics_df.to_csv(os.path.join(model_out_dir, 'gender_classifier_metrics_log.csv'), index=False)
