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
# Math library
import math
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

# For loading the dataset
from torch.utils.data import Dataset, DataLoader
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
# scikit-learn metrics (for evaluation purposes)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# A dictionary that assigns a default value for non-existent keys
from collections import defaultdict
# For generating combinations of samples from lists
from itertools import combinations
# For balanced batch sampling
from pytorch_metric_learning.samplers import MPerClassSampler

# Path to the pretrained Chinmukh.ai Face Verifier model (Comment this out if you want to train the model from scratch)
model_dir = './pretrained'
# Path to the output directory where the model will be saved after training completes
model_out_dir = './output'

# Training set directory
train_dir = './Comys_Hackathon5/Task_B/train'
# Validation set directory
valid_dir = './Comys_Hackathon5/Task_B/val'

# Specify the number of epochs/iterations for learning
num_epochs = 50
# Initial learning rate of the model
learning_rate = 5e-4
# Weight decay
weight_decay = 5e-4
# Layer decay
layer_decay = 0.8
# Size of each batch of image tensors
batch_size = 16

# Fixed size of each image
image_size = (224, 224)

# Early stopping patience
patience = 6

# Number of dimensions of the output embedding of the model
embedding_dimensions = 512

# ArcFace angular margin
angular_margin = 0.5
# ArcFace scaling factor
scaling_factor = 30

# Minimum improvement in F1-score required to consider a model as better
min_delta = 0.0005

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
        self.base_model = timm.create_model('resnet50', pretrained=True, num_classes=0)
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

# Calculates the Arc Margin Product
class ArcMarginProduct(nn.Module):
    # Define the Arc Face Product with the necessary parameters
    def __init__(self, num_classes, embedding_dimensions=512, s=30, m=0.5, easy_margin=False):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions      # Embedding size
        self.num_classes = num_classes                        # Number of classes
        self.s = s                                            # Scale factor
        self.m = m                                            # Angular margin
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.embedding_dimensions))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    # Forward loss calculation
    def forward(self, x, labels):
        labels = labels.long()
        # Convert the ArcFace loss weights to float32 to prevent gradient corruption due to AMP usage
        W = F.normalize(self.weight.float())
        # Normalize input and weights
        cos_theta = torch.clamp(F.linear(x.float(), W), -1 + epsilon, 1 - epsilon)      # cos(theta)
        sin_theta = torch.sqrt(1 - torch.clamp(cos_theta ** 2, 0, 1))           # sin(theta)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m                   # phi = cos(theta + m)
        if self.easy_margin:
            # Only apply margin when cos(theta) > 0
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            # Apply margin always, but handle very small cos(theta) with a threshold
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # Convert the labels to one-hot encodings
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        # Apply the ArcFace transformation
        logits = (one_hot * phi) + ((1 - one_hot) * cos_theta)
        # Scale the logits
        logits = logits * self.s
        return logits

if __name__ == '__main__':
    # Setting the RNG with a fixed, deterministic seed for reproducibility
    set_seed(seed)

    # Preprocessing for the training set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1), ratio=(0.85, 1)),      # Vary aspect and crop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),                                                          # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])                                 # Normalize the data
    ])

    # Preprocessing (resize + crop + normalize) for the validation set
    valid_transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])       # Normalize the data
    ])

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

    # Import the training data from the dataset
    train_dataset = FacecomVerificationDataset(train_dir, train_transform)
    # Import the validation data from the dataset
    valid_dataset = FacecomVerificationDataset(valid_dir, valid_transform)

    # Using a balanced batch sampler which improves intra-class exposure of the model to different identities
    balanced_sampler = MPerClassSampler(train_dataset.data.targets, m=4, length_before_new_iter=len(train_dataset))

    # Data loader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler,
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
    model = DistortionFaceNet(embedding_dimensions)
    # Send the model to the GPU (if available) for faster calculations during training/validation
    model.to(device)

    # Layer groups of ResNet-50 ordered from input → output
    layer_groups = layer_groups = [
        nn.Sequential(model.base_model.conv1, model.base_model.bn1, model.base_model.act1, model.base_model.maxpool),
        model.base_model.layer1,
        model.base_model.layer2,
        model.base_model.layer3,
        model.base_model.layer4,
        model.base_model.global_pool,
        model.embedding
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

    # Define the Arc Margin Product
    arcface_head = ArcMarginProduct(len(train_dataset.classes), embedding_dimensions, m=angular_margin, s=scaling_factor).to(device)
    # Add ArcMarginProduct as a separate param group
    param_groups.append({
        'params': arcface_head.parameters(),
        'lr': learning_rate,
        'weight_decay': weight_decay
    })
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
    # Cosine Embedding Loss for validation
    cos_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')

    # Records the training and validation losses respectively
    train_losses, valid_losses = [], []

    # Records the training and validation accuracies respectively
    valid_accs = []

    # Records the training and validation precisions respectively
    valid_precs = []

    # Records the training and validation recalls respectively
    valid_recalls = []

    # Records the training and validation F1-scores respectively
    valid_f1s = []

    # Records the validation ROC-AUC scores
    valid_aucs = []

    # Records the cosine threshold values
    cosine_thresholds = []

    # Records the learning rates over each epoch
    lrs = []

    # Keeps track of the best validation F1-score across epochs
    best_valid_f1 = -float('inf')

    # Keeps track of the best validation loss across epochs
    best_valid_loss = float('inf')

    # Keeps track of the cosine threshold across epochs
    cosine_threshold = 0

    # Keeps track of epochs with no improvement
    counter = 0

    # Ensure the model's output directory exists, otherwise create it
    os.makedirs(model_out_dir, exist_ok=True)

    # Starting/initial epoch
    start_epoch = 0

    # Find an existing checkpoint (if any)
    for path in [
        os.path.join(model_out_dir, 'face_verifier_checkpoint.pth'),
        os.path.join(model_dir, 'face_verifier_checkpoint.pth')
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
        arcface_head.load_state_dict(checkpoint['arcface_head_state_dict'])
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
        valid_accs = checkpoint.get('valid_accs', [])
        valid_precs = checkpoint.get('valid_precs', [])
        valid_recalls = checkpoint.get('valid_recalls', [])
        valid_f1s = checkpoint.get('valid_f1s', [])
        valid_aucs = checkpoint.get('valid_aucs', [])
        cosine_thresholds = checkpoint.get('cosine_thresholds', [])
        lrs = checkpoint.get('lrs', [])
        if 'best_valid_f1' in checkpoint:
            best_valid_f1 = checkpoint['best_valid_f1']
        if 'best_valid_loss' in checkpoint:
            best_valid_loss = checkpoint['best_valid_loss']
        if 'cosine_threshold' in checkpoint:
            cosine_threshold = checkpoint['cosine_threshold']

    # A dictionary that maps an identity (label) to indices of each of their images
    identity_to_idxs = defaultdict(list)
    for idx, _, label in valid_dataset:
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

    for epoch in range(start_epoch, num_epochs):
        # Display the current epoch
        print(f'\n[Epoch: {epoch + 1}/{num_epochs}]')

        # Training phase
        model.train()
        running_loss = 0
        for _, inputs, labels in tqdm(train_loader, desc='Training loop'):
            # Send the batch inputs and labels to the GPU for faster computation
            inputs, labels = inputs.to(device), labels.to(device)
            # Clears/zeroes out the gradients of all the model parameters that the optimizer is tracking
            optimizer.zero_grad()
            # Automatically casts operations to mixed precision (float16 and float32) for performance and memory efficiency
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                # Compute embeddings for the inputs by passing them through the model
                embeddings = model(inputs)
                # Compute the logits using ArcMarginProduct
                logits = arcface_head(embeddings, labels)
                # Calculate the loss using Cross-Entropy loss
                loss = F.cross_entropy(logits, labels)
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

        # Record the training loss for the current epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Temporarily swap in the EMA weights for evaluation/validation
        with ema.average_parameters():
            # Validation phase
            model.eval()
            arcface_head.eval()
            running_loss = 0
            valid_labels, valid_scores = [], []
            embeddings_dict = {}
            # Disable gradient computation during evaluation
            with torch.no_grad():
                for idxs, inputs, labels in tqdm(valid_loader, desc='Validation loop'):
                    # Send the batch inputs and labels to the GPU for faster computation
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Automatically casts operations to mixed precision (float16 and float32) for performance and memory efficiency
                    with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        # Compute embeddings for the inputs by passing them through the model
                        embeddings = model(inputs)
                    # Convert embeddings to float32
                    embeddings = embeddings.float()
                    # Record the batch embeddings in a dictionary for future evaluation purposes
                    for idx, embedding in zip(idxs, embeddings):
                        embeddings_dict[idx.item()] = embedding.cpu()

            # Generate batches of positive and negative embeddings' pairs to forward through the `CosineEmbeddingLoss`
            pos_emb1, pos_emb2 = [], []
            neg_emb1, neg_emb2 = [], []
            for idx1, idx2 in positive_pairs:
                pos_emb1.append(embeddings_dict[idx1])
                pos_emb2.append(embeddings_dict[idx2])
            for idx1, idx2 in negative_pairs:
                neg_emb1.append(embeddings_dict[idx1])
                neg_emb2.append(embeddings_dict[idx2])
            # Check if list is not empty before stacking
            if pos_emb1:
                # Only move to device if they are not empty
                pos_emb1 = torch.stack(pos_emb1).to(device)
                pos_emb2 = torch.stack(pos_emb2).to(device)
            else:
                pos_emb1 = torch.empty(0, embedding_dimensions).to(device)
                pos_emb2 = torch.empty(0, embedding_dimensions).to(device)
            if neg_emb1:
                neg_emb1 = torch.stack(neg_emb1).to(device)
                neg_emb2 = torch.stack(neg_emb2).to(device)
            else:
                neg_emb1 = torch.empty(0, embedding_dimensions).to(device)
                neg_emb2 = torch.empty(0, embedding_dimensions).to(device)
            # Combine into full batches
            all_emb1 = torch.cat([pos_emb1, neg_emb1], dim=0)
            all_emb2 = torch.cat([pos_emb2, neg_emb2], dim=0)
            # Labels: 1 for positive, -1 for negative
            targets = torch.cat([
                torch.ones(pos_emb1.size(0), dtype=torch.float, device=device),
                -torch.ones(neg_emb1.size(0), dtype=torch.float, device=device)
            ])
            # Record the validation loss for the current epoch
            if all_emb1.size(0) > 0:
                # Only calculate if there are any pairs
                valid_loss = cos_loss(all_emb1, all_emb2, targets).item()
            else:
                # No pairs to evaluate loss on
                valid_loss = 0
            valid_losses.append(valid_loss)

            # Compute the cosine similarity scores for positive pairs
            if pos_emb1.size(0) > 0:
                pos_scores = F.cosine_similarity(pos_emb1, pos_emb2, dim=1).cpu().numpy()
                valid_scores.extend(pos_scores)
                valid_labels.extend([1] * pos_emb1.size(0))
            # Compute the cosine similarity scores for negative pairs
            if neg_emb1.size(0) > 0:
                neg_scores = F.cosine_similarity(neg_emb1, neg_emb2, dim=1).cpu().numpy()
                valid_scores.extend(neg_scores)
                valid_labels.extend([0] * neg_emb1.size(0))

            # Compute the predictions by comparing the scores with a dynamic threshold
            best_f1 = -float('inf')
            best_cosine_threshold = 0
            valid_preds = None
            # Iterate over thresholds only if there are actual scores to evaluate
            if len(valid_scores) > 0:
                for threshold in np.arange(0.1, 0.9, 0.01):
                    sweep_preds = (np.array(valid_scores) >= threshold).astype(int)
                    sweep_f1 = f1_score(valid_labels, sweep_preds, average='macro', zero_division=0)
                    if sweep_f1 > best_f1:
                        best_f1 = sweep_f1
                        best_cosine_threshold = threshold
                        valid_preds = sweep_preds
            else:
                valid_preds = np.array([])

            # Compute validation metrics: accuracy, precision, recall, F1-score and ROC-AUC score
            valid_acc       = accuracy_score(valid_labels, valid_preds)
            valid_precision = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
            valid_recall    = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
            valid_f1        = best_f1
            valid_auc       = roc_auc_score(valid_labels, valid_scores)

        # Record the validation accuracy for the current epoch
        valid_accs.append(valid_acc)

        # Record the validation precision for the current epoch
        valid_precs.append(valid_precision)

        # Record the validation recall for the current epoch
        valid_recalls.append(valid_recall)

        # Record the validation F1-score for the current epoch
        valid_f1s.append(valid_f1)

        # Record the validation ROC-AUC score for the current epoch
        valid_aucs.append(valid_auc)

        # Keeps track of the model's improvement across epochs
        if valid_f1 - best_valid_f1 > min_delta or (abs(valid_f1 - best_valid_f1) < epsilon and valid_loss < best_valid_loss):
            if abs(valid_f1 - best_valid_f1) < epsilon:
                print(f'✅ Validation loss improved at tied F1-score: {best_valid_loss:.4f} → {valid_loss:.4f}, saving model...')
            else:
                print(f'✅ Validation F1-score improved: {best_valid_f1:.4f} → {valid_f1:.4f}, saving model...')
                best_valid_f1 = valid_f1
            best_valid_loss = valid_loss
            cosine_threshold = best_cosine_threshold
            counter = 0
            model.float()
            # Temporarily swap in the EMA weights for saving
            with ema.average_parameters():
                # Save the current best model
                torch.save(model.state_dict(), os.path.join(model_out_dir, 'face_verifier_weights.pth'))
                # Export model using TorchScript for efficient and portable deployment
                model.eval()
                try:
                    scripted_model = torch.jit.script(model)
                    scripted_model.save(os.path.join(model_out_dir, 'face_verifier_scripted.pt'))
                except Exception as e:
                    print(f'⚠️ TorchScript export failed: {e}')
            # Save the current state as a checkpoint from where training can be resumed later if needed
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'arcface_head_state_dict': arcface_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_accs': valid_accs,
                'valid_precs': valid_precs,
                'valid_recalls': valid_recalls,
                'valid_f1s': valid_f1s,
                'valid_aucs': valid_aucs,
                'cosine_thresholds': cosine_thresholds,
                'lrs': lrs,
                'best_valid_f1': best_valid_f1,
                'best_valid_loss': best_valid_loss,
                'cosine_threshold': cosine_threshold
            }
            if scaler.is_enabled():
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_data, os.path.join(model_out_dir, 'face_verifier_checkpoint.pth'))
        else:
            counter += 1
            if abs(valid_f1 - best_valid_f1) < epsilon:
                print(f'⏸️ Validation loss did not improve at tied F1-score for {counter}/{patience} consecutive epochs: {best_valid_loss:.4f} → {valid_loss:.4f}')
            else:
                print(f'⏸️ Validation F1-score did not improve by {min_delta} for {counter}/{patience} consecutive epochs: {best_valid_f1:.4f} → {valid_f1:.4f}')

        # Record the current learning rate of the classifier head
        current_lr = scheduler.get_last_lr()[-1]
        lrs.append(current_lr)

        # Record the cosine threshold for the current epoch
        cosine_thresholds.append(cosine_threshold)

        # Display the current learning rate of the classifier head
        print(f'Current Classifier Learning Rate: {current_lr:.6f}')

        # Display the current cosine threshold
        print(f'Current Cosine Threshold: {cosine_threshold:.2f}')

        # Display the current state of the epoch
        train_loss_str = f'{train_loss:.4f}'
        valid_loss_str = f'{valid_loss:.4f}'
        loss_header_size = max(len(train_loss_str), len(valid_loss_str))
        loss_header = 'Loss'.center(loss_header_size, ' ')
        train_loss_col = train_loss_str.rjust(loss_header_size, ' ')
        valid_loss_col = valid_loss_str.rjust(loss_header_size, ' ')
        print(f'   Phase   | {loss_header} | Accuracy | Precision | Recall | F1-score | ROC-AUC score')
        print('-----------+-' + '-' * loss_header_size + '-+----------+-----------+--------+----------+---------------')
        print(f'Training   | {train_loss_col} |     --   |      --   |   --   |     --   |          --')
        print(f'Validation | {valid_loss_col} |   {valid_acc:.4f} |    {valid_precision:.4f} | {valid_recall:.4f} |   {valid_f1:.4f} |        {valid_auc:.4f}')

        if counter >= patience:
            # Stop training if no improvement in validation F1 for 'patience' epochs
            print('🛑 Early stopping triggered! Learning has stopped.')
            break

    # Save training history and metadata to JSON (for analysis or debugging)
    training_data = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs,
        'valid_precs': valid_precs,
        'valid_recalls': valid_recalls,
        'valid_f1s': valid_f1s,
        'valid_aucs': valid_aucs,
        'cosine_thresholds': cosine_thresholds,
        'lrs': lrs,
        'best_valid_f1': best_valid_f1,
        'best_valid_loss': best_valid_loss,
        'cosine_threshold': cosine_threshold,
        'embedding_dimensions': embedding_dimensions,
        'batch_size': batch_size,
        'image_size': image_size
    }
    with open(os.path.join(model_out_dir, 'face_verifier_training_data.json'), 'w') as f:
        json.dump(training_data, f)

    # Save training and validation metrics per epoch to CSV for further analysis or visualization
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs,
        'valid_precs': valid_precs,
        'valid_recalls': valid_recalls,
        'valid_f1s': valid_f1s,
        'valid_aucs': valid_aucs,
        'cosine_thresholds': cosine_thresholds,
        'lrs': lrs
    })
    metrics_df.to_csv(os.path.join(model_out_dir, 'face_verifier_metrics_log.csv'), index=False)
