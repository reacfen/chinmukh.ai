# PyTorch library (for model creation, training and inference)
import torch
# Timm library (for loading architectures pertaining to image classification)
import timm
# JSON support library
import json
# OS library
import os
# Math library
import math
# Argument parser library
import argparse

# Provides a base Module class to build the neural net on
import torch.nn as nn
# Provides various functions for convolution, pooling, attention and non-linear activation operations
import torch.nn.functional as F
# Matplotlib library (for visualization/plotting purposes)
import matplotlib.pyplot as plt
# For TTA (Test-Time Augmentation capabilities)
import ttach as tta

# For applying transforms on the input image
from torchvision import transforms
# PIL library (for image manipulation and processing)
from PIL import Image

# Path to the pretrained Chinmukh.ai Face Verifier model
model_dir = './pretrained'

# Path to the first test image
test_image1 = './Comys_Hackathon5/Task_B/val/Ralph_Klein/Ralph_Klein_0001.jpg'
# Path to the second test image
test_image2 = './Comys_Hackathon5/Task_B/val/Ralph_Klein/distortion/Ralph_Klein_0002_noisy.jpg'

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
    # Check whether both the test images actually exist
    if not os.path.exists(test_image1):
        raise FileNotFoundError(f'First test image not found: {test_image1}')
    if not os.path.exists(test_image2):
        raise FileNotFoundError(f'Second test image not found: {test_image2}')

    # Load images
    img1 = Image.open(test_image1).convert('RGB')
    img2 = Image.open(test_image2).convert('RGB')

    try:
        with open(os.path.join(model_dir, 'face_verifier_training_data.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Missing \'face_verifier_training_data.json\' in model\'s root directory.')

    # Fetch the image size from the model's training data
    image_size = data['image_size']
    # Fetch the embedding dimensions from the model's training data
    embedding_dimensions = data['embedding_dimensions']
    # Fetch the best cosine threshold from the model's training data
    cosine_threshold = data['cosine_threshold']

    # Initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-tta', action='store_true', help='Use TTA (Test-Time Augmentation)')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU for inference')
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
    transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])       # Normalize the data
    ])

    # Apply the required transformation to the test images
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # TTA preprocessing for the test image
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

    # Disable gradient computation during inference
    with torch.no_grad():
        embedding1 = inference_model(img1_tensor)
        embedding2 = inference_model(img2_tensor)

    # Set a visual theme for the plot
    plt.style.use('seaborn-v0_8-whitegrid')

    # Prepare the plot to display the prediction alongside the test image
    fig, axarr = plt.subplots(1, 3, figsize=(10, 6))
    fig.canvas.manager.set_window_title('Face Verifier - Inference Report')

    # De-normalize the test images for visualization (approximation)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    # Convert the denormalized tensors into a NumPy arrays suitable for plotting with 'matplotlib' (H, W, C)
    img1_np = (img1_tensor.squeeze().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    img2_np = (img2_tensor.squeeze().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    # Display the first test image
    axarr[0].imshow(img1_np)
    axarr[0].axis('off')
    axarr[0].set_title("Test Image 1")

    # Display the second test image
    axarr[1].imshow(img2_np)
    axarr[1].axis('off')
    axarr[1].set_title("Test Image 2")

    # Calculate the cosine similarity between the two embeddings
    score = F.cosine_similarity(embedding1, embedding2).item()

    # Decide match based on threshold
    is_same = score >= cosine_threshold

    # Controls the steepness of the exponential curve
    k = 20
    # Logistic confidence scaling
    confidence = 1 / (1 + math.exp(-k * (score - cosine_threshold)))
    confidence = max(min(confidence, 1), 0)

    # Define the class names as same or different since this is a Face Verification model
    class_names = ['Different Identity', 'Same Identity']

    # Show cosine score
    axarr[2].barh(class_names, [1 - confidence, confidence],
                  color=['skyblue' if i != int(is_same)
                        else 'limegreen'
                        for i in range(len(class_names))])
    axarr[2].set_xlabel('Probability')
    axarr[2].set_title('Similarity Predictions')
    axarr[2].set_xlim(0, 1)

    # Display the final plot
    plt.tight_layout()
    plt.show()

    predicted_confidence = confidence if is_same else 1 - confidence

    # Display the final predicted result onto the console
    print({
        'image1_path': test_image1,
        'image2_path': test_image2,
        'score': round(score, 4),
        'threshold': round(cosine_threshold, 4),
        'predicted_similarity': class_names[is_same],
        'confidence': round(predicted_confidence, 4)
    })
