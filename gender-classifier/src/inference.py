# PyTorch library (for model creation, training and inference)
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
# NumPy library
import numpy as np
# For TTA (Test-Time Augmentation capabilities)
import ttach as tta

# For applying transforms on the input image
from torchvision import transforms
# PIL library (for image manipulation and processing)
from PIL import Image

# Path to the pretrained Chinmukh.ai Gender Classifier model
model_dir = './pretrained'

# Path to the test image
test_image = './Comys_Hackathon5/Task_A/val/male/Ben_Glisan_0001.jpg'

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
    model = GenderNet()
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'gender_classifier_weights.pth'), map_location=device), strict=True)
    # Send the model to the GPU (if available) for faster calculations
    model.to(device)

    # Check whether the test image actually exists
    if not os.path.exists(test_image):
        raise FileNotFoundError(f'Test image not found: {test_image}')

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
    # Fetch the validation sigmoid threshold from the model's training data
    valid_sigmoid_threshold = data['valid_sigmoid_threshold']

    # Preprocessing (resize + crop + normalize) for the test image
    transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                                # Convert to a PyTorch tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])       # Normalize the data
    ])

    # Open the image for manipulation using PIL
    image = Image.open(test_image).convert('RGB')
    # Apply the required transformation to the test image and send it to the GPU (if applicable)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # TTA preprocessing for the test image
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

    # Disable gradient computation during inference
    with torch.no_grad():
        # Calculate the logits using the model with the given input
        logit = inference_model(image_tensor)
        # Turn the logit into a probability using sigmoid
        prob = logit[:, 1]
        # Compare against the validation threshold to get the prediction
        pred = (prob > valid_sigmoid_threshold).long()

    prob = prob.detach().item()
    class_probabilities = [1 - prob, prob]

    # Set a visual theme for the plot
    plt.style.use('seaborn-v0_8-whitegrid')

    # Prepare the plot to display the prediction alongside the test image
    fig, axarr = plt.subplots(1, 2, figsize=(10, 6))
    fig.canvas.manager.set_window_title('Gender Classifier - Inference Report')

    # De-normalize the test image for visualization (approximation)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    # Convert the denormalized tensor into a NumPy array suitable for plotting with 'matplotlib' (H, W, C)
    image_np = (image_tensor.squeeze().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    # Display the image
    axarr[0].imshow(image_np)
    axarr[0].axis('off')
    axarr[0].set_title('Test Image')

    # Show class probabilities
    axarr[1].barh(class_names, class_probabilities,
                  color=['limegreen' if i == pred
                        else 'skyblue'
                        for i in range(len(class_names))])
    axarr[1].set_xlabel('Probability')
    axarr[1].set_title('Class Predictions')
    axarr[1].set_xlim(0, 1)

    # Display the final plot
    plt.tight_layout()
    plt.show()

    # Display the final predicted result onto the console
    print({
        'image_path': test_image,
        'predicted_label': class_names[pred],
        'confidence': round(max(class_probabilities), 4)
    })
