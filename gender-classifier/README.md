# Task A: Gender Classifier

[« Click here to visit the parent repository](../)

[» Click here to see training visualization plots](./visualization/)

## Results

| Phase      | TTA Status | Accuracy | Precision | Recall | F1-score |
|------------|------------|----------|-----------|--------|----------|
| Train      | Disabled   | 99.01%   | 99.87%    | 98.89% | 99.38%   |
| Validation | Disabled   | 95.73%   | 96.57%    | 97.79% | 97.18%   |
| Validation | Enabled    | 95.73%   | 96.87%    | 97.48% | 97.17%   |

## Architecture

The gender classifier is a fine-tuned **EfficientNet-b1** model that takes an input image of size `(224 × 224)` and outputs either `0` (representing **female**) or `1` (representing **male**).

Here is a block diagram of the underlying model architecture using the EfficientNet-b1 backbone:

```mermaid
%%{ init: { "flowchart": { "wrappingWidth": 500 } } }%%
graph TD
    A["Input Image<br/>(3 × 224 × 224)"] --> B[EfficientNet-B1 Backbone]

    subgraph EfficientNet-B1
        B -- Input to Subgraph --> B1["Stem<br/>Conv2d (3→32, 3×3, stride=2)<br/>BN + Swish<br/>(32 × 120 × 120)"]
        
        B1 --> B2["MBConv1<br/>32→16, SE, stride=1<br/>(16 × 120 × 120)"]
        B2 --> B3["MBConv6<br/>16→24, SE, stride=2<br/>(24 × 60 × 60)"]
        B3 --> B4["MBConv6<br/>24→40, SE, stride=2<br/>(40 × 30 × 30)"]
        B4 --> B5["MBConv6<br/>40→80, SE, stride=2<br/>(80 × 15 × 15)"]
        B5 --> B6["MBConv6<br/>80→112, SE, stride=1<br/>(112 × 15 × 15)"]
        B6 --> B7["MBConv6<br/>112→192, SE, stride=2<br/>(192 × 8 × 8)"]
        B7 --> B8["MBConv6<br/>192→320, SE, stride=1<br/>(320 × 8 × 8)"]

        B8 --> B9["Head<br/>Conv2d (320→1280, 1×1)<br/>BN + Swish<br/>(1280 × 8 × 8)"]
        B9 --> B10["AdaptiveAvgPool2d<br/>(1280 × 1 × 1)"]
        B10 --> B11["Flatten<br/>(1280)"]
    end

    B11 --> C[Custom Classifier Head]

    subgraph Classifier Head
        C --> D["Dropout (p=0.4)"]
        D --> E["Linear (1280 → 1)"]
        E --> F["Output Logit<br/>(Gender Score)"]
    end

    F --> G["Sigmoid<br/>Gender Probability"]
```

Additionally, to help the model generalize and learn discriminative features across each gender, the following techniques were incorporated into the training pipeline:

- **Gradient Scaling/`autocast()` (When CUDA is enabled):** Enables mixed-precision training using PyTorch's `autocast()` and `GradScaler`, reducing memory usage and speeding up training by using lower precision (e.g., float16) where safe, while maintaining full precision for sensitive operations like gradient calculations.

- **LLRD (Layer-wise Learning Rate Decay):** Applies progressively smaller learning rates to earlier layers (typically pre-trained), allowing newer layers (e.g., the classifier head) to adapt more aggressively, while preserving useful features in earlier layers.

- **Weighted Sampler:** Balances class distribution during training by assigning higher sampling probabilities to underrepresented classes, ensuring that each batch contains a more even representation of both genders.

- **Stratified Augmentation:** Applies augmentation strategies tailored to each class, aiming to capture class-specific variability and mitigate imbalances in data representation and quality.

- **Binary Focal Loss:** A loss function that down-weights easy examples and focuses learning on harder, misclassified samples. Particularly effective for imbalanced binary classification tasks.

- **Sigmoid Threshold Sweep:** Involves evaluating performance across a range of sigmoid output thresholds (instead of default 0.5) to select the optimal decision threshold, based on metrics like F1-score.

- **One-Cycle Learning Rate Scheduling:** A cyclical learning rate strategy where the learning rate increases rapidly and then decays slowly within a single training cycle, encouraging exploration early and convergence later.

- **Early Stopping Mechanism:** Monitors a validation metric and halts training when performance ceases to improve for a defined number of epochs, reducing overfitting and training time.

- **AdamW Optimizer:** A variant of the Adam optimizer that decouples weight decay from the gradient update, enabling better generalization and improved handling of overfitting through proper [L2 regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization).

- **EMA (Exponential Moving Average):** Maintains a moving average of model weights during training, smoothing out fluctuations and often yielding a more robust final model for evaluation or inference.

- **TTA (Test-Time Augmentation):** Applies multiple augmentations (e.g., flips, crops) to each test image and averages the predictions, improving inference robustness and accounting for potential distribution shifts.
