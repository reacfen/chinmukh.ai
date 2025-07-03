# JSON support library
import json
# OS library
import os

# Matplotlib library (for visualization/plotting purposes)
import matplotlib.pyplot as plt

# Path to the pretrained Chinmukh.ai Face Verifier model
model_dir = './pretrained'

if __name__ == '__main__':
    # Fetch the model's training data for visualization
    try:
        with open(os.path.join(model_dir, 'face_verifier_training_data.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Missing \'face_verifier_training_data.json\' in model\'s root directory.')

    # Extract training data from the JSON
    train_losses = data['train_losses']
    valid_losses = data['valid_losses']
    valid_accs = data['valid_accs']
    valid_precs = data['valid_precs']
    valid_recalls = data['valid_recalls']
    valid_f1s = data['valid_f1s']
    valid_aucs = data['valid_aucs']
    cosine_thresholds = data['cosine_thresholds']
    lrs = data['lrs']
    best_valid_f1 = data['best_valid_f1']
    best_valid_loss = data['best_valid_loss']
    cosine_threshold = data['cosine_threshold']

    # A list of the number of epochs during training
    epochs = list(range(1, len(train_losses) + 1))

    # Set a visual theme for the plot
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot the training (ArcFace) loss over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - ArcFace Loss Curve')
    plt.plot(epochs, train_losses, 'o-', label='Training (ArcFace) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('ArcFace Loss')
    plt.title('Variation in training (ArcFace) loss over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the validation (Cosine Embedding) loss over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - Cosine Embedding Loss Curve')
    plt.plot(epochs, valid_losses, 'o-', label='Validation (Cosine Embedding) Loss')
    plt.axhline(y=best_valid_loss, color='red', linestyle='--', label='Best Validation Loss')
    best_epoch = valid_losses.index(best_valid_loss) + 1
    plt.scatter(best_epoch, best_valid_loss, marker='s', color='green', zorder=5)
    plt.text(best_epoch, best_valid_loss + 0.008, f'Epoch {best_epoch}', ha='center', color='black', fontsize=12)
    plt.text(best_epoch, best_valid_loss - 0.008, f'Best: {best_valid_loss:.4f}', ha='center', va='top', color='black', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Embedding Loss')
    plt.title('Variation in validation (Cosine Embedding) loss over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the training and validation accuracies over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - Accuracy Curve')
    plt.plot(epochs, valid_accs, 'o-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Variation in accuracy over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the training and validation precisions over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - Precision Curve')
    plt.plot(epochs, valid_precs, 'o-', label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Variation in precision over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the training and validation recalls over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - Recall Curve')
    plt.plot(epochs, valid_recalls, 'o-', label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Variation in recall over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the training and validation F1-scores over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - F1-score Curve')
    plt.plot(epochs, valid_f1s, 'o-', label='Validation F1-score')
    plt.axhline(y=best_valid_f1, color='red', linestyle='--', label='Best Validation F1-score')
    best_epoch = valid_f1s.index(best_valid_f1) + 1
    plt.scatter(best_epoch, best_valid_f1, marker='s', color='green', zorder=5)
    plt.text(best_epoch, best_valid_f1 + 0.008, f'Epoch {best_epoch}', ha='center', color='black', fontsize=12)
    plt.text(best_epoch, best_valid_f1 - 0.008, f'Best: {best_valid_f1:.4f}', ha='center', va='top', color='black', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Variation in F1-score over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the validation ROC-AUC scores over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - ROC-AUC scores')
    plt.plot(epochs, valid_aucs, 'o-', label='Validation ROC-AUC score')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC score')
    plt.title('Variation in ROC-AUC score over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the cosine thresholds over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - Cosine Threshold Curve')
    plt.plot(epochs, cosine_thresholds, 'o-', label='Cosine Threshold')
    plt.axhline(y=cosine_threshold, color='red', linestyle='--', label='Best Cosine Threshold')
    best_epoch = cosine_thresholds.index(cosine_threshold) + 1
    plt.scatter(best_epoch, cosine_threshold, marker='o', color='green', zorder=5)
    plt.text(best_epoch, cosine_threshold + 0.008, f'Epoch {best_epoch}', ha='center', color='black', fontsize=12)
    plt.text(best_epoch, cosine_threshold - 0.008, f'Best: {cosine_threshold:.2f}', ha='center', va='top', color='black', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Threshold')
    plt.title('Variation in cosine threshold values over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the learning rates during training over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Face Verifier - Learning Rate Curve')
    plt.plot(epochs, lrs, 'o-', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Variation in learning rate over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
