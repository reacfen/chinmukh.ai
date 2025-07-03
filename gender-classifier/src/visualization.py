# JSON support library
import json
# OS library
import os

# Matplotlib library (for visualization/plotting purposes)
import matplotlib.pyplot as plt

# Path to the pretrained Chinmukh.ai Gender Classifier model
model_dir = './pretrained'

if __name__ == '__main__':
    # Fetch the model's training data for visualization
    try:
        with open(os.path.join(model_dir, 'gender_classifier_training_data.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Missing \'gender_classifier_training_data.json\' in model\'s root directory.')

    # Extract training data from the JSON
    train_losses = data['train_losses']
    valid_losses = data['valid_losses']
    train_accs = data['train_accs']
    valid_accs = data['valid_accs']
    train_precs = data['train_precs']
    valid_precs = data['valid_precs']
    train_recalls = data['train_recalls']
    valid_recalls = data['valid_recalls']
    train_f1s = data['train_f1s']
    valid_f1s = data['valid_f1s']
    valid_sigmoid_thresholds = data['valid_sigmoid_thresholds']
    lrs = data['lrs']
    best_valid_f1 = data['best_valid_f1']
    best_valid_loss = data['best_valid_loss']
    valid_sigmoid_threshold = data['valid_sigmoid_threshold']

    # A list of the number of epochs during training
    epochs = list(range(1, len(train_losses) + 1))

    # Set a visual theme for the plot
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot the training and validation losses over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Gender Classifier - Loss Curve')
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.plot(epochs, valid_losses, 's-', label='Validation Loss')
    plt.axhline(y=best_valid_loss, color='red', linestyle='--', label='Best Validation Loss')
    best_epoch = valid_losses.index(best_valid_loss) + 1
    plt.scatter(best_epoch, best_valid_loss, marker='s', color='green', zorder=5)
    plt.text(best_epoch, best_valid_loss + 0.008, f'Epoch {best_epoch}', ha='center', color='black', fontsize=12)
    plt.text(best_epoch, best_valid_loss - 0.008, f'Best: {best_valid_loss:.4f}', ha='center', va='top', color='black', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Variation in loss over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the training and validation accuracies over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Gender Classifier - Accuracy Curve')
    plt.plot(epochs, train_accs, 'o-', label='Training Accuracy')
    plt.plot(epochs, valid_accs, 's-', label='Validation Accuracy')
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
    fig.canvas.manager.set_window_title('Gender Classifier - Precision Curve')
    plt.plot(epochs, train_precs, 'o-', label='Training Precision')
    plt.plot(epochs, valid_precs, 's-', label='Validation Precision')
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
    fig.canvas.manager.set_window_title('Gender Classifier - Recall Curve')
    plt.plot(epochs, train_recalls, 'o-', label='Training Recall')
    plt.plot(epochs, valid_recalls, 's-', label='Validation Recall')
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
    fig.canvas.manager.set_window_title('Gender Classifier - F1-score Curve')
    plt.plot(epochs, train_f1s, 'o-', label='Training F1-score')
    plt.plot(epochs, valid_f1s, 's-', label='Validation F1-score')
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

    # Plot the sigmoid thresholds over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Gender Classifier - Sigmoid Threshold Curve')
    plt.plot(epochs, valid_sigmoid_thresholds, 'o-', label='Sigmoid Threshold')
    plt.axhline(y=0.5, color='blue', linestyle='--', label='Standard Sigmoid Threshold')
    plt.axhline(y=valid_sigmoid_threshold, color='red', linestyle='--', label='Best Sigmoid Threshold')
    best_epoch = valid_sigmoid_thresholds.index(valid_sigmoid_threshold) + 1
    plt.scatter(best_epoch, valid_sigmoid_threshold, marker='o', color='green', zorder=5)
    plt.text(best_epoch, valid_sigmoid_threshold + 0.008, f'Epoch {best_epoch}', ha='center', color='black', fontsize=12)
    plt.text(best_epoch, valid_sigmoid_threshold - 0.008, f'Best: {valid_sigmoid_threshold:.2f}', ha='center', va='top', color='black', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Sigmoid Threshold')
    plt.title('Variation in sigmoid threshold values over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the learning rates during training over the course of each epoch
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Gender Classifier - Learning Rate Curve')
    plt.plot(epochs, lrs, 'o-', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Variation in learning rate over the course of each epoch')
    plt.xlim([1, len(epochs)])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
