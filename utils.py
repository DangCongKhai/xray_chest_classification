import torch
import torch.nn as nn
from typing import Literal
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
from PIL import Image

plt.style.use("ggplot")


class EarlyStopping:

    def __init__(
        self,
        model_checkpoint: str,
        patience=5,
        min_delta=1e-5,
        mode: Literal["min", "max"] = "max",
    ):
        """Early stopping to prevent model from overfitting

        Args:
            model_checkpoint (str): Model checkpoint path
            patience (int, optional): Number of times the model is allowed to continue training without performance improvement . Defaults to 5.
            min_delta (_type_, optional): Minimal improvement threshold. Defaults to 1e-5.
            mode (Literal[&quot;min&quot;, &quot;max&quot;], optional): Mode to optimize. Defaults to "max".
        """
        # Log model's performance
        self.best_score = None
        self.best_epoch = None
        self.model_checkpoint = model_checkpoint
        self.patience = patience
        self.counter = 0
        self.min_delta = min_delta
        self.mode = mode

    def should_stop(self, model, optimizer, epoch, score):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
            # Save model
            save_checkpoint(model, optimizer, self.model_checkpoint)
            return False, self.best_score, self.best_epoch
        if (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        ):
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0
            save_checkpoint(model, optimizer, self.model_checkpoint)
            return False, self.best_score, self.best_epoch
        else:
            self.counter += 1
            return self.counter >= self.patience, self.best_score, self.best_epoch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    loop = tqdm(dataloader, desc="Training")

    for _, images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_probs_real = []

    with torch.no_grad():
        for _, images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs_real.extend(probs[:, 1].cpu().numpy())

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# NOTE. Get device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print("Checkpoint saved:", filename)


# NOTE: Get all predictions
def get_all_predictions(model, loader, device):
    """
    Runs inference on the entire loader and returns all true labels and predictions.
    """
    model.eval()
    y_pred = []
    y_true = []
    originals = []
    with torch.no_grad():
        for original_batch, images, labels in tqdm(loader, desc="Getting predictions"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            originals.extend(original_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    return originals, y_true, y_pred


def plot_history(history, title, save_path=None, start_finetuned_epoch=None):
    epochs = range(1, len(history["train_losses"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    plt.suptitle(title)

    ax[0].plot(
        epochs,
        history["train_losses"],
        label="training loss",
        marker="*",
        color="green",
    )
    ax[0].plot(
        epochs, history["val_losses"], label="validation loss", marker="s", color="blue"
    )
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    if start_finetuned_epoch is not None:
        ax[0].axvline(
            x=start_finetuned_epoch,
            color="red",
            linestyle="--",
            label="Start Fine-Tuning",
        )
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.7)

    ax[1].plot(
        epochs,
        history["train_accuracy"],
        label="training accuracy",
        marker="*",
        color="green",
    )
    ax[1].plot(
        epochs,
        history["val_accuracy"],
        label="validation accuracy",
        marker="s",
        color="blue",
    )
    ax[1].set_title("Accuracy Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    if start_finetuned_epoch is not None:
        ax[1].axvline(
            x=start_finetuned_epoch,
            color="red",
            linestyle="--",
            label="Start Fine-Tuning",
        )
    ax[1].grid(True, linestyle="--", alpha=0.7)

    if save_path is not None:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def performance(model, loader, device, model_name, class_names, save_path=None):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

    _, y_true, y_pred = get_all_predictions(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy score: {acc:.4f}")

    print("Classification Report")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    if save_path is not None:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def save_experiment_json(history, filename):
    """
    Saves the experiment history to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {filename}")


def get_last_conv_layer(model):
    last_conv_layer = None
    # Iterate through all modules in the model
    for module in model.modules():
        # Check if the module is a convolutional layer (Conv2d, Conv3d, etc.)
        if isinstance(
            module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            last_conv_layer = module
    return last_conv_layer


def process_image_pytorch(image_path, transform=None):
    import torch.nn as nn
    from torchvision import transforms

    """Process image for PyTorch CNN model"""

    img = Image.open(image_path).convert("RGB")
    if transform is not None:
        img = transform(img)
    else:
        img = np.array(img)
    return img.unsqueeze(0)


def get_prediction_pytorch(model, image):
    """Get prediction from PyTorch model"""
    import torch

    CLASSES = ["COVID", "NORMAL", "PNEUMONIA"]
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probability = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probability, dim=1).tolist()[0]
        confidence = probability[:, prediction][0]
    return CLASSES[prediction], confidence


def make_gradcam_heatmap(model, target_layer, img_tensor, pred_index=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Storage for gradients and activations
    gradients = []
    activations = []

    # Define hook functions
    def backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple, we want the first element
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks to the target layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # 1. Forward Pass
    raw_logits = model(img_tensor)
    output = nn.Softmax(dim=1)(raw_logits)

    # 2. Get Class Score
    if pred_index is None:
        pred_index = torch.argmax(output)
    class_score = output[:, pred_index]

    # 3. Backward Pass
    model.zero_grad()
    class_score.backward()

    # --- Grad-CAM Logic ---

    # Get the gradients and activations (and remove hooks)
    grads = gradients[0]  # (B, C, H, W) e.g., (1, 512, 7, 7)
    acts = activations[0]  # (B, C, H, W) e.g., (1, 512, 7, 7)
    forward_handle.remove()
    backward_handle.remove()

    # 4. Compute alpha weights (Step 4)
    # Global Average Pooling on gradients
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])  # (C,)

    # 5. Create Heatmap (Step 5)
    # Get activations for the specific image in batch (dim 0)
    acts = acts[0]  # (C, H, W)
    # Weighted sum: (C, H, W) * (C,) -> needs broadcasting
    for i in range(acts.shape[0]):
        acts[i, :, :] *= pooled_grads[i]  # In-place multiplication

    # Sum across the channel dimension to get the raw heatmap
    heatmap = torch.sum(acts, dim=0)

    # 6. ReLU (Step 6)
    heatmap = F.relu(heatmap)

    # Normalize
    heatmap /= torch.max(heatmap) + 1e-10  # Add epsilon to avoid div by zero

    return heatmap.cpu().detach().numpy()


def get_gradcam(img, heatmap, alpha=0.4):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Upsample heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB (using 'jet' colormap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay the heatmap on the original image (alpha blending)
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img
