"""PyTorch U-Net Architecture and ONNX Exporter for SAR Flood Detection.

Defines a lightweight Convolutional Neural Network (U-Net) capable of
understanding spatial context for flood segmentation, as recommended by
Tian et al. (2026), Li et al. (2023), and Jamali et al. (2024).

Training uses Dice Loss + BCE to handle severe class imbalance
(flood pixels ≈ 0.13% of total), per Milletari et al. (2016).
"""

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Lightweight U-Net for Flood Segmentation.
    
    Input: 6 Channels (NDWI, SAR_mask, Slope, VV, VH, HAND)
    Output: 1 Channel (Flood Probability)
    """
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) # 128 (upsampled) + 128 (skip)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        # Output layer
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder path with skip connections
        y = self.up1(x4)
        y = torch.cat([y, x3], dim=1)
        y = self.conv_up1(y)
        
        y = self.up2(y)
        y = torch.cat([y, x2], dim=1)
        y = self.conv_up2(y)
        
        y = self.up3(y)
        y = torch.cat([y, x1], dim=1)
        y = self.conv_up3(y)
        
        logits = self.outc(y)
        return torch.sigmoid(logits)


class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features.

    Per Oktay et al. (2018): "Attention U-Net: Learning Where to
    Look for the Pancreas" — 3,000+ citations.

    The attention gate learns to suppress irrelevant features in
    skip connections, focusing the model on flood-relevant regions.
    """
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        """
        gate: from coarser level (decoder)
        skip: from encoder (same level)
        """
        gate_conv = self.W_gate(gate)
        skip_conv = self.W_skip(skip)

        # Upsample gate to match skip spatial dimensions
        if gate_conv.shape[2:] != skip_conv.shape[2:]:
            gate_conv = nn.functional.interpolate(
                gate_conv, size=skip_conv.shape[2:], mode='bilinear', align_corners=True
            )

        attention = self.relu(gate_conv + skip_conv)
        attention = self.psi(attention)
        attention = torch.sigmoid(attention)

        return skip * attention


class AttentionUNet(nn.Module):
    """U-Net with Attention Gates for flood segmentation.

    Combines encoder-decoder architecture with attention mechanisms
    that learn to focus on flood-relevant spatial regions.

    Key improvement over standard U-Net:
    - Attention gates suppress irrelevant skip connection features
    - Better performance on small flood areas (< 1% of pixels)
    - More robust to noise and non-flood water bodies
    """
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Attention Gates (gate_channels, skip_channels, inter_channels)
        # After up1: y has 128 channels → after conv_up1: 128
        # After up2: y has 64 channels → after conv_up2: 64
        self.att1 = AttentionGate(256, 128, 64)  # gate=x4(256), skip=x3(128)
        self.att2 = AttentionGate(64, 64, 32)    # gate=y_after_up2(64), skip=x2(64)
        self.att3 = AttentionGate(32, 32, 16)    # gate=y_after_up3(32), skip=x1(32)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with attention
        y = self.up1(x4)
        x3_att = self.att1(x4, x3)
        y = torch.cat([y, x3_att], dim=1)
        y = self.conv_up1(y)

        y = self.up2(y)
        x2_att = self.att2(y, x2)
        y = torch.cat([y, x2_att], dim=1)
        y = self.conv_up2(y)

        y = self.up3(y)
        x1_att = self.att3(y, x1)
        y = torch.cat([y, x1_att], dim=1)
        y = self.conv_up3(y)

        return torch.sigmoid(self.outc(y))


class FPNHead(nn.Module):
    """Feature Pyramid Network head for multi-scale flood detection.

    Per Lin et al. (2017): "Feature Pyramid Networks for Object Detection"
    15,000+ citations. FPN creates multi-scale feature representations
    that can detect both large flood areas and small water bodies.

    Key advantage: detects floods at multiple scales simultaneously.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1)

    def forward(self, features):
        """
        features: list of feature maps at different scales
        Returns: fused multi-scale feature map
        """
        # Build top-down pathway
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = nn.functional.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=True
            )
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Resize all to same scale and concatenate
        target_size = outputs[0].shape[2:]
        resized = [nn.functional.interpolate(o, size=target_size, mode='bilinear', align_corners=True)
                   for o in outputs]

        fused = self.fusion(torch.cat(resized, dim=1))
        return fused


class FPNUNet(nn.Module):
    """U-Net with Feature Pyramid Network for multi-scale flood detection.

    Combines U-Net encoder-decoder with FPN for multi-scale feature fusion.
    Detects both large flood plains and small water bodies simultaneously.
    """
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # FPN
        self.fpn = FPNHead([32, 64, 128, 256], 64)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Multi-scale encoder features
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # FPN fusion
        fpn_out = self.fpn([e1, e2, e3, e4])

        # Decode
        decoded = self.decoder(fpn_out)

        return torch.sigmoid(self.outc(decoded))


class DiceBCELoss(nn.Module):
    """Combined Dice Loss + Binary Cross Entropy for imbalanced segmentation.

    Per Milletari et al. (2016): Dice Loss handles class imbalance by
    measuring overlap between prediction and ground truth, while BCE
    provides stable gradients.

    Loss = α * BCE + (1-α) * (1 - Dice)
    """
    def __init__(self, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        # Dice
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.dice_weight * bce_loss + (1.0 - self.dice_weight) * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in segmentation.

    Per Lin et al. (2017): "Focal Loss for Dense Object Detection"
    40,000+ citations. Reduces loss for well-classified examples,
    focusing training on hard negatives.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    alpha : float — Weighting factor for rare class (default 0.25)
    gamma : float — Focusing parameter (default 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

        # Binary cross entropy per pixel
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Focal weight: (1 - p_t)^gamma
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice Loss + Focal Loss for imbalanced segmentation.

    Combines the overlap sensitivity of Dice with the hard-example
    mining of Focal Loss. Best of both worlds for flood detection
    where flood pixels are < 1% of total area.

    Loss = β * Focal + (1-β) * (1 - Dice)
    """
    def __init__(self, focal_weight=0.5, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth = smooth

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)

        # Dice
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.focal_weight * focal_loss + (1.0 - self.focal_weight) * dice_loss


class FloodPatchDataset(Dataset):
    """Dataset for loading pre-saved .npy patches for U-Net training.

    Includes data augmentation (Shorten & Khoshgoftaar 2019):
    - Random horizontal/vertical flip
    - Random 90° rotation
    - Random Gaussian noise injection
    """
    def __init__(self, features_dir: str, labels_dir: str, augment: bool = True):
        self.features_dir = Path(features_dir)
        self.labels_dir = Path(labels_dir)
        self.patch_files = sorted([f.stem for f in self.features_dir.glob("*.npy")])
        self.augment = augment
        logger.info("FloodPatchDataset: %d patches from %s (augment=%s)",
                    len(self.patch_files), features_dir, augment)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        name = self.patch_files[idx]
        features = np.load(self.features_dir / f"{name}.npy").astype(np.float32)
        label = np.load(self.labels_dir / f"{name}.npy").astype(np.float32)

        if label.ndim == 2:
            label = label[np.newaxis, :, :]

        # Data augmentation
        if self.augment:
            features, label = self._augment(features, label)

        return torch.from_numpy(features), torch.from_numpy(label)

    def _augment(self, features, label):
        """Apply random augmentations to feature-label pair."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=2).copy()
            label = np.flip(label, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        # Random 90° rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            features = np.rot90(features, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(1, 2)).copy()

        # Random Gaussian noise on SAR bands (VV, VH only)
        if np.random.rand() > 0.5:
            noise = np.random.randn(*features[0:2].shape).astype(np.float32) * 0.5
            features[0:2] += noise

        return features, label


def train_unet(
    patches_dir: str = None,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    dice_weight: float = 0.5,
):
    """Train U-Net on pre-saved patches and export to ONNX.

    Parameters
    ----------
    patches_dir : str — Path to data/patches/ directory
    epochs : int — Number of training epochs
    batch_size : int — Batch size
    learning_rate : float — Learning rate
    dice_weight : float — Weight for BCE vs Dice loss (0.5 = equal)
    """
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    if patches_dir is None:
        patches_dir = str(PROJECT_ROOT / "data" / "patches")

    patches_path = Path(patches_dir)
    features_dir = patches_path / "features"
    labels_dir = patches_path / "labels"

    if not features_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Patches not found at {patches_path}. Run dataset_builder.py first."
        )

    logger.info("=" * 60)
    logger.info("U-NET TRAINING")
    logger.info("=" * 60)

    # Determine input channels from first patch
    sample_patch = np.load(next(features_dir.glob("*.npy")))
    in_channels = sample_patch.shape[0]
    logger.info("Input channels: %d", in_channels)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Dataset & DataLoader
    dataset = FloodPatchDataset(str(features_dir), str(labels_dir))

    # Split: 80% train, 20% val
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Train: %d patches, Val: %d patches", n_train, n_val)

    # Model
    model = UNet(in_channels=in_channels, out_channels=1).to(device)
    criterion = DiceBCELoss(dice_weight=dice_weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        logger.info("Epoch %3d/%d: train_loss=%.4f  val_loss=%.4f  lr=%.6f",
                    epoch + 1, epochs, train_loss, val_loss, optimizer.param_groups[0]['lr'])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            best_path = MODELS_DIR / "unet_best.pth"
            torch.save(model.state_dict(), str(best_path))

    # Load best and export to ONNX
    model.load_state_dict(torch.load(str(MODELS_DIR / "unet_best.pth"), weights_only=True))
    model.eval()

    onnx_path = MODELS_DIR / "unet_flood.onnx"
    dummy_input = torch.randn(1, in_channels, 256, 256, device=device)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    logger.info("ONNX exported: %s", onnx_path)
    logger.info("=" * 60)


def export_to_onnx():
    """Initializes a blank U-Net model and exports it to ONNX format.
    
    This acts as a placeholder generator. In production, this would be
    called AFTER training the model with the dataset_builder patches.
    """
    logger.info("=" * 40)
    logger.info("EXPORTING U-NET TO ONNX")
    logger.info("=" * 40)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS_DIR / "unet_flood.onnx"
    
    # Instantiate model
    model = UNet(in_channels=6, out_channels=1)
    model.eval()
    
    # Create dummy input: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 6, 256, 256, requires_grad=True)
    
    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        str(onnx_path),
        export_params=True,
        opset_version=11,          # Standard, stable opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},    # Allow variable batch sizes
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Successfully exported ONNX model to: {onnx_path}")
    logger.info("This ONNX file is ready to be embedded into the Rust PyO3 engine.")


def transfer_learning_finetune(
    pretrained_path: str,
    patches_dir: str = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    freeze_encoder: bool = True,
):
    """Fine-tune a pre-trained U-Net on NTB flood data.

    Per Tajbakhsh et al. (2016): "Convolutional Neural Networks for
    Medical Image Analysis: Full Training or Fine Tuning?" — 3,000+ citations.

    Transfer learning from a model trained on a larger flood dataset
    can improve performance by +10-15% F1 when local training data
    is limited.

    Parameters
    ----------
    pretrained_path : str — Path to pre-trained .pth file
    patches_dir : str — Path to local training patches
    epochs : int — Fine-tuning epochs
    batch_size : int — Batch size
    learning_rate : float — Lower LR for fine-tuning (default 1e-4)
    freeze_encoder : bool — If True, freeze encoder layers
    """
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    if patches_dir is None:
        patches_dir = str(PROJECT_ROOT / "data" / "patches")

    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING — FINE-TUNE")
    logger.info("=" * 60)

    # Load pre-trained model
    model = UNet(in_channels=5, out_channels=1)
    model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    logger.info("Loaded pre-trained: %s", pretrained_path)

    # Determine input channels from local data
    sample_patch = np.load(next(Path(patches_dir, "features").glob("*.npy")))
    in_channels = sample_patch.shape[0]

    if in_channels != 5:
        # Adapt first layer for different input channels
        old_inc = model.inc
        model.inc = DoubleConv(in_channels, 32)
        # Copy weights for first 5 channels
        with torch.no_grad():
            model.inc.double_conv[0].weight[:, :5] = old_inc.double_conv[0].weight
        logger.info("Adapted first layer: 5 → %d channels", in_channels)

    # Freeze encoder if requested
    if freeze_encoder:
        for name, param in model.named_parameters():
            if 'inc' in name or 'down' in name:
                param.requires_grad = False
        logger.info("Encoder frozen — only decoder will be trained")

    # Load local dataset
    dataset = FloodPatchDataset(
        str(Path(patches_dir) / "features"),
        str(Path(patches_dir) / "labels"),
        augment=True,
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Train: %d, Val: %d patches", n_train, n_val)

    # Fine-tune
    criterion = DiceFocalLoss(focal_weight=0.5, alpha=0.25, gamma=2.0)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        logger.info("Epoch %3d/%d: train=%.4f  val=%.4f  lr=%.6f",
                    epoch + 1, epochs, train_loss, val_loss, optimizer.param_groups[0]['lr'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(MODELS_DIR / "unet_finetuned.pth"))

    logger.info("Fine-tuning complete. Best val_loss=%.4f", best_val_loss)
    logger.info("=" * 60)


def quantize_onnx_int8(onnx_path: str = None, output_path: str = None):
    """Quantize ONNX model from FP32 to INT8 for edge deployment.

    Per Jacob et al. (2018): INT8 quantization reduces model size by ~4x
    and increases inference speed by ~2.5x with minimal accuracy loss.

    Compatible with ONNX Runtime on CPU, NVIDIA Jetson (TensorRT),
    and Raspberry Pi (ARM).

    Parameters
    ----------
    onnx_path : str — Path to input FP32 ONNX model
    output_path : str — Path for output INT8 ONNX model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        return None

    if onnx_path is None:
        onnx_path = str(MODELS_DIR / "unet_flood.onnx")
    if output_path is None:
        output_path = str(MODELS_DIR / "unet_flood_int8.onnx")

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger.info("=" * 40)
    logger.info("ONNX INT8 QUANTIZATION")
    logger.info("=" * 40)
    logger.info("Input: %s", onnx_path)

    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=QuantType.QInt8,
    )

    # Report size reduction
    orig_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    logger.info("Output: %s", output_path)
    logger.info("Size: %.2f MB → %.2f MB (%.1f%% reduction)", orig_size, quant_size, reduction)
    logger.info("=" * 40)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="U-Net for Flood Detection")
    parser.add_argument("--mode", choices=["train", "export", "quantize", "finetune", "attention", "fpn"],
                        default="export", help="Mode: train/export/quantize/finetune/attention/fpn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--onnx-path", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, help="Pre-trained model for fine-tuning")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder during fine-tuning")
    args = parser.parse_args()

    if args.mode == "train":
        train_unet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.mode == "quantize":
        quantize_onnx_int8(args.onnx_path)
    elif args.mode == "finetune":
        if not args.pretrained:
            print("Error: --pretrained required for finetune mode")
        else:
            transfer_learning_finetune(args.pretrained, epochs=args.epochs,
                                       batch_size=args.batch_size, learning_rate=args.lr,
                                       freeze_encoder=args.freeze_encoder)
    elif args.mode == "attention":
        # Export AttentionUNet to ONNX
        model = AttentionUNet(in_channels=5, out_channels=1)
        model.eval()
        dummy = torch.randn(1, 5, 256, 256)
        onnx_path = MODELS_DIR / "attention_unet.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), export_params=True, opset_version=11,
                          input_names=['input'], output_names=['output'])
        print(f"Exported: {onnx_path}")
    elif args.mode == "fpn":
        # Export FPNUNet to ONNX
        model = FPNUNet(in_channels=5, out_channels=1)
        model.eval()
        dummy = torch.randn(1, 5, 256, 256)
        onnx_path = MODELS_DIR / "fpn_unet.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), export_params=True, opset_version=11,
                          input_names=['input'], output_names=['output'])
        print(f"Exported: {onnx_path}")
    else:
        export_to_onnx()
