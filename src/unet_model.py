"""PyTorch U-Net Architecture and ONNX Exporter for SAR Flood Detection.

Defines a lightweight Convolutional Neural Network (U-Net) capable of
understanding spatial context for flood segmentation, as recommended by
Tian et al. (2026).
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn

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
    
    Input: 5 Channels (NDWI, SAR_mask, Slope, VV, VH)
    Output: 1 Channel (Flood Probability)
    """
    def __init__(self, in_channels=5, out_channels=1):
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
    model = UNet(in_channels=5, out_channels=1)
    model.eval()
    
    # Create dummy input: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 5, 256, 256, requires_grad=True)
    
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


if __name__ == "__main__":
    export_to_onnx()
