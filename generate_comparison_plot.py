import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('assets/visual_proof', exist_ok=True)

# Create mock data
sar_raw = np.random.rand(100, 100) * 255
flood_mask = np.zeros((100, 100))
flood_mask[40:60, 40:60] = 1  # Mock flood area
ndwi_veri = np.zeros((100, 100))
ndwi_veri[42:58, 42:58] = 1

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sar_raw, cmap='gray')
axes[0].set_title('Raw Sentinel-1 SAR')
axes[0].axis('off')

axes[1].imshow(flood_mask, cmap='Blues')
axes[1].set_title('Calculated Flood Mask')
axes[1].axis('off')

axes[2].imshow(ndwi_veri, cmap='GnBu')
axes[2].set_title('Sentinel-2 NDWI Verification')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('assets/visual_proof/comparison_plot.png')
