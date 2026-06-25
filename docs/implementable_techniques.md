# Teknik dari Riset Paper yang Bisa Diimplementasikan ke A.E.C.O

## Prioritas 1: Langsung Naikkan F1 (High Impact, Low Effort)

### 1. GLCM Texture Features
**Paper:** Haralick et al. (1973), 12,000+ citations
**Efek:** +5-10% F1
**Cara:** Tambah 4 fitur tekstur (contrast, correlation, energy, homogeneity) ke feature stack
```python
from skimage.feature import graycomatrix, graycoprops
# 5 band → 9 band (5 original + 4 texture)
```

### 2. Focal Loss untuk Class Imbalance
**Paper:** Lin et al. (2017), 40,000+ citations
**Efek:** +3-5% F1 pada imbalanced data
**Cara:** Ganti BCE+Focal di U-Net loss function
```python
focal_loss = -alpha * (1-pt)^gamma * log(pt)
```

### 3. Multi-temporal Change Detection
**Paper:** Clement et al. (2018), 503 citations
**Efek:** Bedakan permanent water vs flood
**Cara:** Bandingkan S1 baseline (kering) vs S1 saat banjir
```python
delta_vv = vv_wet - vv_dry  # Negative = flood
```

### 4. MNDWI (Modified NDWI)
**Paper:** Xu (2006), 7,594 citations
**Efek:** Lebih baik dari NDWI untuk air keruh
**Cara:** Ganti NDWI dengan MNDWI = (Green - SWIR) / (Green + SWIR)

### 5. Transfer Learning dari Pre-trained U-Net
**Paper:** Tajbakhsh et al. (2016), 3,000+ citations
**Efek:** +10-15% F1 dengan data kecil
**Cara:** Fine-tune model yang sudah trained pada dataset flood global

---

## Prioritas 2: Improve Existing (Medium Impact)

### 6. Connected Component Analysis (Sudah ada, bisa improve)
**Paper:** Haralock & Shapiro, 5,000+ citations
**Efek:** Kurangi false positive
**Cara:** Filter komponen < 50 pixels (sudah ada, tapi bisa tuning)

### 7. Morphological Closing + Opening (Sudah ada)
**Paper:** Soille (2003), 8,000+ citations
**Efek:** Bersihkan noise
**Cara:** Sudah diimplementasikan di auto_digitize.py

### 8. Spatial Cross-Validation (Sudah ada)
**Paper:** Roberts et al. (2017), 1,500+ citations
**Efek:** Evaluasi lebih valid
**Cara:** Sudah diimplementasikan di model.py

### 9. Ensemble dengan Stacking
**Paper:** Wolpert (1992), 5,000+ citations
**Efek:** +2-3% F1 dari majority voting
**Cara:** Meta-learner di atas RF+XGB+LGBM predictions

### 10. SAR Texture Features (GLCM pada VV/VH)
**Paper:** Dekker (2003), 1,000+ citations
**Efek:** Deteksi flooded vegetation lebih baik
**Cara:** Tambah GLCM contrast dan homogeneity dari VV band

---

## Prioritas 3: Advanced (High Impact, High Effort)

### 11. Attention Mechanism di U-Net
**Paper:** Oktay et al. (2018), 3,000+ citations
**Efek:** Focus pada area penting
**Cara:** Tambah attention gate di decoder

### 12. Multi-scale Feature Fusion (FPN)
**Paper:** Lin et al. (2017), 15,000+ citations
**Efek:** Deteksi banjir kecil dan besar
**Cara:** Feature Pyramid Network di U-Net

### 13. Data Augmentation untuk SAR
**Paper:** Shorten & Khoshgoftaar (2019), 5,000+ citations
**Efek:** Kurangi overfitting
**Cara:** Flip, rotate, noise injection pada patches

### 14. Semi-supervised Learning
**Paper:** Berthelot et al (2019), 2,000+ citations
**Efek:** Manfaatkan data tanpa label
**Cara:** MixMatch pada data SAR tanpa ground truth

### 15. Active Learning untuk Annotation
**Paper:** Settles (2009), 5,000+ citations
**Efek:** Kurangi effort labeling
**Cara:** Pilih pixel paling informatif untuk di-annotate

---

## Implementasi yang Sudah Selesai

| # | Teknik | Status | Paper Referensi |
|---|--------|--------|-----------------|
| 1 | Refined Lee Speckle Filter | ✓ | Lee (1986), Twele (2016) |
| 2 | Otsu Adaptive Thresholding | ✓ | Otsu (1979), Tran (2022) |
| 3 | AND Fusion Logic | ✓ | Twele (2016) |
| 4 | HAND (Height Above Nearest Drainage) | ✓ | Nobre (2011), Tian (2026) |
| 5 | Spatial Cross-Validation | ✓ | Roberts (2017) |
| 6 | Ensemble ML (RF+XGB+LGBM) | ✓ | Chen (2016), Ke (2017) |
| 7 | Dice Loss + BCE | ✓ | Milletari (2016) |
| 8 | Morphological Post-Processing | ✓ | Soille (2003) |
| 9 | Connected Component Analysis | ✓ | Haralock & Shapiro |
| 10 | Multi-temporal Change Detection | ✓ | Clement (2018), Schlaffer (2015) |
| 11 | SHAP Explainability | ✓ | Lundberg & Lee (2017) |
| 12 | ONNX Edge Deployment | ✓ | Jacob et al. (2018) |
