# Panduan Manual Digitization untuk Ground Truth Banjir NTB

## 1. Kapan Banjir Terjadi di NTB?

### Sumber Informasi Historis Banjir NTB

| Tanggal | Lokasi | Sumber |
|---------|--------|--------|
| **Feb 2023** | Taliwang, Sumbawa Barat | BNPB, berita lokal |
| **Jan 2024** | Bima, Dompu | BNPB, media |
| **Feb 2025** | Taliwang, Sumbawa Barat | BNPB, media |
| **Mar 2025** | Lombok Tengah | Media lokal |
| **Jan 2026** | Bima, Sumbawa | Media lokal |

### Cara Cari Tanggal Banjir Pasti

1. **BNPB DIBI** (https://dibi.bnpb.go.id) — Database bencana Indonesia
2. **Media lokal** — radarlombok.co.id, tribunlombok.com, lombokpost.co.id
3. **Copernicus EMS** (https://rapidmapping.emergency.copernicus.eu) — cari "Indonesia"
4. **Twitter/X** — search "banjir NTB" + tahun

---

## 2. Cara Manual Digitization

### Metode A: Dari PlanetScope NICFI (Gratis untuk Tropis)

1. **Daftar** di https://www.planet.com/nicfi/
2. **Login** ke Planet Explorer
3. **Cari** area NTB pada tanggal banjir
4. **Download** imagery resolusi 4.7m
5. **Buka** di QGIS
6. **Digitize** polygon banjir secara manual

### Metode B: Dari Sentinel-2 (Gratis)

1. **Buka** Copernicus Browser (https://browser.dataspace.copernicus.eu)
2. **Cari** area NTB pada tanggal banjir
3. **Download** Sentinel-2 L2A (Band 3, 8 untuk NDWI)
4. **Buka** di QGIS
5. **Compute** NDWI = (B3 - B8) / (B3 + B8)
6. **Threshold** NDWI > 0.3 = air
7. **Edit** manual: hapus awan, tambah area yang terlewat

### Metode C: Dari Google Earth Pro (Gratis)

1. **Download** Google Earth Pro
2. **Navigate** ke lokasi banjir
3. **Historical Imagery** — cari tanggal banjir
4. **Draw** polygon banjir
5. **Export** sebagai KML
6. **Convert** KML → GeoTIFF di QGIS

---

## 3. Tools untuk Digitization

### QGIS (Gratis, Open Source)
```bash
# Install
sudo apt install qgis

# Workflow:
1. Open QGIS
2. Load satellite imagery (GeoTIFF)
3. Create New Shapefile (Polygon)
4. Draw polygons over flood areas
5. Save as GeoTIFF (Rasterize)
```

### GeoJSON Editor Online
- https://geojson.io — Draw polygons langsung di browser
- Export sebagai GeoJSON
- Convert ke GeoTIFF dengan GDAL

### Python Script untuk Semi-Automated
```python
# Semi-automated: NDWI + manual cleanup
import rasterio, numpy as np

# Load Sentinel-2
with rasterio.open('sentinel2.tif') as src:
    green = src.read(1)  # B03
    nir = src.read(2)    # B08

# Compute NDWI
ndwi = (green - nir) / (green + nir)

# Auto-detect water
water = (ndwi > 0.3).astype(np.uint8)

# Save for manual editing in QGIS
with rasterio.open('water_auto.tif', 'w', **profile) as dst:
    dst.write(water, 1)

# Manual: buka di QGIS, edit polygon, save
```

---

## 4. Format Ground Truth yang Dibutuhkan

```
File: data/labels/ground_truth_banjir.tif
Format: GeoTIFF
CRS: EPSG:4326 (atau sama dengan SAR)
Resolusi: sama dengan SAR (30m untuk S1)
Values: 0=non-flood, 1=flood, 255=nodata
```

### Cara Buat dari Polygon
```python
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

# Load polygon (dari QGIS/GeoJSON)
gdf = gpd.read_file('flood_polygons.geojson')

# Load SAR untuk reference grid
with rasterio.open('sentinel1_reproj.tif') as src:
    profile = src.profile.copy()
    transform = src.transform
    out_shape = src.shape

# Rasterize polygon ke grid SAR
shapes = [(geom, 1) for geom in gdf.geometry]
flood_mask = rasterize(shapes, out_shape=out_shape, transform=transform)

# Save
profile.update(count=1, dtype='uint8', nodata=255)
with rasterio.open('data/labels/ground_truth_banjir.tif', 'w', **profile) as dst:
    dst.write(flood_mask, 1)
```

---

## 5. Prioritas Digitization

### Tier 1 (Paling Mudah)
1. **Taliwang Feb 2023** — Banyak berita, area kecil
2. **Bima Jan 2024** — Sungai Bima, mudah identifikasi

### Tier 2 (Medium)
3. **Lombok Tengah Mar 2025** — Area luas
4. **Sumbawa Feb 2025** — Taliwang lagi

### Tier 3 (Sulit)
5. **Seluruh NTB** — Multi-event composite

---

## 6. Validasi Ground Truth

Setelah digitize, validasi dengan:
1. **Cross-check** berita/media untuk konfirmasi tanggal
2. **Visual check** di Google Earth Pro historical imagery
3. **Compare** dengan SAR backscatter (banjir harus gelap di SAR)
4. **Multiple annotators** — minta 2-3 orang digitize area yang sama

---

## 7. Quick Start: Taliwang Feb 2023

```python
# 1. Download Sentinel-2 untuk Taliwang, Feb 2023
# Copernicus Browser → search "Taliwang" → date Feb 2023

# 2. Download Sentinel-1 untuk tanggal yang sama
# ASF Search → POLYGON((116.7 -8.8, 116.9 -8.8, 116.9 -8.6, 116.7 -8.6, 116.7 -8.8))

# 3. Process S2 → NDWI
# 4. Process S1 → SAR mask
# 5. Manual cleanup di QGIS
# 6. Save sebagai ground truth
```
