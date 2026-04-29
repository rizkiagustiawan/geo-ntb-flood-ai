# 🛰️ Sumbawa-A.E.C.O (Autonomous ESG Compliance Oracle) v1.1

> **Autonomous Geospatial Monitoring Station for Real-time Flood Detection in Sumbawa Island, NTB.**
> *Bilingual Documentation: 🇮🇩 Indonesian & 🇬🇧 English*

---

## 🇮🇩 Ringkasan (Indonesian)
**Sumbawa-A.E.C.O** adalah sistem monitoring banjir otonom yang menggabungkan kecanggihan **Multisensor Fusion** (Sentinel-1 SAR & Sentinel-2) dengan kecerdasan buatan. Sistem ini dirancang untuk mendeteksi genangan air secara real-time di Pulau Sumbawa dengan tingkat presisi tinggi, mengeliminasi *false positive* menggunakan **Terrain Awareness** (DEMNAS), dan beroperasi secara mandiri (Autonomous). Proyek ini membuktikan bahwa teknologi pemantauan canggih dapat dioptimalkan untuk berjalan pada perangkat keras standar tanpa mengurangi akurasi ilmiah.

---

## 🇬🇧 Executive Summary (English)
**A.E.C.O** is a production-ready MLOps pipeline designed for real-time flood monitoring. It bridges the gap between raw satellite telemetry and actionable ESG insights. By utilizing a hybrid stack of Python (Inference) and Rust (Autonomous Signaling), A.E.C.O provides high-fidelity metrics for disaster resilience and environmental compliance. The architecture is "Caveman-Optimized," ensuring high-performance geospatial processing (40M+ pixels) on consumer-grade hardware.

---

## 🚀 Key Technical Features
- **Multisensor Fusion:** Sentinel-1 (SAR) for cloud-penetrating radar + Sentinel-2 (MSI) for NDWI cross-verification.
- **Terrain & Ocean Awareness:** Automated masking using SRTM/DEMNAS to eliminate shadows and sea backscatter.
- **Autonomous Oracle:** Orchestrated via Rust-based `rtk` binary for zero-manual intervention.
- **Metric Precision:** Automated degree-to-meter compensation (EPSG:4326) for accurate Hectare calculations.
- **Code Integrity:** **25 Unit Tests passed ✅** using PyTest to ensure pipeline reliability.
- **Hardware Efficient:** Designed for stable execution on Intel i7-8550U class processors via windowed processing.

---

## 📊 Latest Performance & Results
- **Target Area:** Sumbawa Island, West Nusa Tenggara.
- **Estimated Impact:** **5,053.10 Hectares** (Latest Cycle).
- **Model Accuracy:** 99.85% (XGBoost F1-Score).
- **Processing Time:** ~4.5 minutes for 40M+ pixels (i7-8550U Optimized).

---

## 🛠️ Tech Stack
- **Engine:** Python 3.12+, Rust (Parallel Compute), XGBoost.
- **Geospatial:** Rasterio, GDAL, Google Earth Engine.
- **Interface:** FastAPI, Leaflet.js, CartoDB Dark Matter.
- **DevOps:** Docker, PyTest, GitHub Actions.

---

## 📂 Project Structure
```text
.
├── api/                # FastAPI logic & routes
├── data/
│   ├── raw/            # Raw .tif satellite tiles
│   └── processed/      # Feature stack (The Big 5)
├── outputs/
│   ├── models/         # Saved .pkl & .json metrics
│   └── predictions/    # final_flood_map.tif & previews
├── tests/              # PyTest units (25 tests passed)
├── flood_agent.py      # Main Autonomous Agent
├── Dockerfile          # Production deployment
├── docker-compose.yml  # Orchestration
└── LICENSE             # MIT License

Deployment & Usage
Option 1: Local Setup

    Clone & Setup Environment
    Bash

    git clone [https://github.com/rizki-agustiawan/geo-ntb-flood-ai.git](https://github.com/rizki-agustiawan/geo-ntb-flood-ai.git)
    cd geo-ntb-flood-ai
    python -m venv venv
    source venv/bin/activate  # or .fish / .bash
    pip install -r requirements.txt

    Run Autonomous Agent
    Bash

    python flood_agent.py

Option 2: Docker (Recommended)

    Set Environment Variables: Prepare your .env with GEE_KEY and BMKG_ENDPOINT.

    Launch System:
    Bash

    docker-compose up --build