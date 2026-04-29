#!/bin/bash
# ============================================================
# 🛰️  Sumbawa Flood AI — Full Pipeline Launcher (Final Version)
# ============================================================
set -e

# 1. Setup Path & Environment
# Masuk ke folder root (geo-ntb-flood-ai)
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/api"

# 2. Aktivasi Virtual Environment
if [ -d "../venv" ]; then
    echo "🐍 Activating Virtual Environment: $(realpath ../venv)"
    source ../venv/bin/activate
else
    echo "⚠️  Warning: venv not found at ../venv. Using system python."
fi

echo "============================================================"
echo "🛰️  STAGE 1: Data Ingestion (GEE + BMKG)"
echo "============================================================"
# SKIPPED: Karena lu download manual (Limitasi GEE 50MB)
echo "⏩ Ingestion skipped. Using manual download tiles in data/raw/"
# python src/ingest.py "$@"

echo "============================================================"
echo "🔧 STAGE 2: Preprocessing (CRS + Resample + Tile)"
echo "============================================================"
python src/preprocess.py

echo "============================================================"
echo "🧬 STAGE 3: Feature Engineering (NDWI + SAR + Slope)"
echo "============================================================"
python src/features.py

echo "============================================================"
echo "🧠 STAGE 4: Model Training (XGBoost)"
echo "============================================================"
python src/model.py

echo "============================================================"
echo "🗺️  STAGE 5: Prediction"
echo "============================================================"
python src/predict.py

echo "============================================================"
echo "🌊 STAGE 6: Postprocess (Ocean Masking)"
echo "============================================================"
python src/postprocess.py

echo "============================================================"
echo "📊 STAGE 7: Evaluation"
echo "============================================================"
python src/evaluate.py || echo "⚠️  Evaluation skipped (no labels found)"

echo "============================================================"
echo "🖼️  STAGE 8: Visualization & Optimization"
echo "============================================================"
# Render preview image
python src/visualize.py --input outputs/predictions/final_flood_map.tif --output outputs/predictions/final_flood_map_preview.png

# --- OPTIMASI ANTI-FREEZE (Wajib buat i7-8550U) ---
echo "⚡ Building Image Pyramids (gdaladdo)..."
if [ -f "outputs/predictions/final_flood_map.tif" ]; then
    gdaladdo -r average outputs/predictions/final_flood_map.tif 2 4 8 16 32 64
    echo "✅ Optimization Complete: Map is now smooth like butter."
else
    echo "❌ Error: final_flood_map.tif not found!"
fi

echo "============================================================"
echo "✅ PIPELINE COMPLETE — SYSTEM READY"
echo "============================================================"
echo ""
echo "Untuk menjalankan Dashboard, buka terminal baru (Konsole) dan ketik:"
echo "  source ../venv/bin/activate.fish  # Karena lu pakai Fish"
echo "  cd geo-ntb-flood-ai"
echo "  python api/main.py"
echo ""
echo "Happy Monitoring, Awan! 🚀🛰️"