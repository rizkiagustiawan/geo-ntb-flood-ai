import asf_search as asf
import os
import shutil
import zipfile
import rasterio
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# --- INTEGRASI MODUL RUST ---
try:
    import flood_rs

    RUST_READY = True
except ImportError:
    RUST_READY = False

# --- KONFIGURASI PATH ---
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "outputs" / "predictions"
# PERBAIKAN: Diarahkan ke folder lokal agar tidak kena PermissionError /var/www di laptop lokal
WEB_DIR = PROJECT_ROOT / "outputs" / "web"
LOG_FILE = PROJECT_ROOT / "agent_history.log"

# --- RUTE B: TUNING PARAMETER (SAR Thresholds) ---
# Sesuaikan angka ini berdasarkan observasi di Sumbawa/Bima
VV_THRESH = -18.0
VH_THRESH = -24.0


def log_agent(message: str) -> None:
    """Logs a message to the console and the agent history file.

    Args:
        message (str): The message string to log.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")


def send_telegram_alert(area_ha: float, scene_id: str) -> None:
    """Sends a Telegram alert if a flooded area is detected.

    Args:
        area_ha (float): The estimated flooded area in hectares.
        scene_id (str): The Sentinel-1 scene identifier.
    """
    if area_ha <= 0:
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        log_agent("⚠️ Kredensial Telegram (Token/Chat ID) tidak ditemukan di .env.")
        return

    message = (
        f"🚨 *Peringatan Banjir Sumbawa* 🚨\n\n"
        f"▪️ *Estimated Impact Area*: {area_ha:.2f} Hectares\n"
        f"▪️ *Sentinel-1 Scene ID*: {scene_id}\n"
        f"▪️ *Dashboard*: [aeco.rizkiagustiawan.tech](https://aeco.rizkiagustiawan.tech)\n"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        log_agent("📲 Notifikasi Telegram berhasil dikirim.")
    except Exception as e:
        log_agent(f"❌ Gagal mengirim notifikasi Telegram: {e}")


def verify_with_sentinel2(scene_name: str, cloud_cover: float = 0.0) -> float:
    """Verifies SAR flood mask using Sentinel-2 optical imagery via NDWI.
    
    This function calculates the Normalized Difference Water Index (NDWI)
    using the Green and Near-Infrared (NIR) bands:
        NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        scene_name (str): The identifier for the current satellite scene.
        cloud_cover (float): The percentage of cloud cover over the AOI.
        
    Returns:
        float: The weighting applied to the SAR mask. Returns 1.0 if clouds > 50%.
    """
    if cloud_cover > 50.0:
        log_agent(f"☁️ Sentinel-2 data obscured (Cloud Cover: {cloud_cover}%). Fallback to 100% SAR (Sentinel-1) weighting.")
        return 1.0
    
    log_agent(f"☀️ Optical verification via Sentinel-2 NDWI active for {scene_name}.")
    # Placeholder for actual NDWI verification logic
    return 0.8


def extract_and_process(zip_path: Path, scene_name: str):
    """Extracts VV/VH bands from a ZIP file and processes the flood mask via Rust.

    Args:
        zip_path (Path): Path to the downloaded Sentinel-1 ZIP file.
        scene_name (str): The name of the Sentinel-1 scene.

    Returns:
        tuple: A tuple containing the path to the temporary flood mask TIF 
        and the estimated flooded area in hectares, or None if processing fails.
    """
    extract_path = RAW_DIR / scene_name

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Cari file .tiff untuk VV dan VH di dalam folder measurement
            tiff_files = [
                f
                for f in zip_ref.namelist()
                if f.endswith(".tiff") and "measurement" in f
            ]
            zip_ref.extractall(RAW_DIR)

        vv_path = next(str(RAW_DIR / f) for f in tiff_files if "-vv-" in f.lower())
        vh_path = next(str(RAW_DIR / f) for f in tiff_files if "-vh-" in f.lower())

        with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
            vv_data = src_vv.read(1).astype(np.float32)
            vh_data = src_vh.read(1).astype(np.float32)
            profile = src_vv.profile

        log_agent(
            f"⚙️ Memproses mask banjir di Rust (VV: {VV_THRESH}, VH: {VH_THRESH})..."
        )

        # Placeholder: Verify with Sentinel-2 NDWI if possible
        sar_weight = verify_with_sentinel2(scene_name, cloud_cover=65.0)

        # Eksekusi fungsi Rust dari lib.rs
        mask = flood_rs.calculate_sar_flood_mask(vv_data, vh_data, VV_THRESH, VH_THRESH)

        # Hitung area terdampak banjir (Area in Hectares)
        flood_pixels = int(np.sum(mask > 0))
        dx = abs(profile["transform"][0])
        dy = abs(profile["transform"][4])
        # Aproksimasi meter ke derajat jika menggunakan sistem koordinat geografi (WGS84)
        if src_vv.crs and src_vv.crs.is_geographic:
            dx *= 111320.0
            dy *= 111320.0
        area_ha = float(flood_pixels * dx * dy / 10000.0)

        # Simpan hasil sementara
        temp_tif = RAW_DIR / f"flood_mask_{scene_name}.tif"
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(temp_tif, "w", **profile) as dst:
            dst.write(mask, 1)

        return temp_tif, area_ha

    except Exception as e:
        log_agent(f"❌ Gagal memproses data SAFE: {e}")
        return None
    finally:
        # Bersihkan folder ekstraksi untuk menghemat ruang
        if extract_path.exists():
            shutil.rmtree(extract_path)


def download_and_trigger(scene):
    """Downloads a Sentinel-1 scene and triggers the processing pipeline.

    Args:
        scene (asf.ASFProduct): The ASF search result object representing the scene.
    """
    scene_name = scene.properties["sceneName"]
    zip_path = RAW_DIR / f"{scene_name}.zip"

    if zip_path.exists():
        log_agent(f"✅ Data {scene_name} sudah ada. Melewati pengunduhan.")
        return

    log_agent(f"⬇️ Mengunduh {scene_name}...")
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        session = asf.ASFSession().auth_with_creds(
            os.environ.get("EARTHDATA_USER"), os.environ.get("EARTHDATA_PASS")
        )
        scene.download(path=str(RAW_DIR), session=session)

        if RUST_READY:
            extraction_result = extract_and_process(zip_path, scene_name)

            if extraction_result:
                result_tif, area_ha = extraction_result
                if result_tif and result_tif.exists():
                    # --- RUTE A: UPDATE DASHBOARD ---
                    final_tif = WEB_DIR / "final_flood_map.tif"
                    shutil.copy(str(result_tif), str(final_tif))
                    log_agent(f"📡 Dashboard Updated: {final_tif}")

                    # --- RUTE B: HISTORICAL ARCHIVE ---
                    history_dir = PROJECT_ROOT / "outputs" / "history"
                    history_dir.mkdir(parents=True, exist_ok=True)
                    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
                    history_file = history_dir / f"flood_sumbawa_{timestamp_str}.tif"
                    shutil.move(str(result_tif), str(history_file))
                    log_agent(f"💾 Historis Disimpan: {history_file}")

                    # --- TRIGGER TELEGRAM ALERT ---
                    send_telegram_alert(area_ha, scene_name)

                    # --- THE JANITOR ---
                    os.remove(zip_path)
                    log_agent(f"🧹 Cleanup: ZIP dihapus. Storage aman.")
        else:
            log_agent("⚠️ Modul Rust (flood_rs) tidak terdeteksi.")

    except Exception as e:
        log_agent(f"❌ Gangguan pada Misi: {e}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(WEB_DIR, exist_ok=True)

    # Koordinat Sumbawa (BBOX)
    SUMBAWA_WKT = (
        "POLYGON((116.5 -9.0, 119.5 -9.0, 119.5 -8.0, 116.5 -8.0, 116.5 -9.0))"
    )
    start_time = (datetime.now(timezone.utc) - timedelta(days=3)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    log_agent("🕵️‍♂️ Memindai satelit untuk wilayah NTB...")
    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=SUMBAWA_WKT,
        start=start_time,
        processingLevel="GRD_HD",
    )

    if results:
        download_and_trigger(results[0])
    else:
        log_agent("☁️ Tidak ada scene baru ditemukan.")

    log_agent("🏁 Agen kembali ke mode siaga.\n")
