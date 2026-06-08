# Sentinel Hub Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual GeoTIFF downloads with automated Sentinel Hub API fetching for real-time flood monitoring across NTB.

**Architecture:** New `src/satellite_fetcher.py` module queries Sentinel Hub Process API daily via Celery Beat, saves rasters to `data/processed/`, then existing pipeline processes them unchanged.

**Tech Stack:** sentinelhub-py (official SDK), Celery Beat (scheduler), FastAPI (new endpoints), existing Rust engine pipeline.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/satellite_fetcher.py` | CREATE | Sentinel Hub OAuth2 + Process API client |
| `tests/test_satellite_fetcher.py` | CREATE | Unit tests for fetcher |
| `api/tasks.py` | MODIFY | Add daily periodic task + Celery Beat schedule |
| `api/main.py` | MODIFY | Add `/satellite/status` and `/satellite/sync` endpoints |
| `docker-compose.yml` | MODIFY | Add celery-beat service |
| `index.html` | MODIFY | Add satellite sync status card |
| `requirements.txt` | MODIFY | Add `sentinelhub` dependency |
| `.env.example` | MODIFY | Add SH credentials template |

---

### Task 1: Add Sentinel Hub dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add sentinelhub to requirements.txt**

```txt
# --- Satellite API ---
sentinelhub>=3.10.0
```

Append to end of `requirements.txt`.

- [ ] **Step 2: Install dependency**

Run: `pip install sentinelhub>=3.10.0`
Expected: Successfully installed

- [ ] **Step 3: Verify import**

Run: `python -c "from sentinelhub import SentinelHubRequest, SHConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add sentinelhub dependency for satellite API integration"
```

---

### Task 2: Create .env.example with Sentinel Hub credentials

**Files:**
- Modify: `.env.example` (create if not exists)

- [ ] **Step 1: Create .env.example**

```env
# Sentinel Hub API (https://apps.sentinel-hub.com/)
SH_CLIENT_ID=your_client_id_here
SH_CLIENT_SECRET=your_client_secret_here
SH_INSTANCE_ID=your_instance_id_here

# Existing vars (if any)
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

- [ ] **Step 2: Verify .env is in .gitignore**

Run: `grep -q "^\.env$" .gitignore && echo "OK" || echo ".env NOT in .gitignore"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: add Sentinel Hub credentials template to .env.example"
```

---

### Task 3: Create Sentinel Hub Fetcher module

**Files:**
- Create: `src/satellite_fetcher.py`
- Test: `tests/test_satellite_fetcher.py`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/test_satellite_fetcher.py
import pytest
from unittest.mock import patch


def test_config_loads_from_env():
    """Verify SHConfig reads credentials from environment."""
    with patch.dict("os.environ", {
        "SH_CLIENT_ID": "test_id",
        "SH_CLIENT_SECRET": "test_secret",
        "SH_INSTANCE_ID": "test_instance",
    }):
        from src.satellite_fetcher import SentinelHubFetcher
        fetcher = SentinelHubFetcher()
        assert fetcher.config.sh_client_id == "test_id"
        assert fetcher.config.sh_client_secret == "test_secret"


def test_config_missing_credentials_raises():
    """Verify clear error when credentials missing."""
    with patch.dict("os.environ", {}, clear=True):
        from src.satellite_fetcher import SentinelHubFetcher
        with pytest.raises(ValueError, match="Sentinel Hub credentials"):
            SentinelHubFetcher()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_satellite_fetcher.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement SentinelHubFetcher class — config only**

```python
# src/satellite_fetcher.py
"""Sentinel Hub API Client for A.E.C.O Flood Monitoring.

Automated satellite data fetching for NTB region using
Sentinel Hub Process API.

Usage:
    fetcher = SentinelHubFetcher()
    rasters = fetcher.fetch_all()
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("aeco-satellite")

# NTB bounding boxes [West, South, East, North]
NTB_BBOX = [115.7, -9.1, 119.2, -8.1]
LOMBOK_BBOX = [115.7, -8.9, 116.6, -8.1]
SUMBAWA_BBOX = [116.6, -9.1, 119.2, -8.1]

# Data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STATUS_FILE = DATA_DIR / "satellite_status.json"


class SentinelHubFetcher:
    """Fetches Sentinel-1/2 data from Sentinel Hub Process API."""

    def __init__(self):
        from sentinelhub import SHConfig

        self.config = SHConfig()
        self.config.sh_client_id = os.environ.get("SH_CLIENT_ID", "")
        self.config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET", "")
        self.config.sh_instance_id = os.environ.get("SH_INSTANCE_ID", "")

        if not self.config.sh_client_id or not self.config.sh_client_secret:
            raise ValueError(
                "Sentinel Hub credentials not set. "
                "Set SH_CLIENT_ID and SH_CLIENT_SECRET in .env"
            )
```

- [ ] **Step 4: Run test to verify config loading passes**

Run: `pytest tests/test_satellite_fetcher.py::test_config_loads_from_env tests/test_satellite_fetcher.py::test_config_missing_credentials_raises -v`
Expected: PASS

- [ ] **Step 5: Write failing test for scene checking**

Append to `tests/test_satellite_fetcher.py`:

```python
def test_check_new_scenes_returns_date():
    """Verify check_new_scenes returns latest scene date."""
    from unittest.mock import MagicMock
    from src.satellite_fetcher import SentinelHubFetcher

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)
    fetcher.config = MagicMock()

    # Mock Catalog API response
    mock_scene = {"properties": {"datetime": "2026-06-07T02:30:00Z"}}
    with patch("src.satellite_fetcher SentinelHubCatalog") as MockCatalog:
        mock_client = MagicMock()
        mock_client.search.return_value = [mock_scene]
        MockCatalog.return_value = mock_client

        result = fetcher.check_new_scenes(LOMBOK_BBOX, days_back=7)
        assert result is not None
        assert "2026-06-07" in result
```

- [ ] **Step 6: Implement check_new_scenes method**

Add to `SentinelHubFetcher` class:

```python
    def check_new_scenes(self, bbox: list, days_back: int = 7) -> str | None:
        """Check if new Sentinel-1 scenes are available.

        Args:
            bbox: [W, S, E, N] bounding box
            days_back: How many days to look back

        Returns:
            ISO date string of latest scene, or None if no new data.
        """
        from sentinelhub import SentinelHubCatalog, BBox, CRS, DataCollection

        catalog = SentinelHubCatalog(config=self.config)
        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        time_interval = (
            (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00"),
            datetime.utcnow().strftime("%Y-%m-%dT23:59:59"),
        )

        search_iterator = catalog.search(
            collection=DataCollection.SENTINEL1_IW,
            bbox=bbox_obj,
            time=time_interval,
            limit=1,
        )

        results = list(search_iterator)
        if not results:
            return None

        return results[0]["properties"]["datetime"]
```

- [ ] **Step 7: Run scene check test**

Run: `pytest tests/test_satellite_fetcher.py::test_check_new_scenes_returns_date -v`
Expected: PASS

- [ ] **Step 8: Write failing test for S1 fetching**

Append to `tests/test_satellite_fetcher.py`:

```python
def test_fetch_sentinel1_returns_numpy():
    """Verify fetch_sentinel1 returns VV/VH numpy arrays."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from src.satellite_fetcher import SentinelHubFetcher

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)
    fetcher.config = MagicMock()

    mock_response = MagicMock()
    mock_response.get_data.return_value = [np.zeros((256, 256), dtype=np.float32)] * 2

    with patch("src.satellite_fetcher SentinelHubRequest") as MockReq:
        instance = MagicMock()
        instance.get_data.return_value = [mock_response]
        MockReq.return_value = instance

        vv, vh = fetcher.fetch_sentinel1(LOMBOK_BBOX, "2026-06-07")
        assert vv.shape == (256, 256)
        assert vh.shape == (256, 256)
```

- [ ] **Step 9: Implement fetch_sentinel1 method**

Add to `SentinelHubFetcher` class:

```python
    def fetch_sentinel1(self, bbox: list, date: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch Sentinel-1 SAR VV/VH bands via Process API.

        Args:
            bbox: [W, S, E, N] bounding box
            date: ISO date string (YYYY-MM-DD)

        Returns:
            Tuple of (VV, VH) numpy arrays in dB.
        """
        from sentinelhub import (
            SentinelHubRequest,
            BBox,
            CRS,
            DataCollection,
            MimeType,
            SentinelHubDownloadClient,
        )

        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        time_interval = (f"{date}T00:00:00", f"{date}T23:59:59")

        evalscript_s1 = """
        //VERSION=3
        function setup() {
          return {
            input: ["VV", "VV"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }
        function evaluatePixel(sample) {
          return [sample.VV, sample.VH];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript_s1,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW.define_from(
                        "s1iw", service_url=self.config.sh_base_url
                    ),
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=[512, 512],
            config=self.config,
        )

        data = request.get_data()[0]
        vv = data[:, :, 0].astype(np.float32)
        vh = data[:, :, 1].astype(np.float32)
        return vv, vh
```

- [ ] **Step 10: Run S1 fetch test**

Run: `pytest tests/test_satellite_fetcher.py::test_fetch_sentinel1_returns_numpy -v`
Expected: PASS

- [ ] **Step 11: Write failing test for S2 fetching**

Append to `tests/test_satellite_fetcher.py`:

```python
def test_fetch_sentinel2_returns_numpy():
    """Verify fetch_sentinel2 returns Green/NIR numpy arrays."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from src.satellite_fetcher import SentinelHubFetcher

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)
    fetcher.config = MagicMock()

    with patch("src.satellite_fetcher SentinelHubRequest") as MockReq:
        instance = MagicMock()
        instance.get_data.return_value = [np.zeros((512, 512, 2), dtype=np.float32)]
        MockReq.return_value = instance

        green, nir = fetcher.fetch_sentinel2(LOMBOK_BBOX, "2026-06-07")
        assert green.shape == (512, 512)
        assert nir.shape == (512, 512)
```

- [ ] **Step 12: Implement fetch_sentinel2 method**

Add to `SentinelHubFetcher` class:

```python
    def fetch_sentinel2(self, bbox: list, date: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch Sentinel-2 Green/NIR bands via Process API.

        Args:
            bbox: [W, S, E, N] bounding box
            date: ISO date string (YYYY-MM-DD)

        Returns:
            Tuple of (Green, NIR) numpy arrays as reflectance.
        """
        from sentinelhub import (
            SentinelHubRequest,
            BBox,
            CRS,
            DataCollection,
            MimeType,
        )

        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        time_interval = (f"{date}T00:00:00", f"{date}T23:59:59")

        evalscript_s2 = """
        //VERSION=3
        function setup() {
          return {
            input: ["B03", "B08"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B03, sample.B08];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript_s2,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=[512, 512],
            config=self.config,
        )

        data = request.get_data()[0]
        green = data[:, :, 0].astype(np.float32)
        nir = data[:, :, 1].astype(np.float32)
        return green, nir
```

- [ ] **Step 13: Run S2 fetch test**

Run: `pytest tests/test_satellite_fetcher.py::test_fetch_sentinel2_returns_numpy -v`
Expected: PASS

- [ ] **Step 14: Write failing test for save_rasters**

Append to `tests/test_satellite_fetcher.py`:

```python
def test_save_rasters_writes_files(tmp_path):
    """Verify save_rasters writes GeoTIFF to processed dir."""
    import numpy as np
    from src.satellite_fetcher import SentinelHubFetcher

    fetcher = SentinelHubFetcher.__new__(SentinelHubFetcher)

    rasters = {
        "s1_vv": np.zeros((64, 64), dtype=np.float32),
        "s1_vh": np.zeros((64, 64), dtype=np.float32),
        "s2_green": np.zeros((64, 64), dtype=np.float32),
        "s2_nir": np.zeros((64, 64), dtype=np.float32),
        "date": "2026-06-07",
    }

    with patch("src.satellite_fetcher.PROCESSED_DIR", tmp_path):
        fetcher.save_rasters(rasters)

    assert (tmp_path / "sentinel1_reproj.tif").exists()
    assert (tmp_path / "sentinel2_reproj.tif").exists()
```

- [ ] **Step 15: Implement save_rasters and save_status methods**

Add to `SentinelHubFetcher` class:

```python
    def save_rasters(self, rasters: dict) -> None:
        """Save fetched rasters to data/processed/ as GeoTIFF.

        Overwrites existing files so pipeline always uses latest data.

        Args:
            rasters: Dict with s1_vv, s1_vh, s2_green, s2_nir arrays + metadata.
        """
        import rasterio
        from rasterio.transform import from_bounds

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        bbox = rasters.get("bbox", NTB_BBOX)
        transform = from_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3],
            rasters["s1_vv"].shape[1],
            rasters["s1_vv"].shape[0],
        )

        # Save Sentinel-1 (VV + VH as 2-band)
        s1_path = PROCESSED_DIR / "sentinel1_reproj.tif"
        with rasterio.open(
            str(s1_path), "w", driver="GTiff",
            height=rasters["s1_vv"].shape[0],
            width=rasters["s1_vv"].shape[1],
            count=2, dtype="float32",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(rasters["s1_vv"], 1)
            dst.write(rasters["s1_vh"], 2)
        logger.info(f"Saved S1: {s1_path}")

        # Save Sentinel-2 (Green + NIR as 2-band)
        s2_path = PROCESSED_DIR / "sentinel2_reproj.tif"
        with rasterio.open(
            str(s2_path), "w", driver="GTiff",
            height=rasters["s2_green"].shape[0],
            width=rasters["s2_green"].shape[1],
            count=2, dtype="float32",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(rasters["s2_green"], 1)
            dst.write(rasters["s2_nir"], 2)
        logger.info(f"Saved S2: {s2_path}")

        # Save sync status
        self.save_status(rasters)

    def save_status(self, rasters: dict) -> None:
        """Save sync status to JSON for dashboard."""
        import json

        status = {
            "last_sync": datetime.utcnow().isoformat() + "Z",
            "sentinel1": {
                "date": rasters["date"],
                "status": "ok",
            },
            "sentinel2": {
                "date": rasters["date"],
                "cloud": rasters.get("cloud_pct", 0),
                "status": "ok",
            },
            "bbox": rasters.get("bbox", NTB_BBOX),
        }

        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        logger.info(f"Saved status: {STATUS_FILE}")
```

- [ ] **Step 16: Run save_rasters test**

Run: `pytest tests/test_satellite_fetcher.py::test_save_rasters_writes_files -v`
Expected: PASS

- [ ] **Step 17: Implement fetch_all orchestrator**

Add to `SentinelHubFetcher` class:

```python
    def fetch_all(self, bbox: list | None = None) -> dict | None:
        """Full fetch cycle: check scenes → fetch S1 + S2 → return rasters.

        Args:
            bbox: Override bounding box. Defaults to NTB_BBOX.

        Returns:
            Dict with raster arrays + metadata, or None if no new data.
        """
        if bbox is None:
            bbox = NTB_BBOX

        # 1. Check for new scenes
        logger.info(f"Checking new scenes for bbox={bbox}")
        latest_date = self.check_new_scenes(bbox)

        if not latest_date:
            logger.info("No new Sentinel-1 scenes found")
            return None

        date_str = latest_date[:10]  # YYYY-MM-DD
        logger.info(f"New scene found: {date_str}")

        # 2. Fetch Sentinel-1 SAR
        logger.info(f"Fetching Sentinel-1 for {date_str}")
        vv, vh = self.fetch_sentinel1(bbox, date_str)

        # 3. Fetch Sentinel-2 Optical
        logger.info(f"Fetching Sentinel-2 for {date_str}")
        green, nir = self.fetch_sentinel2(bbox, date_str)

        return {
            "s1_vv": vv,
            "s1_vh": vh,
            "s2_green": green,
            "s2_nir": nir,
            "date": date_str,
            "bbox": bbox,
            "cloud_pct": 0,  # TODO: extract from S2 metadata
        }
```

- [ ] **Step 18: Run all fetcher tests**

Run: `pytest tests/test_satellite_fetcher.py -v`
Expected: ALL PASS

- [ ] **Step 19: Commit fetcher module**

```bash
git add src/satellite_fetcher.py tests/test_satellite_fetcher.py
git commit -m "feat: add Sentinel Hub fetcher for automated satellite data acquisition"
```

---

### Task 4: Add daily periodic task to Celery

**Files:**
- Modify: `api/tasks.py`

- [ ] **Step 1: Add daily sync task**

Append to `api/tasks.py`:

```python
# ---------------------------------------------------------------------------
# Task: Daily Satellite Data Sync (Sentinel Hub)
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, name="aeco.daily_satellite_sync")
def task_daily_satellite_sync(self) -> dict:
    """Daily task: Fetch new satellite data and run flood pipeline.

    Scheduled via Celery Beat at 06:00 WITA (22:00 UTC previous day).

    Returns:
        Dict with sync result metadata.
    """
    from satellite_fetcher import SentinelHubFetcher
    from report_generator import compute_aoi_flood_stats, generate_esg_pdf
    from notifier import send_flood_alert

    self.update_state(state="PROCESSING", meta={"step": "fetching_satellite_data"})

    try:
        fetcher = SentinelHubFetcher()
        new_data = fetcher.fetch_all()
    except Exception as exc:
        logger.error(f"Satellite fetch failed: {exc}")
        send_flood_alert(
            area_ha=0, lat=-8.5, lon=116.8,
            timestamp=datetime.utcnow().isoformat(),
            message=f"Satellite fetch error: {exc}",
        )
        return {"status": "error", "error": str(exc)}

    if not new_data:
        logger.info("No new satellite data available")
        return {"status": "no_new_data"}

    # Save rasters (overwrites existing)
    self.update_state(state="PROCESSING", meta={"step": "saving_rasters"})
    fetcher.save_rasters(new_data)

    # Pipeline runs automatically on next API request
    # (final_flood_map.tif is generated on-demand or cached)
    logger.info(f"Satellite data updated: {new_data['date']}")

    return {
        "status": "updated",
        "date": new_data["date"],
        "bbox": new_data["bbox"],
    }
```

- [ ] **Step 2: Add Celery Beat schedule**

Add to `celery_app.conf.update(...)` block in `api/tasks.py`:

```python
    beat_schedule={
        "daily-satellite-sync": {
            "task": "aeco.daily_satellite_sync",
            "schedule": crontab(hour=22, minute=0),  # 22:00 UTC = 06:00 WITA
        },
    },
```

Also add import at top of file:
```python
from celery.schedules import crontab
from datetime import datetime
```

- [ ] **Step 3: Verify task registration**

Run: `python -c "from api.tasks import celery_app; print([t for t in celery_app.conf.get('beat_schedule', {}).keys()])"`
Expected: `['daily-satellite-sync']`

- [ ] **Step 4: Commit**

```bash
git add api/tasks.py
git commit -m "feat: add daily satellite sync task with Celery Beat schedule"
```

---

### Task 5: Add satellite status API endpoints

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Add SatelliteStatus response model**

Add after existing Pydantic models in `api/main.py`:

```python
class SatelliteStatusResponse(BaseModel):
    """Status of satellite data sync."""
    last_sync: str | None = Field(None, description="Last sync ISO timestamp")
    sentinel1: dict | None = Field(None, description="S1 status + date")
    sentinel2: dict | None = Field(None, description="S2 status + date + cloud")
    next_check: str = Field(..., description="Next scheduled sync")
    quota_remaining: int = Field(3000, description="Monthly API quota remaining")
```

- [ ] **Step 2: Add /satellite/status endpoint**

```python
@app.get("/satellite/status", response_model=SatelliteStatusResponse)
def satellite_status():
    """Get satellite data sync status."""
    import json
    from datetime import datetime, timedelta

    status_file = DATA_DIR / "satellite_status.json"

    if status_file.exists():
        status = json.loads(status_file.read_text())
    else:
        status = {}

    # Next check: 06:00 WITA tomorrow (22:00 UTC today)
    now = datetime.utcnow()
    next_check = now.replace(hour=22, minute=0, second=0, microsecond=0)
    if next_check <= now:
        next_check += timedelta(days=1)

    return SatelliteStatusResponse(
        last_sync=status.get("last_sync"),
        sentinel1=status.get("sentinel1"),
        sentinel2=status.get("sentinel2"),
        next_check=next_check.isoformat() + "Z",
        quota_remaining=3000,  # TODO: track actual usage
    )
```

- [ ] **Step 3: Add /satellite/sync endpoint**

```python
@app.post("/satellite/sync")
def satellite_sync():
    """Trigger manual satellite data sync."""
    from api.tasks import task_daily_satellite_sync

    task = task_daily_satellite_sync.delay()
    return {
        "task_id": task.id,
        "status": "PENDING",
        "message": "Satellite sync triggered",
        "poll_url": f"/predict/status/{task.id}",
    }
```

- [ ] **Step 4: Verify endpoints**

Run: `python -c "from api.main import app; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add api/main.py
git commit -m "feat: add /satellite/status and /satellite/sync API endpoints"
```

---

### Task 6: Add celery-beat service to docker-compose

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add celery-beat service**

Add after `worker` service in `docker-compose.yml`:

```yaml
  celery-beat:
    build: .
    command: celery -A api.tasks beat --loglevel=info
    env_file:
      - .env
    environment:
      - PYTHONPATH=/app/src:/app/api
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
    restart: always
```

- [ ] **Step 2: Validate docker-compose syntax**

Run: `docker-compose config --quiet && echo "OK" || echo "INVALID"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build: add celery-beat service for daily satellite sync scheduler"
```

---

### Task 7: Update dashboard with satellite sync status

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add satellite status card HTML**

Insert after the existing "System Performance" metric-card in the left panel (around line 412):

```html
<div class="metric-card" style="border-left-color: var(--info-blue);">
    <span class="metric-label" style="color: var(--info-blue);">🛰️ Satellite Data</span>
    <div style="font-size: 11px; margin-top: 8px; line-height: 1.8; color: var(--text-main);">
        <div>Last Sync: <span id="last-sync" style="color: var(--info-blue); font-weight: bold;">--</span></div>
        <div>Sentinel-1: <span id="s1-status" style="font-weight: bold;">--</span></div>
        <div>Sentinel-2: <span id="s2-status" style="font-weight: bold;">--</span></div>
        <div>Next Check: <span id="next-check" style="color: var(--text-muted);">--</span></div>
        <div>Quota: <span id="quota-remaining" style="color: var(--primary-neon);">--</span> req/month</div>
    </div>
    <button onclick="manualSync()" class="btn-primary" style="font-size: 11px; padding: 8px; margin-top: 10px;">
        🔄 Sync Now
    </button>
    <div id="sync-status-msg" style="font-size: 10px; margin-top: 6px; color: var(--text-muted); display: none;"></div>
</div>
```

- [ ] **Step 2: Add JavaScript functions**

Add before closing `</script>` tag:

```javascript
async function fetchSatelliteStatus() {
    try {
        const res = await fetch('/satellite/status');
        if (!res.ok) return;
        const d = await res.json();

        document.getElementById('last-sync').textContent = d.last_sync
            ? new Date(d.last_sync).toLocaleString('id-ID', { timeZone: 'Asia/Makassar' })
            : 'Never';

        document.getElementById('s1-status').textContent = d.sentinel1
            ? d.sentinel1.date + ' (' + d.sentinel1.status + ')'
            : 'No data';

        document.getElementById('s2-status').textContent = d.sentinel2
            ? d.sentinel2.date + ' (cloud: ' + d.sentinel2.cloud + '%)'
            : 'No data';

        document.getElementById('next-check').textContent = d.next_check
            ? new Date(d.next_check).toLocaleString('id-ID', { timeZone: 'Asia/Makassar' })
            : '--';

        document.getElementById('quota-remaining').textContent = d.quota_remaining;
    } catch (e) {
        console.error('Satellite status error:', e);
    }
}

async function manualSync() {
    const msgEl = document.getElementById('sync-status-msg');
    msgEl.style.display = 'block';
    msgEl.style.color = 'var(--info-blue)';
    msgEl.textContent = 'Syncing...';

    try {
        const res = await fetch('/satellite/sync', { method: 'POST' });
        const d = await res.json();
        msgEl.textContent = 'Sync triggered! Task: ' + d.task_id;
        msgEl.style.color = 'var(--primary-neon)';

        setTimeout(() => { msgEl.style.display = 'none'; }, 5000);
        setTimeout(fetchSatelliteStatus, 10000);
    } catch (e) {
        msgEl.textContent = 'Sync failed: ' + e.message;
        msgEl.style.color = 'var(--danger-red)';
    }
}

fetchSatelliteStatus();
setInterval(fetchSatelliteStatus, 60000);
```

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add satellite sync status card to dashboard"
```

---

### Task 8: End-to-end verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify all imports work**

Run: `python -c "from src.satellite_fetcher import SentinelHubFetcher; from api.tasks import task_daily_satellite_sync; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Verify Docker build**

Run: `docker-compose config --quiet && echo "Docker config OK"`
Expected: `Docker config OK`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Sentinel Hub integration complete — real-time satellite data for NTB flood monitoring"
```

---

## Setup Instructions (for user)

After implementation, user needs to:

1. **Create Sentinel Hub account:** https://apps.sentinel-hub.com/
2. **Get credentials:** Settings → OAuth Clients → Create
3. **Add to `.env`:**
   ```
   SH_CLIENT_ID=your_id
   SH_CLIENT_SECRET=your_secret
   SH_INSTANCE_ID=your_instance
   ```
4. **Deploy:** `docker-compose up --build`
