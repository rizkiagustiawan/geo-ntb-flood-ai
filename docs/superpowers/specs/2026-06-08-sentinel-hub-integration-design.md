# Sentinel Hub Integration — Design Spec

**Date:** 2026-06-08
**Status:** Approved
**Scope:** Integrate Sentinel Hub API for real-time satellite data fetching

---

## 1. Problem Statement

The A.E.C.O flood monitoring system currently relies on manually downloaded GeoTIFF files stored in `data/processed/`. This makes the system static — it cannot detect new floods without manual intervention. The goal is to make the system fully autonomous by integrating Sentinel Hub API for automated daily satellite data fetching.

## 2. Approach

**Sentinel Hub Process API** — direct pixel data request per AOI tile, in-memory processing through existing Rust engine pipeline.

### Why Process API over alternatives:
- Catalog API + Batch: Too complex for daily single-scene fetch
- Statistical API: No visual flood map output
- Direct download: Too large files, quota inefficient

## 3. Architecture

```
Celery Beat (daily @ 06:00 WITA)
    ↓
Sentinel Hub Fetcher (NEW)
    ├─ SH Catalog API → check new scenes
    ├─ SH Process API → fetch S1 (VV/VH) + S2 (Green/NIR) + DEM
    └─ Save to data/processed/ (overwrite existing)
    ↓
Existing Pipeline (UNCHANGED)
    ├─ Preprocess → features
    ├─ Rust Engine → NDWI + SAR mask
    ├─ XGBoost → flood classification
    └─ final_flood_map.tif
    ↓
Output Layer (UNCHANGED)
    ├─ Telegram alert (if flood detected)
    ├─ PDF ESG report
    └─ Dashboard update
```

## 4. New Component: `src/satellite_fetcher.py`

### Functions:
1. `authenticate()` — OAuth2 token from Sentinel Hub
2. `check_new_scenes(bbox, date_range)` — Catalog API query
3. `fetch_sentinel1(bbox, date)` — Process API → SAR VV/VH GeoTIFF
4. `fetch_sentinel2(bbox, date)` — Process API → Green/NIR GeoTIFF
5. `fetch_dem(bbox)` — Process API → DEMNAS elevation
6. `save_to_processed(rasters)` — Overwrite `data/processed/`

### Coverage — Full NTB:
```python
NTB_BBOX = [115.7, -9.1, 119.2, -8.1]  # [W, S, E, N]
LOMBOK_BBOX = [115.7, -8.9, 116.6, -8.1]
SUMBAWA_BBOX = [116.6, -9.1, 119.2, -8.1]
```

### Quota Management (Free Tier: 3000 req/month):
- Daily budget: 100 requests
- Per cycle: ~3 requests (S1 + S2 + DEM check)
- Buffer: 97 requests/day

### Configuration (`.env`):
```env
SH_CLIENT_ID=xxx
SH_CLIENT_SECRET=xxx
SH_INSTANCE_ID=xxx
```

## 5. Scheduler Integration

### New Celery periodic task in `api/tasks.py`:
```python
@app.task
def task_daily_satellite_sync():
    """Daily @ 06:00 WITA — fetch + process + alert"""
    fetcher = SentinelHubFetcher()
    new_data = fetcher.fetch_all(NTB_BBOX)
    
    if not new_data:
        return "No new scenes"
    
    fetcher.save_rasters(new_data)
    run_flood_pipeline()
    stats = compute_aoi_flood_stats(...)
    generate_esg_pdf(stats)
    send_flood_alert(...)
```

### docker-compose.yml addition:
```yaml
celery-beat:
  command: celery -A api.tasks beat --loglevel=info
```

### Retry logic:
- Fetch failure → 3 retries, exponential backoff
- Pipeline failure → Telegram alert "Pipeline Error"
- Quota exhausted → skip + alert "Quota limit reached"

## 6. Dashboard Update

### New card in `index.html` left panel:
- Last sync timestamp
- Sentinel-1 status + date
- Sentinel-2 status + date + cloud %
- Next scheduled check
- Quota remaining
- "Sync Now" button (manual trigger)

### New API endpoints:
- `GET /satellite/status` → sync metadata
- `POST /satellite/sync` → trigger manual sync

## 7. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `src/satellite_fetcher.py` | NEW | Sentinel Hub API client |
| `api/tasks.py` | MODIFY | Add daily periodic task |
| `api/main.py` | MODIFY | Add `/satellite/status` and `/satellite/sync` endpoints |
| `docker-compose.yml` | MODIFY | Add celery-beat service |
| `index.html` | MODIFY | Add satellite sync status card |
| `requirements.txt` | MODIFY | Add `sentinelhub-py` |
| `.env.example` | MODIFY | Add SH credentials template |

## 8. Dependencies

```
sentinelhub>=3.10.0  # Official Sentinel Hub Python SDK
```

## 9. Security

- Sentinel Hub credentials stored in `.env` (not committed)
- OAuth2 token cached in memory, auto-refresh
- No raw API keys exposed to frontend
- Rate limiting: max 100 requests/day enforced in code

## 10. Success Criteria

1. Daily automatic fetch of Sentinel-1/2 data for NTB
2. Flood map auto-updated within 30 minutes of new data
3. Telegram alert sent when flood detected
4. Dashboard shows real-time sync status
5. Manual "Sync Now" button works
6. Quota never exceeds 3000/month limit
