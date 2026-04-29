from fastapi import FastAPI, HTTPException
import rasterio, os, numpy as np

app = FastAPI()
MAP_PATH = "outputs/maps/flood_map.tif"


@app.get("/health")
def health():
    return {"status": "antigravity-stable"}


@app.get("/predict/at")
def predict_at(lat: float, lon: float):
    # Logika RTK: Cek status banjir di titik koordinat spesifik
    if not os.path.exists(MAP_PATH):
        raise HTTPException(404)
    with rasterio.open(MAP_PATH) as src:
        # Convert koordinat ke pixel index
        row, col = src.index(lon, lat)
        try:
            val = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
            return {"lat": lat, "lon": lon, "flood": bool(val == 1)}
        except:
            return {"error": "Outside bounds"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
