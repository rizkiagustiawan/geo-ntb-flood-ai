import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Buka gerbang buat GitHub Pages lu
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data dummy buat ngetes (Nanti bisa lu sambungin ke output Rust lu)
demo_geojson = {
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"risk": "Critical"},
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[118.62, -8.42], [118.72, -8.38], [118.82, -8.44], [118.84, -8.54], [118.75, -8.60], [118.64, -8.56], [118.60, -8.48], [118.62, -8.42]]]
    }
  }]
}

@app.get("/api/floods")
def get_flood_data():
    return {
        "total_ha": "1,893",
        "methodology": "AWS Production: Near Real-Time Sentinel-1 SAR via Rust Zero-Copy Engine.",
        "geojson": demo_geojson
    }

if __name__ == "__main__":
    # Kita jalanin di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
