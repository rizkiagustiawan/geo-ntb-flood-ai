import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from rio_tiler.io import Reader
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.utils import render
import numpy as np
import os

app = FastAPI(title="Sumbawa Flood AI Server")

TIF_PATH = "outputs/predictions/final_flood_map.tif"


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("outputs/web/index.html", "r") as f:
        return f.read()


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Menyajikan tile peta (Z/X/Y) dengan pewarnaan dinamis"""
    if not os.path.exists(TIF_PATH):
        raise HTTPException(status_code=404, detail="Peta final belum dibuat!")

    with Reader(TIF_PATH) as dst:
        try:
            # Coba potong petanya sesuai koordinat dari Leaflet
            img = dst.tile(x, y, z)
        except TileOutsideBounds:
            # Kalau Leaflet minta area di luar peta (lautan lepas), lempar 404 (Skip)
            raise HTTPException(status_code=404, detail="Tile di luar batas")

        # Ambil data raster (Band 1)
        data = img.data[0]

        # Bikin Alpha Mask (Transparansi):
        # Value 1 (Banjir) -> 255 (Solid)
        # Value 0 (Aman)   -> 0 (Transparan)
        alpha_mask = (data == 1).astype(np.uint8) * 255

        # Bikin array RGB (3 channel: Red, Green, Blue)
        rgb = np.zeros((3, img.height, img.width), dtype=np.uint8)
        rgb[0, :, :] = 0  # R
        rgb[1, :, :] = 180  # G
        rgb[2, :, :] = 216  # B

        # Render array RGB (3 band) + alpha_mask (1 band) = 4 band (RGBA valid untuk PNG!)
        content = render(rgb, alpha_mask, img_format="PNG")

        return Response(content=content, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
