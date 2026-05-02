import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix MathJax
content = re.sub(
    r'\$\$\s*NDWI\s*=\s*\x0crac\{Green - NIR\}\{Green \+ NIR\}\s*\$\$',
    r'$$ NDWI = \\frac{Green - NIR}{Green + NIR} $$',
    content
)

# Replace legend
old_legend_regex = re.compile(
    r'<div class="legend">\s*<div class="section-title" style="margin-top:0; border:none; padding:0; font-size:11px;">Multisensor Fusion Logic</div>.*?<div class="legend-item">.*?<div style="font-size: 10px; color: var\(--text-muted\); line-height: 1.6; margin-top: 5px;">.*?</div>\s*</div>',
    re.DOTALL
)

new_legend = """<div class="legend" id="fusion-legend">
        <div class="legend-header" onclick="toggleLegend()" style="display: flex; justify-content: space-between; align-items: center; cursor: pointer; border-bottom: none; margin-bottom: 0;">
            <div class="section-title" style="margin: 0; border: none; padding: 0; font-size: 11px; display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 14px;">ℹ️</span> Multisensor Fusion Logic
            </div>
            <span id="legend-chevron" style="transition: transform 0.3s; font-size: 12px; margin-left: 10px; color: var(--text-muted);">▼</span>
        </div>
        <div id="legend-content" style="overflow: hidden; transition: max-height 0.3s ease, opacity 0.3s ease, margin-top 0.3s ease; max-height: 200px; opacity: 1; margin-top: 10px;">
            <div class="legend-item">
                <div class="legend-dot" style="background:var(--danger-red); box-shadow: 0 0 10px var(--danger-red);"></div>
                <span>Critical Flood Zone</span>
            </div>
            <div style="font-size: 10px; color: var(--text-muted); line-height: 1.6; margin-top: 5px;">
                • Sentinel-1: Radar detection (All-weather)<br>
                • Sentinel-2: NDWI verification (Optical)<br>
                • DEM: Terrain-aware noise reduction
            </div>
        </div>
    </div>"""

content = old_legend_regex.sub(new_legend, content)

# Add toggleLegend function
toggle_script = """
        // Toggle Legend
        function toggleLegend() {
            const content = document.getElementById('legend-content');
            const chevron = document.getElementById('legend-chevron');
            if (content.style.maxHeight === '0px' || content.style.maxHeight === 0) {
                content.style.maxHeight = '200px';
                content.style.opacity = '1';
                content.style.marginTop = '10px';
                chevron.style.transform = 'rotate(0deg)';
            } else {
                content.style.maxHeight = '0px';
                content.style.opacity = '0';
                content.style.marginTop = '0px';
                chevron.style.transform = 'rotate(-90deg)';
            }
        }
"""

content = content.replace(
    "function toggleTSF(checkbox) {",
    toggle_script + "\n        function toggleTSF(checkbox) {"
)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)
