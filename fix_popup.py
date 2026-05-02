import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = """                const d = await res.json();
                const isFlood = d.flood === 1;
                const accent = isFlood ? '#ff3b3b' : '#00e676';
                const glow = isFlood ? 'rgba(255,59,59,0.3)' : 'rgba(0,230,118,0.2)';
                const icon = isFlood ? '🚨' : '✅';
                const label = isFlood ? 'CRITICAL: FLOOD DETECTED' : 'SAFE: NO FLOOD';
                const ndwiStr = d.ndwi !== null ? d.ndwi.toFixed(4) : 'N/A';
                const sarStr = d.sar_vv !== null ? d.sar_vv.toFixed(2) + ' dB' : 'N/A';

                popup.setContent(`
          <div style="font-family:'Inter',sans-serif; padding:10px; min-width:220px;">
            <div style="text-align:center; margin-bottom:10px;">
              <span style="font-size:22px;">${icon}</span><br>
              <strong style="color:${accent}; font-size:13px; letter-spacing:0.5px; text-shadow:0 0 12px ${glow};">${label}</strong>
            </div>
            <div style="background:rgba(255,255,255,0.04); border-radius:6px; padding:8px 10px; border-left:3px solid ${accent};">
              <table style="width:100%; font-size:11px; color:#ccc; border-collapse:collapse;">
                <tr>
                  <td style="padding:3px 0; color:#888;">Coordinates</td>
                  <td style="text-align:right; font-family:monospace; font-size:10px;">${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}</td>
                </tr>
                <tr>
                  <td style="padding:3px 0; color:#888;">NDWI</td>
                  <td style="text-align:right; font-family:monospace; color:${d.ndwi !== null && d.ndwi > 0.3 ? '#00b0ff' : '#aaa'};">${ndwiStr}</td>
                </tr>
                <tr>
                  <td style="padding:3px 0; color:#888;">SAR VV</td>
                  <td style="text-align:right; font-family:monospace; color:${d.sar_vv !== null && d.sar_vv < -15 ? '#ff6e40' : '#aaa'};">${sarStr}</td>
                </tr>
                <tr>
                  <td style="padding:3px 0; color:#888;">Method</td>
                  <td style="text-align:right; font-size:9px; color:#666;">${d.method.split(' ')[0]}</td>
                </tr>
              </table>
            </div>
            <div style="text-align:center; margin-top:8px; font-size:9px; color:#555; font-family:monospace;">
              CRS: ${d.crs} | ${new Date(d.timestamp).toLocaleTimeString('en-GB')} UTC
            </div>
          </div>
        `);"""

new_logic = """                const d = await res.json();
                let accent, glow, icon, label, extraMsg = '';

                if (d.status === "permanent_water") {
                    accent = 'var(--info-blue)';
                    glow = 'rgba(0, 210, 255, 0.3)';
                    icon = '🌊';
                    label = 'OCEAN / PERMANENT WATER';
                    extraMsg = '<div style="font-size: 9px; color: var(--text-muted); margin-top: 6px; text-align: center; line-height: 1.3;">Excluded from monitoring (Elevation &le; 0m)</div>';
                } else if (d.status === "flood_detected" || d.flood === 1) {
                    accent = 'var(--danger-red)';
                    glow = 'var(--danger-glow)';
                    icon = '🚨';
                    label = 'CRITICAL: FLOOD DETECTED';
                } else {
                    accent = '#00e676';
                    glow = 'rgba(0,230,118,0.2)';
                    icon = '✅';
                    label = 'SAFE: DRY LAND';
                }

                const ndwiStr = d.ndwi !== null ? d.ndwi.toFixed(4) : 'N/A';
                const sarStr = d.sar_vv !== null ? d.sar_vv.toFixed(2) + ' dB' : 'N/A';
                const statusBadge = d.status ? `<span style="background: var(--card-bg); border: 1px solid var(--panel-border); padding: 2px 6px; border-radius: 4px; font-size: 8px; text-transform: uppercase; color: var(--text-muted);">${d.status.replace('_', ' ')}</span>` : '';

                popup.setContent(`
          <div style="font-family:'Inter',sans-serif; padding:10px; min-width:220px;">
            <div style="text-align:center; margin-bottom:10px;">
              <span style="font-size:22px;">${icon}</span><br>
              <strong style="color:${accent}; font-size:13px; letter-spacing:0.5px; text-shadow:0 0 12px ${glow};">${label}</strong>
              ${extraMsg}
              <div style="margin-top: 5px;">${statusBadge}</div>
            </div>
            <div style="background:var(--card-bg); border-radius:6px; padding:8px 10px; border-left:3px solid ${accent}; border: 1px solid var(--panel-border); border-left-width: 3px;">
              <table style="width:100%; font-size:11px; color:var(--text-main); border-collapse:collapse;">
                <tr>
                  <td style="padding:3px 0; color:var(--text-muted);">Coordinates</td>
                  <td style="text-align:right; font-family:monospace; font-size:10px;">${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}</td>
                </tr>
                <tr>
                  <td style="padding:3px 0; color:var(--text-muted);">NDWI</td>
                  <td style="text-align:right; font-family:monospace; color:${d.ndwi !== null && d.ndwi > 0.3 ? 'var(--info-blue)' : 'var(--text-muted)'};">${ndwiStr}</td>
                </tr>
                <tr>
                  <td style="padding:3px 0; color:var(--text-muted);">SAR VV</td>
                  <td style="text-align:right; font-family:monospace; color:${d.sar_vv !== null && d.sar_vv < -15 ? '#ff6e40' : 'var(--text-muted)'};">${sarStr}</td>
                </tr>
                <tr>
                  <td style="padding:3px 0; color:var(--text-muted);">Method</td>
                  <td style="text-align:right; font-size:9px; color:var(--text-muted);">${d.method.split(' ')[0]}</td>
                </tr>
              </table>
            </div>
            <div style="text-align:center; margin-top:8px; font-size:9px; color:var(--text-muted); font-family:monospace;">
              CRS: ${d.crs} | ${new Date(d.timestamp).toLocaleTimeString('en-GB')} UTC
            </div>
          </div>
        `);
        
                // Fix MathJax rendering issues on dynamic DOM updates
                if (window.MathJax) {
                    MathJax.typesetPromise().catch(err => console.error("MathJax error:", err));
                }"""

# Escape backticks for Python script string
content = content.replace(old_logic, new_logic)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)
