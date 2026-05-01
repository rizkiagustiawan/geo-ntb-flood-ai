import uuid
from pathlib import Path
from fpdf import FPDF

def generate_esg_pdf(report_data: dict) -> str:
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "GeoESG A.E.C.O Audit Report", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Timestamp: {report_data.get('timestamp')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Input Geometry Type: {report_data.get('geometry_type')}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Flood Statistics", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Total Area: {report_data.get('total_area_ha')} Ha", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Flooded Area: {report_data.get('flooded_area_ha')} Ha", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Flood Percentage: {report_data.get('flood_percentage')}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Pixel Resolution: {report_data.get('pixel_resolution_m')} m", new_x="LMARGIN", new_y="NEXT")
    
    out_dir = Path("/tmp/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"esg_report_{uuid.uuid4().hex[:8]}.pdf"
    
    pdf.output(str(out_path))
    return str(out_path)
