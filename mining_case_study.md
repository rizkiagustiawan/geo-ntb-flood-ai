# 🏭 Industrial Case Study: PT Amman Mineral

## Overview
While A.E.C.O is designed primarily for regional flood disaster management in Sumbawa, its high-resolution Multisensor Fusion architecture makes it highly applicable to heavy industry. This case study details a specific Area of Interest (AOI) configuration tailored for the **Batu Hijau Pit** and **Tailing Storage Facilities (TSF)** operated by PT Amman Mineral in West Sumbawa (Sumbawa Barat).

## Area of Interest (AOI) Configuration
The system leverages a constrained bounding box specifically focused on the mining operational zones to ensure high-frequency, low-latency processing.

- **Target Zone:** Batu Hijau Pit & TSF Perimeter
- **Coordinates (BBOX):** `[116.7, -9.0, 116.9, -8.8]`
- **Data Sources:** Sentinel-1 (Interferometric Wide Swath) & Sentinel-2 (High-Res Optical)
- **Processing Cadence:** Bi-weekly (aligned with Sentinel satellite passes)

## Tailing Dam Integrity Monitoring via SAR
Monitoring a Tailing Storage Facility requires extreme precision. The integrity of the dam perimeter is critical to preventing environmental catastrophes. A.E.C.O provides an autonomous early-warning system through the following mechanisms:

### 1. SAR Backscatter Anomalies
Smooth, undisturbed water in the tailing pond exhibits a very specific, low backscatter profile (typically below -18 dB in VV polarization). However, if there is a breach, seepage, or unauthorized discharge into the surrounding dry perimeter, the radar signature changes dramatically. A.E.C.O continuously monitors the perimeter for unexpected drops in backscatter (indicating new standing water) outside the designated pond boundaries.

### 2. Terrain-Aware Verification (DEM)
By coupling the SAR data with high-resolution Digital Elevation Models (DEM), A.E.C.O ensures that anomalies are only flagged if they occur along hydrologically feasible paths (e.g., downhill from the TSF crest). This significantly reduces false alarms caused by temporary equipment staging or vehicle movements.

### 3. Optical Fallback (Sentinel-2)
When cloud cover is minimal, the system utilizes Sentinel-2 NDWI (Normalized Difference Water Index) to cross-verify the SAR anomalies. This multisensor approach provides a highly robust, audit-ready compliance trail.

## Value Proposition
- **Autonomous ESG Compliance:** Continuous, unbiased monitoring without manual intervention.
- **Risk Mitigation:** Early detection of potential leaks before they escalate into structural failures.
- **Cost Efficiency:** Utilizes open-source European Space Agency (ESA) data, reducing the need for expensive commercial satellite tasking or hazardous physical patrols.
