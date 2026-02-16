# MUSEAstrometryFieldSearch

Scripts to search for field candidates from star catalogs to do astrometry calibration in MUSE for Wide Field Mode (WFM) and Narrow Field Mode (NFM).

## 🔭 Project Overview

Astrometric calibration for the MUSE instrument requires specific stellar densities and magnitude ranges to ensure an accurate World Coordinate System (WCS) solution. This repository provides two examples of automated workflows to identify these fields using existing catalogs (such as ACS, HSC, or Gaia).

### Key Features:
* **Narrow Field Mode (NFM):** Searches for isolated reference stars within a specific magnitude range (typically 11.9–12.9) and verifies a $7.5'' \times 7.5''$ field of view.
* **Wide Field Mode (WFM):** Uses a sliding-grid algorithm with GUI to find $1' \times 1'$ fields with optimal star counts and density.
* **Automated Data Retrieval:** Functions to query Gaia DR3, the Hubble Source Catalog (HSC), and download FITS cutouts from the Legacy Survey.
* **Visualization:** Generates diagnostic scatter plots and FITS-based finding charts for every accepted field candidate.

---

## 🛠 Installation & Data Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/ojedarcelo/MUSEAstrometryFieldSearch.git](https://github.com/ojedarcelo/MUSEAstrometryFieldSearch.git)
   cd MUSEAstrometryFieldSearch
```

2. **Install dependencies:**
```bash
pip install numpy pandas matplotlib astropy astroquery

```


3. **Download project data:**
Run the provided script to download the necessary catalogs and FITS images from Zenodo:
```bash
chmod +x download_use_cases_data.sh
./download_use_cases_data.sh

```



---

## 🚀 Usage

### Narrow Field Mode (NFM)

Run the NFM search to find fields centered on bright reference stars:

```bash
python nfm_use_case.py

```

This script iterates through cluster catalogs, identifies reference stars, and filters the surrounding area based on crowding and brightness limits.

### Wide Field Mode (WFM)

Run the WFM sliding-grid search:

```bash
python wfm_use_case.py

```

This performs a grid-based search over the outskirts of astronomical targets, applying density constraints and automatically downloading background images for finding charts.

---

## 📁 Repository Structure

* `field_finder/`: Core logic modules.
* `nfm_finder.py`: Functions for reference star identification and NFM filtering.
* `wfm_finder.py`: Sliding-grid implementation and WFM-specific criteria.


* `aux_funcs.py`: Utility functions for sky coordinate formatting, online catalog queries (Gaia/HSC), and FITS table manipulation.
* `nfm_use_case.py` / `wfm_use_case.py`: Top-level execution scripts for each mode.
* `download_use_cases_data.sh`: Bash script to fetch required datasets.

---

## 📊 Outputs

For every accepted field, the tool saves:

1. **CSV Catalog:** A source list ready for the MUSE pipeline.
2. **Diagnostic Plot:** A scatter plot showing star distribution and magnitudes.
3. **Finding Chart:** A FITS image with the MUSE FOV and reference stars highlighted.

---

## 🤝 Contributing
If you find a bug or wish to add support for other star catalogs or imaging (e.g., HST, VVV, Pan-STARRS), feel free to open an issue or submit a pull request.

---

## Author

**Marcelo Nicolás Ojeda Cárdenas** Pontificia Universidad Católica de Chile
