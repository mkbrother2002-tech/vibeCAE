# VibeCAE - Engineering CAE Analysis Platform

**VibeCAE** is a Streamlit-based interactive platform for comprehensive engineering finite element analysis (FEA) with focus on stress analysis, modal analysis, and report generation.

## Features

- **CAD Import & Meshing**
  - Support for STL and STEP file formats
  - Automatic mesh generation with configurable element size
  - Interactive 3D mesh visualization
  - Region selection for load application

- **Material Database**
  - GOST-compliant Russian steel standards
  - Aerospace-grade alloys (Titanium, Nickel-based)
  - Quick material selection for rapid prototyping

- **Analysis Capabilities**
  - Static structural analysis
  - Thermal-stress coupling analysis
  - Modal frequency analysis
  - Seismic/spectral response analysis
  - Temperature-dependent material properties

- **Load Scenarios**
  - Multiple simultaneous scenarios
  - Configurable load directions (+X, -X, +Y, -Y, +Z, -Z)
  - Temperature and load type selection
  - Custom region selection for load application

- **Results & Reporting**
  - Interactive 3D stress field visualization
  - Comparative scenario analysis
  - Automated PDF reports with:
    - Title pages
    - Summary tables
    - Stress distribution charts
    - Engineering recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run appCAE.py
```

## Usage Workflow

1. **Tab 1 - Import**: Upload CAD model (STL or STEP) and configure mesh parameters
2. **Tab 2 - Scenarios**: Select analysis scenarios and customize parameters
3. **Tab 3 - Results**: View analysis results, compare scenarios, and export PDF report

## Project Structure

```
VibeCAE/
├── appCAE.py           # Main Streamlit application
├── requirements.txt    # Python dependencies
└── support_phone.stl   # Example model
```

## Technology Stack

- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **Trimesh**: 3D mesh processing
- **CadQuery/OCP**: CAD file handling
- **Plotly**: Interactive 3D visualizations
- **ReportLab**: PDF generation

## Development Notes

This application follows a modular architecture with clear separation:
- Material databases and presets
- Engineering analysis engine
- 3D mesh processing utilities
- Stress field calculation
- UI components (3 analysis tabs)
- Report generation pipeline

## License

Internal tool for technical analysis and engineering design support.
