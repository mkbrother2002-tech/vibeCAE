# VibeCAE - Engineering CAE Analysis Platform

**VibeCAE** is a Streamlit-based interactive tool for rapid engineering screening of loading scenarios on 3D models. It provides mesh visualization, physics-based analytical stress estimation (membrane + bending + thermal), natural frequency estimation, and automated PDF reporting.

> **Note:** Stress and frequency values are computed with analytical engineering formulas (beam/section models), **not** a finite element solver. All loads are entered in physical units (N, MPa, g, kg). Results are suitable for preliminary screening and order-of-magnitude checks; they must not be used as a substitute for verified FEA calculations. Stress concentration (holes, fillets, welds) is not captured.

## Features

- **CAD Import & Meshing**
  - Support for STL and STEP file formats
  - Automatic surface triangulation with configurable element size (STEP geometry is re-tessellated on size change; STL meshes can only be refined)
  - Interactive 3D mesh visualization
  - Mass properties: volume, mass, bounding dimensions, center of mass
  - Load region selection: whole model, axis-range region, or point-with-radius

- **Material Database**
  - GOST-compliant materials with density, Poisson ratio, thermal expansion
  - Temperature-dependent yield strength curves σy(T)
  - Normative safety factors (PNAE G-7-002-86, GOST 34233.1, custom)

- **Scenario Screening**
  - Multiple simultaneous scenarios
  - Physical load inputs: self-weight, contents mass (kg), point force (N), pressure (MPa), seismic acceleration (g)
  - Boundary conditions per scenario: constraint face, fixed/pinned type, zone depth
  - Analysis types: static, thermal (E·α·ΔT upper bound), modal (Rayleigh beam estimate), spectral (quasi-static seismic)
  - Allowable stress [σ] = σy(T)/n verdicts per selected norm

- **Results & Reporting**
  - Stress breakdown: membrane σ=F/A, bending σ=M/W, thermal components
  - First natural frequency estimate with 0.5–33 Hz seismic band check
  - Interactive 3D stress / mode-shape visualization
  - Automated PDF reports with summary tables, worst-case chart, and a methodology & assumptions section

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run appCAE.py
```

## Usage Workflow

1. **Tab 1 - Import**: Upload CAD model (STL or STEP), configure mesh parameters; review mass properties
2. **Tab 2 - Scenarios**: Select analysis scenarios, set physical loads, constraints, and load regions, then press "Сформировать набор сценариев"
3. **Tab 3 - Results**: View screening results with stress breakdown and frequency estimates, compare scenarios, and export a PDF report

## Project Structure

```
VibeCAE/
├── appCAE.py           # Main Streamlit application
└── requirements.txt    # Python dependencies
```

## Technology Stack

- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **Trimesh**: 3D mesh processing
- **CadQuery/OCP**: STEP file import and tessellation
- **Plotly + Kaleido**: Interactive 3D visualizations and static image export
- **ReportLab**: PDF generation

## Development Notes

This application follows a modular architecture with clear separation:
- Material databases (with σy(T) curves) and normative safety factors
- Analytical solver behind a `solve_scenario(mesh, material, params)` interface — designed to be swapped for an FEA backend (gmsh + CalculiX) without UI changes
- 3D mesh processing utilities
- UI components (3 tabs)
- Report generation pipeline

## License

Internal tool for technical analysis and engineering design support.
