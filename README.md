# VibeCAE - Engineering CAE Analysis Platform

**VibeCAE** is a Streamlit-based interactive tool for engineering screening of loading scenarios on 3D models. It provides mesh visualization, stress analysis (analytical formulas or a real FEM backend), modal analysis, response-spectrum seismic analysis, and automated PDF reporting.

> **Note:** Two calculation modes are available. The analytical mode uses beam/section engineering formulas for order-of-magnitude checks. The FEM mode (Level 2) runs a real finite element analysis with gmsh + CalculiX: linear-elastic C3D10 (quadratic tetrahedra) static, thermal, modal, and response-spectrum solutions. Contacts and welds are not modeled (multi-body STEP assemblies are merged with shared nodes); results are for engineering screening, not certified design verification.

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
  - Analysis types: static, thermal, modal, spectral (response-spectrum seismic)
  - Allowable stress [σ] = σy(T)/n verdicts per selected norm; for seismic combinations per PNAE G-7-002-86: NOC+DE (НУЭ+ПЗ) — [σ]×1.2, NOC+SSE (НУЭ+МРЗ) — [σ]×1.4

- **FEM Backend (Level 2, gmsh + CalculiX)**
  - Volume meshing of STEP geometry with quadratic tetrahedra (C3D10), configurable element size
  - Static analysis: gravity + quasi-static seismic as body loads, point force / pressure distributed over the load region by nodal areas, contents as point masses, support reaction check
  - Thermal analysis: uniform heating with real constrained-expansion stresses (not an upper-bound estimate)
  - Modal analysis: 10 modes, natural frequencies and effective modal masses per X/Y/Z
  - Response-spectrum method: per-mode response q = Γ·Sa(f)/ω² from a user floor-response spectrum, SRSS mode combination, conservative superposition with the static (NOC) von Mises field; effective-mass completeness check along the excitation axis
  - Von Mises stress maps on the surface mesh, 95th-percentile reporting to flag constraint singularities
  - Verified against beam theory: see `test_verification.py` (cantilever 200×20×10 mm — tip deflection 0.7 %, mid-span stress 0.03 %, reactions 0.00 %, f1/f2 within 0.5 % of theory)

- **Results & Reporting**
  - Stress breakdown: membrane σ=F/A, bending σ=M/W, thermal components
  - First natural frequency estimate with 0.5–33 Hz seismic band check
  - Interactive 3D stress / mode-shape visualization
  - Automated PDF reports with summary tables, worst-case chart, and a methodology & assumptions section

## Installation

```bash
pip install -r requirements.txt
```

### FEM backend (optional, Level 2)

The FEM mode requires gmsh (meshing) and CalculiX `ccx` (solver):

```bash
pip install gmsh                    # meshing (already in requirements.txt)

# CalculiX via conda-forge (micromamba/conda):
micromamba create -p ~/.local/ccx-env -c conda-forge calculix
```

The application looks for `ccx` in `PATH` and in `~/.local/ccx-env/bin/ccx`. If either component is missing, the app falls back to analytical mode automatically.

To run the verification suite:

```bash
python test_verification.py
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
├── appCAE.py              # Main Streamlit application
├── fem_solver.py          # FEM backend: gmsh meshing, CalculiX .inp/.frd/.dat, spectral method
├── test_verification.py   # FEM verification against beam theory (cantilever)
└── requirements.txt       # Python dependencies
```

## Technology Stack

- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **Trimesh**: 3D mesh processing
- **CadQuery/OCP**: STEP file import and tessellation
- **gmsh**: Volume meshing (C3D10) for the FEM backend
- **CalculiX (ccx)**: Finite element solver
- **Plotly + Kaleido**: Interactive 3D visualizations and static image export
- **ReportLab**: PDF generation

## Development Notes

This application follows a modular architecture with clear separation:
- Material databases (with σy(T) curves) and normative safety factors
- Two interchangeable solvers behind the same result contract: analytical `solve_scenario(mesh, material, params)` and FEM `fem_solver.solve_scenario_fem(fem, material, params)`
- gmsh meshing runs in a subprocess (`build_fem_mesh_subprocess`) — gmsh is not thread-safe inside the Streamlit server
- 3D mesh processing utilities
- UI components (3 tabs)
- Report generation pipeline

## License

Internal tool for technical analysis and engineering design support.
