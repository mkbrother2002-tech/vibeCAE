import os

from PyInstaller.utils.hooks import collect_all, collect_submodules


root = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
ccx_path = os.environ["VIBECAE_CCX_PATH"]

packages = [
    "streamlit",
    "streamlit_plotly_events",
    "plotly",
    "trimesh",
    "cadquery",
    "OCP",
    "gmsh",
    "numpy",
]

datas = [
    (os.path.join(root, "VibCAELight.py"), "."),
    (os.path.join(root, "fem_solver.py"), "."),
]
binaries = [(ccx_path, ".")]
hiddenimports = ["fem_solver", "appdirs"]

for package in packages:
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("reportlab")

a = Analysis(
    [os.path.join(SPECPATH, "VibeCAELightLauncher.py")],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "matplotlib", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VibeCAE Light",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep the console enabled for the first Windows beta so startup errors
    # remain visible on user machines and can be captured in GitHub Actions.
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VibeCAE Light",
)
