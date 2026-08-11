import os
import sys


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


if __name__ == "__main__":
    # Streamlit asks for an onboarding email on a fresh Windows machine unless
    # this is set before importing its CLI, which would block app startup.
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    ccx = resource_path("ccx.exe")
    if os.path.isfile(ccx):
        os.environ["CCX_PATH"] = ccx

    # fem_solver uses sys.executable for isolated gmsh meshing. In a frozen
    # build that executable is this launcher, so handle the child protocol.
    if (
        len(sys.argv) >= 6
        and os.path.basename(sys.argv[1]).lower() == "fem_solver.py"
        and sys.argv[2] == "--mesh"
    ):
        import numpy as np
        import fem_solver

        with open(sys.argv[3], "rb") as fh:
            mesh = fem_solver.build_fem_mesh(
                fh.read(), mesh_size_mm=float(sys.argv[5])
            )
        np.savez_compressed(sys.argv[4], **mesh)
        raise SystemExit(0)

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        resource_path("VibCAELight.py"),
        "--global.developmentMode=false",
        "--server.headless=false",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(stcli.main())
