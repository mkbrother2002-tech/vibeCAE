"""Collect non-system DLL dependencies needed by the bundled CalculiX executable."""

import os
import shutil
import sys
from collections import deque
from pathlib import Path

import pefile


def imported_dlls(path: Path) -> set[str]:
    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
        return {
            entry.dll.decode("ascii").lower()
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        }
    finally:
        pe.close()


def main() -> None:
    ccx = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    conda_prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
    conda_root = Path(os.environ.get("CONDA", conda_prefix)).resolve()
    available: dict[str, Path] = {}
    # MKL's directory layout changes between Conda releases (recent packages
    # place the versioned runtime outside Library/bin), so index the complete
    # build environment instead of relying on a fixed directory list.
    for root in dict.fromkeys((conda_prefix, conda_root)):
        for dll in root.rglob("*"):
            if dll.is_file() and dll.suffix.lower() == ".dll":
                available.setdefault(dll.name.lower(), dll)

    # Conda may install a versioned MKL runtime (for example mkl_rt.3.dll)
    # while BLAS/LAPACK imports the stable mkl_rt.dll name. Ship both names.
    if "mkl_rt.dll" not in available:
        versioned_mkl = sorted(
            (path for path in available.values() if path.name.lower().startswith("mkl_rt.")),
            key=lambda path: path.name,
        )
        if versioned_mkl:
            available["mkl_rt.dll"] = versioned_mkl[-1]

    destination.mkdir(parents=True, exist_ok=True)
    queue = deque([ccx])
    inspected = {ccx.name.lower()}
    copied: list[Path] = []

    while queue:
        binary = queue.popleft()
        for name in imported_dlls(binary):
            if name in inspected:
                continue
            inspected.add(name)
            dependency = available.get(name)
            # Windows system DLLs are deliberately not copied and will not be
            # present in the isolated Conda search directories.
            if dependency is None:
                continue
            target = destination / name
            shutil.copy2(dependency, target)
            copied.append(target)
            if target.name.lower() != dependency.name.lower():
                versioned_target = destination / dependency.name
                shutil.copy2(dependency, versioned_target)
                copied.append(versioned_target)
            queue.append(dependency)

    if not copied:
        raise RuntimeError("No CalculiX runtime DLLs were discovered")
    print(f"Collected {len(copied)} CalculiX DLLs:")
    for path in copied:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
