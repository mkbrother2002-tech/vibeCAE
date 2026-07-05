# -*- coding: utf-8 -*-
"""Верификация КЭ-конвейера VibeCAE на задаче о консольной балке.

Балка 200x20x10 мм (сталь), заделка на грани X=0.
Сравнение с сопроматом:
  1. Статика, сила P на торце: прогиб  δ = P·L³/(3EI) (+ сдвиг),
     напряжение в середине пролёта σ = P·(L−x)/W.
  2. Модальный анализ: f1 = (1.875²/2π)·√(EI/(ρA·L⁴)).

Запуск:  .venv/bin/python test_verification.py
"""
import io
import numpy as np

import fem_solver

# --- параметры задачи -------------------------------------------------------
L, B, H = 200.0, 20.0, 10.0      # мм
P = 100.0                        # Н, на торце, направление -Z
E_GPA, NU, RHO = 200.0, 0.3, 7850.0
MESH_MM = 4.0
TOL_PCT = 5.0

MATERIAL = {
    "elastic_modulus": E_GPA,       # ГПа
    "poisson": NU,
    "density": RHO,                 # кг/м³
    "thermal_expansion": 12e-6,
    "yield_strength": 245.0,
}


def make_beam_step():
    import gmsh
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("beam")
    gmsh.model.occ.addBox(0, 0, 0, L, B, H)
    gmsh.model.occ.synchronize()
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "beam.stp")
    gmsh.write(path)
    gmsh.finalize()
    return open(path, "rb").read()


def check(name, fem_val, theory, tol_pct=TOL_PCT):
    err = abs(fem_val - theory) / abs(theory) * 100.0
    ok = err <= tol_pct
    print(f"  {name:44s} МКЭ = {fem_val:10.4g}   теория = {theory:10.4g}   "
          f"расхожд. = {err:5.2f} %   {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print("Генерация STEP и сетки...")
    step_bytes = make_beam_step()
    fem = fem_solver.build_fem_mesh(step_bytes, mesh_size_mm=MESH_MM)
    print(f"  узлов: {fem['n_nodes']}, элементов C3D10: {fem['n_elements']}")

    E = E_GPA * 1000.0                     # МПа
    I = B * H ** 3 / 12.0                  # мм⁴ (изгиб вокруг Y, нагрузка по Z)
    W = B * H ** 2 / 6.0                   # мм³
    A = B * H                              # мм²

    base_params = {
        "constraint_face": "Грань X min",
        "constraint_zone_frac": 0.0,       # только узлы точно на X=0
        "include_gravity": False,
        "contents_mass_kg": 0.0,
        "temperature": 20.0,
        "direction": "-Z",
        "region_mode": "Пользовательская область",
        "region_axis": "X",
        "region_min": L - 1e-3,
        "region_max": L + 1e-3,
    }
    ok = True

    # --- 1. статика: сила на торце ------------------------------------------
    print("\n[1] Статика: сила P = %.0f Н на торце (X=L), направление -Z" % P)
    params = dict(base_params, analysis_type="Статический", load_type="Сила",
                  point_force_n=P, pressure_mpa=0.0, seismic_g=0.0)
    res = fem_solver.solve_scenario_fem(fem, MATERIAL, params)

    # прогиб: изгиб + сдвиг (Тимошенко, k = 5/6)
    G = E / (2 * (1 + NU))
    delta_th = P * L ** 3 / (3 * E * I) + P * L / (5.0 / 6.0 * G * A)
    ok &= check("прогиб торца, мм", res["max_disp_mm"], delta_th)

    # напряжение в середине пролёта (вдали от заделки и зоны нагрузки)
    x_mid = L / 2.0
    coords, vm = res["viz_coords"], res["viz_field"]
    sel = (np.abs(coords[:, 0] - x_mid) < 1.5) & (coords[:, 2] > H - 0.5)
    sigma_fem_mid = float(vm[sel].mean())
    sigma_th_mid = P * (L - x_mid) / W
    ok &= check("σ по Мизесу в середине пролёта, МПа", sigma_fem_mid, sigma_th_mid)

    # реакция опоры
    rz = res["reactions_n"][2]
    ok &= check("реакция опоры Rz, Н", rz, P, tol_pct=1.0)

    # --- 2. модальный анализ -------------------------------------------------
    print("\n[2] Модальный анализ: первая изгибная частота")
    params = dict(base_params, analysis_type="Модальный", load_type="Гравитация",
                  point_force_n=0.0, pressure_mpa=0.0, seismic_g=0.0)
    res = fem_solver.solve_scenario_fem(fem, MATERIAL, params)

    rho_mm = RHO * 1e-12                    # т/мм³
    f1_th = (1.875104 ** 2 / (2 * np.pi)) * np.sqrt(E * I / (rho_mm * A * L ** 4))
    ok &= check("f1 (изгиб по Z), Гц", res["first_frequency_hz"], f1_th)

    # вторая изгибная — вокруг Z (жёсткая ось), I2 = H·B³/12
    I2 = H * B ** 3 / 12.0
    f2_th = (1.875104 ** 2 / (2 * np.pi)) * np.sqrt(E * I2 / (rho_mm * A * L ** 4))
    f2_fem = res["modes"][1]["f_hz"]
    ok &= check("f2 (изгиб по Y), Гц", f2_fem, f2_th)

    print("\nИТОГ:", "все проверки пройдены (допуск %.0f %%)" % TOL_PCT if ok else "есть расхождения")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
