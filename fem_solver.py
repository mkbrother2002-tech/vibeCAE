# -*- coding: utf-8 -*-
"""КЭ-решатель VibeCAE (уровень 2): gmsh (тетраэдры C3D10) + CalculiX (ccx).

Интерфейс совместим с аналитическим solve_scenario:
    solve_scenario_fem(fem_mesh, material_data, params) -> dict

Единицы: мм, Н, МПа, т/мм³ (плотность), мм/с² (ускорения).
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

GRAVITY_MM_S2 = 9810.0
N_MODES = 10


# ---------------------------------------------------------------------------
# Поиск решателя
# ---------------------------------------------------------------------------

def find_ccx():
    candidates = [
        os.environ.get("CCX_PATH"),
        os.path.expanduser("~/.local/ccx-env/bin/ccx"),
        shutil.which("ccx"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def fem_available():
    return find_ccx() is not None


# ---------------------------------------------------------------------------
# Сетка из STEP
# ---------------------------------------------------------------------------

def build_fem_mesh(step_bytes, mesh_size_mm=8.0):
    """STEP -> КЭ-сетка C3D10. Возвращает словарь простых numpy-массивов (кэшируемый)."""
    import gmsh

    with tempfile.NamedTemporaryFile(suffix=".stp", delete=False) as f:
        f.write(step_bytes)
        step_path = f.name
    try:
        # interruptible=False: без обработчиков сигналов (Streamlit выполняет
        # скрипт не в главном потоке — иначе "signal only works in main thread")
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        if not vols:
            raise ValueError("В STEP-файле нет объёмных тел")
        if len(vols) > 1:
            # конформная сшивка нескольких тел (общие узлы на контакте)
            gmsh.model.occ.fragment(vols, [])
            gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_mm))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_mm) * 0.25)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("HighOrderElastic")
        gmsh.model.mesh.optimize("HighOrder")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_ids = node_tags.astype(np.int64)
        node_coords = coords.reshape(-1, 3).astype(float)
        order = np.argsort(node_ids)
        node_ids = node_ids[order]
        node_coords = node_coords[order]

        etypes, etags, enodes = gmsh.model.mesh.getElements(dim=3)
        tet_ids = tet_conn = None
        for et, tags, conn in zip(etypes, etags, enodes):
            if et == 11:  # 10-узловой тетраэдр
                conn = conn.astype(np.int64).reshape(-1, 10)
                # gmsh -> Abaqus/ccx: своп промежуточных узлов 9 и 10
                conn = conn[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
                tet_ids, tet_conn = tags.astype(np.int64), conn
        if tet_ids is None:
            raise ValueError("Не удалось построить тетраэдры 2-го порядка")

        # поверхностные треугольники (углы TRI6) для визуализации/нагрузок
        stypes, stags, snodes = gmsh.model.mesh.getElements(dim=2)
        surf_tris = []
        for et, tags, conn in zip(stypes, stags, snodes):
            if et == 9:  # 6-узловой треугольник
                surf_tris.append(conn.astype(np.int64).reshape(-1, 6)[:, :3])
        surf_tris = np.vstack(surf_tris) if surf_tris else np.zeros((0, 3), dtype=np.int64)
    finally:
        gmsh.finalize()
        os.unlink(step_path)

    id2idx = {int(nid): i for i, nid in enumerate(node_ids)}

    # объём по угловым узлам тетраэдров
    corner_idx = np.vectorize(id2idx.get)(tet_conn[:, :4])
    p = node_coords[corner_idx]
    v = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6.0
    volume_mm3 = float(v.sum())
    centroid = (p.mean(axis=1) * v[:, None]).sum(axis=0) / max(volume_mm3, 1e-9)

    surf_idx = np.vectorize(id2idx.get)(surf_tris)
    surf_node_idx = np.unique(surf_idx)

    return {
        "node_ids": node_ids,
        "node_coords": node_coords,
        "tet_ids": tet_ids,
        "tet_conn": tet_conn,
        "surf_tris_idx": surf_idx,          # индексы узлов (не id) поверхностных треугольников
        "surf_node_idx": surf_node_idx,     # индексы поверхностных узлов
        "volume_mm3": volume_mm3,
        "centroid": centroid,
        "n_nodes": int(len(node_ids)),
        "n_elements": int(len(tet_ids)),
    }


# ---------------------------------------------------------------------------
# Зоны: закрепление и область нагрузки (на узлах КЭ-сетки)
# ---------------------------------------------------------------------------

_CONSTRAINT_FACES = {
    "Нижняя грань (Z min)": (2, "min"),
    "Верхняя грань (Z max)": (2, "max"),
    "Грань X min": (0, "min"),
    "Грань X max": (0, "max"),
    "Грань Y min": (1, "min"),
    "Грань Y max": (1, "max"),
}


def constraint_node_indices(coords, face_option, zone_frac=0.05):
    axis, side = _CONSTRAINT_FACES.get(face_option, (2, "min"))
    lo, hi = float(coords[:, axis].min()), float(coords[:, axis].max())
    depth = max((hi - lo) * float(zone_frac), 1e-6)
    if side == "min":
        mask = coords[:, axis] <= lo + depth
    else:
        mask = coords[:, axis] >= hi - depth
    idx = np.where(mask)[0]
    return idx if len(idx) else np.arange(len(coords))


def region_surface_mask(coords, params):
    """Маска узлов области приложения нагрузки (та же логика, что в аналитике)."""
    mode = params.get("region_mode", "Автоматическая зона")
    if mode == "Точка с радиусом":
        pts = params.get("selection_points") or []
        r = float(params.get("selection_radius", 1.0))
        if pts:
            pts = np.asarray(pts, dtype=float).reshape(-1, 3)
            dist = np.linalg.norm(coords[:, None, :] - pts[None, :, :], axis=2)
            mask = np.any(dist <= r, axis=1)
            if mask.any():
                return mask
        return np.ones(len(coords), dtype=bool)
    if mode != "Пользовательская область":
        return np.ones(len(coords), dtype=bool)
    axis = {"X": 0, "Y": 1, "Z": 2}.get(params.get("region_axis", "Z"), 2)
    lo = float(params.get("region_min", coords[:, axis].min()))
    hi = float(params.get("region_max", coords[:, axis].max()))
    if lo > hi:
        lo, hi = hi, lo
    mask = (coords[:, axis] >= lo) & (coords[:, axis] <= hi)
    return mask if mask.any() else np.ones(len(coords), dtype=bool)


def _region_nodal_areas(fem, params):
    """Площади (мм²) на узлах области: 1/3 площади каждого треугольника области.
    Возвращает (nodal_areas, area_valid): если в области нет целых треугольников,
    веса равномерные и площадь недостоверна."""
    coords = fem["node_coords"]
    tris = fem["surf_tris_idx"]
    mask = region_surface_mask(coords, params)
    tri_in = mask[tris].all(axis=1)
    nodal = np.zeros(len(coords))
    if tri_in.any():
        t = tris[tri_in]
        p = coords[t]
        a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        for k in range(3):
            np.add.at(nodal, t[:, k], a / 3.0)
        return nodal, True
    # область без целых треугольников: равномерно по узлам маски
    idx = np.where(mask)[0]
    nodal[idx] = 1.0
    return nodal, False


def get_direction_vector(direction):
    m = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
         "+Z": (0, 0, 1), "-Z": (0, 0, -1)}
    return np.array(m.get(direction, (0, 0, -1)), dtype=float)


# ---------------------------------------------------------------------------
# Запись .inp
# ---------------------------------------------------------------------------

def _write_nset(f, name, ids):
    f.write(f"*NSET, NSET={name}\n")
    ids = sorted(int(i) for i in ids)
    for i in range(0, len(ids), 8):
        f.write(", ".join(map(str, ids[i:i + 8])) + "\n")


def write_inp(path, fem, material_data, params):
    """Формирует .inp для статического/температурного или модального расчёта."""
    coords = fem["node_coords"]
    node_ids = fem["node_ids"]
    analysis_type = params.get("analysis_type", "Статический")
    is_modal = analysis_type == "Модальный"

    e_mpa = float(material_data["elastic_modulus"]) * 1000.0
    nu = float(material_data.get("poisson", 0.3))
    rho_t_mm3 = float(material_data.get("density", 7850.0)) * 1e-12  # кг/м³ -> т/мм³
    alpha = float(material_data.get("thermal_expansion", 12e-6))

    fixed_idx = constraint_node_indices(coords, params.get("constraint_face"),
                                        params.get("constraint_zone_frac", 0.05))
    fixed_ids = node_ids[fixed_idx]

    # содержимое: точечные массы на узлах области нагрузки
    contents_kg = max(float(params.get("contents_mass_kg", 0.0)), 0.0)
    nodal_areas, area_valid = _region_nodal_areas(fem, params)
    region_idx = np.where(nodal_areas > 0)[0]
    region_w = nodal_areas[region_idx] / nodal_areas[region_idx].sum()

    # результирующее ускорение (гравитация + сейсмика) одной GRAV-нагрузкой
    g_vec = np.zeros(3)
    if params.get("include_gravity", True):
        g_vec += GRAVITY_MM_S2 * np.array([0.0, 0.0, -1.0])
    seismic_g = float(params.get("seismic_g", 0.0))
    if params.get("load_type") in ("Сейсмика", "Комбинированная") and seismic_g > 0:
        g_vec += seismic_g * GRAVITY_MM_S2 * get_direction_vector(params.get("direction", "-Z"))
    g_mag = float(np.linalg.norm(g_vec))

    d = get_direction_vector(params.get("direction", "-Z"))
    point_force = float(params.get("point_force_n", 0.0))
    pressure = float(params.get("pressure_mpa", 0.0))
    region_area = float(nodal_areas.sum()) if area_valid else 0.0
    pressure_force = pressure * region_area if pressure > 1e-12 else 0.0

    temperature = float(params.get("temperature", 20.0))
    apply_temp = analysis_type == "Температурный" and abs(temperature - 20.0) > 1e-6

    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for nid, xyz in zip(node_ids, coords):
            f.write(f"{nid}, {xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}\n")
        f.write("*ELEMENT, TYPE=C3D10, ELSET=EALL\n")
        for eid, conn in zip(fem["tet_ids"], fem["tet_conn"]):
            f.write(f"{eid}, " + ", ".join(map(str, conn)) + "\n")

        mass_elset = None
        if contents_kg > 1e-9 and len(region_idx):
            mass_elset = "CONTENTS"
            eid0 = int(fem["tet_ids"].max()) + 1
            f.write(f"*ELEMENT, TYPE=MASS, ELSET={mass_elset}\n")
            for k, idx in enumerate(region_idx):
                f.write(f"{eid0 + k}, {int(node_ids[idx])}\n")

        _write_nset(f, "FIX", fixed_ids)

        f.write("*MATERIAL, NAME=MAT1\n*ELASTIC\n")
        f.write(f"{e_mpa:.1f}, {nu}\n*DENSITY\n{rho_t_mm3:.6e}\n")
        f.write(f"*EXPANSION\n{alpha:.4e}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n")
        if mass_elset:
            # равные точечные массы искажают распределение незначительно на мелкой сетке;
            # суммарная масса точная (в тоннах)
            f.write(f"*MASS, ELSET={mass_elset}\n{contents_kg * 1e-3 / len(region_idx):.6e}\n")

        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 20.\n")
        f.write("*BOUNDARY\nFIX, 1, 3\n")

        f.write("*STEP\n")
        if is_modal:
            f.write(f"*FREQUENCY\n{N_MODES}\n")
            f.write("*NODE FILE\nU\n")
        else:
            f.write("*STATIC\n")
            if g_mag > 1e-9:
                gd = g_vec / g_mag
                f.write(f"*DLOAD\nEALL, GRAV, {g_mag:.3f}, {gd[0]:.6f}, {gd[1]:.6f}, {gd[2]:.6f}\n")
                if mass_elset:
                    f.write(f"*DLOAD\n{mass_elset}, GRAV, {g_mag:.3f}, {gd[0]:.6f}, {gd[1]:.6f}, {gd[2]:.6f}\n")
            total_cload = point_force + pressure_force
            if abs(total_cload) > 1e-9 and len(region_idx):
                f.write("*CLOAD\n")
                for idx, w in zip(region_idx, region_w):
                    fv = total_cload * w * d
                    nid = int(node_ids[idx])
                    for dof in range(3):
                        if abs(fv[dof]) > 1e-12:
                            f.write(f"{nid}, {dof + 1}, {fv[dof]:.6e}\n")
            if apply_temp:
                f.write(f"*TEMPERATURE\nNALL, {temperature:.2f}\n")
            f.write("*NODE FILE\nU\n*EL FILE\nS\n")
            f.write("*NODE PRINT, NSET=FIX, TOTALS=ONLY\nRF\n")
        f.write("*END STEP\n")

    return {
        "region_area_mm2": region_area,
        "pressure_force_n": pressure_force,
        "g_vec_mm_s2": g_vec,
        "fixed_count": int(len(fixed_ids)),
        "fixed_idx": fixed_idx,
        "region_idx": region_idx,
    }


# ---------------------------------------------------------------------------
# Запуск и разбор результатов
# ---------------------------------------------------------------------------

def run_ccx(job_path):
    ccx = find_ccx()
    if ccx is None:
        raise RuntimeError("CalculiX (ccx) не найден")
    env = dict(os.environ, OMP_NUM_THREADS=str(max(os.cpu_count() // 2, 1)))
    r = subprocess.run([ccx, "-i", os.path.basename(job_path)], capture_output=True,
                       text=True, env=env, cwd=os.path.dirname(job_path), timeout=1800)
    if "Job finished" not in (r.stdout or ""):
        tail = (r.stdout or "")[-2500:] + "\n" + (r.stderr or "")[-1000:]
        raise RuntimeError(f"CalculiX завершился с ошибкой:\n{tail}")
    return r.stdout


def parse_frd(frd_path):
    """Список блоков результатов по порядку: [{'name': 'DISP'|'STRESS', 'data': {nid: [..]}}]."""
    blocks = []
    current = None
    with open(frd_path) as f:
        for line in f:
            if line.startswith(" -4"):
                name = line.split()[1]
                current = {"name": name, "data": {}}
                blocks.append(current)
            elif line.startswith(" -1") and current is not None:
                nid = int(line[3:13])
                body = line.rstrip("\n")[13:]
                vals = [float(body[12 * k:12 * (k + 1)]) for k in range(len(body) // 12)]
                current["data"][nid] = vals
            elif line.startswith(" -3"):
                current = None
    return blocks


def parse_static_dat(text):
    """Суммарные реакции (Н) из .dat."""
    reactions = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) in (3, 4):
            try:
                vals = [float(p) for p in parts[-3:]]
            except ValueError:
                continue
            reactions.append(vals)
    return reactions[-1] if reactions else None


def parse_modal_dat(text):
    """Частоты (Гц) и эффективные модальные массы (кг) из .dat CalculiX."""
    freqs = []
    eff_rows = []
    total_eff = None
    section = None
    for ln in text.splitlines():
        s = ln.strip()
        if "E I G E N V A L U E   O U T P U T" in ln:
            section = "eig"
            continue
        if "P A R T I C I P A T I O N   F A C T O R S" in ln:
            section = "part"
            continue
        if "E F F E C T I V E   M O D A L   M A S S" in ln:
            section = "eff"
            continue
        if "T O T A L   E F F E C T I V E   M A S S" in ln:
            section = None
            continue
        if "F R A C T I O N" in ln:
            section = None
            continue
        if section == "eff" and s.startswith("TOTAL"):
            parts = s.split()
            try:
                total_eff = [float(p) * 1e3 for p in parts[1:4]]  # т -> кг
            except ValueError:
                pass
            continue
        if not s or not s[0].isdigit():
            continue
        parts = s.split()
        try:
            row = [float(p) for p in parts]
        except ValueError:
            continue
        if section == "eig" and len(row) >= 4:
            freqs.append(row[3])  # столбец CYCLES/TIME = Гц
        elif section == "eff" and len(row) >= 4:
            eff_rows.append([v * 1e3 for v in row[1:4]])  # т -> кг (X, Y, Z)
    return freqs, eff_rows, total_eff


def _von_mises(s6):
    sx, sy, sz, sxy, syz, szx = (s6[:, k] for k in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def _yield_strength_at_temp(material_data, temperature):
    curve = material_data.get("yield_temp_curve")
    if not curve:
        return float(material_data["yield_strength"])
    temps = sorted(curve)
    return float(np.interp(float(temperature), temps, [curve[t] for t in temps]))


# ---------------------------------------------------------------------------
# Главный интерфейс
# ---------------------------------------------------------------------------

def solve_scenario_fem(fem, material_data, params):
    """КЭ-расчёт сценария. Возвращает словарь, совместимый с аналитическим solve_scenario."""
    coords = fem["node_coords"]
    node_ids = fem["node_ids"]
    id2idx = {int(nid): i for i, nid in enumerate(node_ids)}
    analysis_type = params.get("analysis_type", "Статический")
    is_modal = analysis_type == "Модальный"

    rho = float(material_data.get("density", 7850.0))
    model_mass_kg = rho * fem["volume_mm3"] * 1e-9
    contents_kg = max(float(params.get("contents_mass_kg", 0.0)), 0.0)
    total_mass_kg = model_mass_kg + contents_kg

    with tempfile.TemporaryDirectory() as tmp:
        job = os.path.join(tmp, "job")
        meta = write_inp(job + ".inp", fem, material_data, params)
        run_ccx(job)
        blocks = parse_frd(job + ".frd")
        dat_text = open(job + ".dat").read() if os.path.exists(job + ".dat") else ""

    result = {
        "fem": True,
        "analysis_type": analysis_type,
        "total_mass_kg": total_mass_kg,
        "sigma_yield_t": _yield_strength_at_temp(material_data, params.get("temperature", 20.0)),
        "mass_props": {
            "mass_kg": model_mass_kg,
            "volume_mm3": fem["volume_mm3"],
            "volume_cm3": fem["volume_mm3"] / 1000.0,
            "center_of_mass": fem["centroid"],
            "is_exact": True,
        },
        "region_area_mm2": meta["region_area_mm2"],
        "n_nodes": fem["n_nodes"],
        "n_elements": fem["n_elements"],
        "fixed_count": meta["fixed_count"],
        "viz_coords": coords,
        "viz_tris": fem["surf_tris_idx"],
        "constraint_centroid": coords[meta["fixed_idx"]].mean(axis=0),
        "load_center": coords[meta["region_idx"]].mean(axis=0) if len(meta["region_idx"]) else fem["centroid"],
        "sigma_membrane": 0.0,
        "sigma_bending": 0.0,
        "sigma_thermal": 0.0,
    }

    # составляющие нагрузки (для отчёта)
    force_terms = []
    direction_label = params.get("direction", "-Z")
    if params.get("include_gravity", True) and total_mass_kg > 0:
        force_terms.append({"name": "Вес (модель + содержимое)",
                            "value_n": total_mass_kg * 9.81, "direction": "-Z"})
    seismic_g = float(params.get("seismic_g", 0.0))
    if params.get("load_type") in ("Сейсмика", "Комбинированная") and seismic_g > 0:
        force_terms.append({"name": f"Сейсмическая инерционная ({seismic_g:.2f} g)",
                            "value_n": total_mass_kg * 9.81 * seismic_g, "direction": direction_label})
    pf = float(params.get("point_force_n", 0.0))
    if abs(pf) > 1e-9:
        force_terms.append({"name": "Сосредоточенная сила", "value_n": pf, "direction": direction_label})
    if meta["pressure_force_n"] > 1e-9:
        force_terms.append({"name": f"Давление {params.get('pressure_mpa'):g} МПа × {meta['region_area_mm2']:,.0f} мм²",
                            "value_n": meta["pressure_force_n"], "direction": direction_label})
    result["force_terms"] = force_terms
    result["force_total_n"] = float(sum(abs(t["value_n"]) for t in force_terms))

    if is_modal:
        freqs, eff_rows, total_eff = parse_modal_dat(dat_text)
        modes = []
        for i, f_hz in enumerate(freqs):
            em = eff_rows[i] if i < len(eff_rows) else [0.0, 0.0, 0.0]
            modes.append({"mode": i + 1, "f_hz": f_hz,
                          "effmass_x_kg": em[0], "effmass_y_kg": em[1], "effmass_z_kg": em[2]})
        result["modes"] = modes
        if total_eff is None and eff_rows:
            total_eff = [float(sum(r[k] for r in eff_rows)) for k in range(3)]
        result["total_effective_mass_kg"] = total_eff
        result["first_frequency_hz"] = freqs[0] if freqs else None
        # форма первого тона: |U| (относительные, 0..1)
        disp_blocks = [b for b in blocks if b["name"] == "DISP"]
        if disp_blocks:
            u = np.zeros((len(coords), 3))
            for nid, vals in disp_blocks[0]["data"].items():
                u[id2idx[nid]] = vals[:3]
            umag = np.linalg.norm(u, axis=1)
            m = umag.max()
            result["viz_field"] = umag / m if m > 0 else umag
        result["sigma_total"] = 0.0
        return result

    # статика/термика: поля перемещений и напряжений
    disp = np.zeros((len(coords), 3))
    stress = np.zeros((len(coords), 6))
    for b in blocks:
        if b["name"] == "DISP":
            for nid, vals in b["data"].items():
                disp[id2idx[nid]] = vals[:3]
        elif b["name"] == "STRESS":
            for nid, vals in b["data"].items():
                stress[id2idx[nid]] = vals[:6]

    vm = _von_mises(stress)
    umag = np.linalg.norm(disp, axis=1)
    result["sigma_total"] = float(vm.max())
    result["sigma_p95"] = float(np.percentile(vm, 95))
    result["max_disp_mm"] = float(umag.max())
    result["viz_field"] = vm
    result["disp_field_mm"] = umag
    # Печать RF в ccx не включает внешние силы, приложенные к закреплённым узлам
    # (вес самой зоны закрепления). Добавляем консистентную поправку C3D10:
    # для тетраэдра 2-го порядка углы получают -V/20, середины рёбер +V/5.
    rf = np.array(parse_static_dat(dat_text), dtype=float)
    conn_idx = np.searchsorted(node_ids, fem["tet_conn"])
    p = coords[conn_idx[:, :4]]
    v_e = np.abs(np.einsum("ij,ij->i",
                           np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                           p[:, 3] - p[:, 0])) / 6.0
    w = np.zeros(len(coords))
    np.add.at(w, conn_idx[:, :4].ravel(), np.repeat(-v_e / 20.0, 4))
    np.add.at(w, conn_idx[:, 4:].ravel(), np.repeat(v_e / 5.0, 6))
    rho_t = rho * 1e-12  # т/мм³
    g_vec = np.array(meta["g_vec_mm_s2"], dtype=float)
    f_ext_fixed = rho_t * w[meta["fixed_idx"]].sum() * g_vec  # Н
    result["reactions_n"] = [float(x) for x in (rf - f_ext_fixed)]
    result["first_frequency_hz"] = None
    return result


# ---------------------------------------------------------------------------
# Построение сетки в отдельном процессе (gmsh в потоке Streamlit -> segfault)
# ---------------------------------------------------------------------------

def build_fem_mesh_subprocess(step_bytes, mesh_size_mm=8.0):
    """Как build_fem_mesh, но запускает gmsh в дочернем python-процессе."""
    import sys

    with tempfile.NamedTemporaryFile(suffix=".stp", delete=False) as f:
        f.write(step_bytes)
        step_path = f.name
    out_path = step_path + ".npz"
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--mesh",
             step_path, out_path, str(float(mesh_size_mm))],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            tail = (proc.stderr or proc.stdout or "нет вывода").strip().splitlines()[-5:]
            raise RuntimeError("генерация сетки (gmsh): " + " | ".join(tail))
        data = np.load(out_path, allow_pickle=False)
        fem = {k: data[k] for k in data.files}
        fem["volume_mm3"] = float(fem["volume_mm3"])
        fem["n_nodes"] = int(fem["n_nodes"])
        fem["n_elements"] = int(fem["n_elements"])
        return fem
    finally:
        for p in (step_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 5 and sys.argv[1] == "--mesh":
        with open(sys.argv[2], "rb") as fh:
            mesh_dict = build_fem_mesh(fh.read(), mesh_size_mm=float(sys.argv[4]))
        np.savez_compressed(sys.argv[3], **mesh_dict)
