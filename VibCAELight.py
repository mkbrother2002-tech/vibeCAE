# VibCAELight — облегчённая версия VibeCAE.
# Только статический расчёт; область приложения нагрузки выбирается мышью
# на модели (точка / ребро / грань), как в ANSYS.
# Запуск: streamlit run VibCAELight.py

import os
import tempfile
from collections import defaultdict
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import trimesh

try:
    import fem_solver
    FEM_AVAILABLE = fem_solver.fem_available()
except Exception:
    fem_solver = None
    FEM_AVAILABLE = False

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    plotly_events = None
    PLOTLY_EVENTS_AVAILABLE = False

st.set_page_config(page_title="VibCAELight — статический расчёт", layout="wide")

# ---------------------------------------------------------------------------
# Справочные данные (как в VibeCAE)
# ---------------------------------------------------------------------------

MATERIALS_GOST = {
    "Сталь 08Х18Н10Т (Аустенитная)": {
        "yield_strength": 220, "elastic_modulus": 195, "density": 7900, "poisson": 0.30, "thermal_expansion": 16.6e-6,
        "yield_temp_curve": {20: 220, 100: 205, 200: 185, 300: 165, 350: 155, 450: 145, 600: 120},
        "desc": "Применяется в корпусных элементах ИЗК",
    },
    "Сталь 12Х18Н10Т": {
        "yield_strength": 196, "elastic_modulus": 198, "density": 7900, "poisson": 0.30, "thermal_expansion": 16.6e-6,
        "yield_temp_curve": {20: 196, 100: 177, 200: 157, 300: 137, 350: 132, 450: 125, 600: 105},
        "desc": "Высокая коррозионная стойкость",
    },
    "Сталь 20": {
        "yield_strength": 245, "elastic_modulus": 200, "density": 7850, "poisson": 0.30, "thermal_expansion": 12.0e-6,
        "yield_temp_curve": {20: 245, 100: 230, 200: 215, 300: 175, 400: 140, 500: 100},
        "desc": "Общего назначения для неответственных узлов",
    },
    "Сталь 09Г2С": {
        "yield_strength": 345, "elastic_modulus": 205, "density": 7850, "poisson": 0.30, "thermal_expansion": 12.0e-6,
        "yield_temp_curve": {20: 345, 100: 315, 200: 280, 300: 235, 400: 200, 500: 150},
        "desc": "Низколегированная конструкционная сталь",
    },
    "Сталь 15Х5М": {
        "yield_strength": 280, "elastic_modulus": 210, "density": 7750, "poisson": 0.30, "thermal_expansion": 11.5e-6,
        "yield_temp_curve": {20: 280, 100: 270, 200: 255, 300: 235, 400: 210, 500: 180},
        "desc": "Жаростойкая и коррозионностойкая сталь",
    },
    "Сплав ХН78Т (Жаропрочный)": {
        "yield_strength": 350, "elastic_modulus": 210, "density": 8400, "poisson": 0.31, "thermal_expansion": 12.9e-6,
        "yield_temp_curve": {20: 350, 300: 315, 500: 290, 700: 250, 800: 200},
        "desc": "Для высокотемпературных узлов оборудования",
    },
    "Титан ВТ6": {
        "yield_strength": 830, "elastic_modulus": 114, "density": 4430, "poisson": 0.34, "thermal_expansion": 8.6e-6,
        "yield_temp_curve": {20: 830, 100: 770, 200: 680, 300: 590, 400: 480},
        "desc": "Легкий высокопрочный сплав для ответственных узлов",
    },
    "Алюминий АМг6": {
        "yield_strength": 275, "elastic_modulus": 70, "density": 2640, "poisson": 0.33, "thermal_expansion": 24.0e-6,
        "yield_temp_curve": {20: 275, 100: 250, 150: 210, 200: 150, 250: 90},
        "desc": "Лёгкий конструкционный сплав",
    },
    "Бронза БрАЖ9-4": {
        "yield_strength": 280, "elastic_modulus": 105, "density": 7500, "poisson": 0.34, "thermal_expansion": 16.2e-6,
        "yield_temp_curve": {20: 280, 100: 270, 200: 250, 300: 220},
        "desc": "Антикоррозионный сплав для трущихся узлов",
    },
}

SAFETY_NORMS = {
    "ПНАЭ Г-7-002-86 (оборудование ИЯУ, НУЭ)": {"n_yield": 1.5, "desc": "Запас по пределу текучести для нормальных условий эксплуатации."},
    "ГОСТ 34233.1 (сосуды и аппараты)": {"n_yield": 1.5, "desc": "Общий коэффициент запаса по пределу текучести."},
    "Общемашиностроительный": {"n_yield": 1.4, "desc": "Типовой запас для неответственных конструкций."},
    "Пользовательский": {"n_yield": None, "desc": "Коэффициент запаса задаётся вручную."},
}

GRAVITY_MS2 = 9.81

CONSTRAINT_FACE_OPTIONS = {
    "Нижняя грань (Z min)": (2, "min"),
    "Верхняя грань (Z max)": (2, "max"),
    "Грань X min": (0, "min"),
    "Грань X max": (0, "max"),
    "Грань Y min": (1, "min"),
    "Грань Y max": (1, "max"),
}

PICK_MODES = {
    "Грань (поверхность)": "face",
    "Ребро": "edge",
    "Точка (вершина)": "point",
}

PICK_ANGLE_DEG = 30.0  # порог двугранного угла для сегментации граней/рёбер


def yield_strength_at_temp(material_data, temperature):
    curve = material_data.get("yield_temp_curve")
    base = float(material_data["yield_strength"])
    if not curve:
        return base
    temps = np.array(sorted(curve.keys()), dtype=float)
    values = np.array([curve[t] for t in sorted(curve.keys())], dtype=float)
    return float(np.interp(float(temperature), temps, values))


# ---------------------------------------------------------------------------
# Геометрия: загрузка STEP/STL и вспомогательные функции (как в VibeCAE)
# ---------------------------------------------------------------------------

def load_step_to_trimesh(file_bytes, linear_deflection=0.5):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    step_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as fh:
            fh.write(file_bytes)
            step_path = fh.name

        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != IFSelect_RetDone:
            return None
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return None

        BRepMesh_IncrementalMesh(shape, max(float(linear_deflection), 0.01))

        verts, faces_idx, current_offset = [], [], 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            explorer.Next()
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, loc)
            if triangulation is None:
                continue
            trsf = loc.Transformation()
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i).Transformed(trsf)
                verts.append((pnt.X(), pnt.Y(), pnt.Z()))
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for i in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(i).Get()
                if reversed_face:
                    n2, n3 = n3, n2
                faces_idx.append((current_offset + n1 - 1, current_offset + n2 - 1, current_offset + n3 - 1))
            current_offset += triangulation.NbNodes()

        if not verts or not faces_idx:
            return None
        mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces_idx), process=True)
        return mesh if len(mesh.faces) > 0 else None
    except Exception as e:
        st.warning(f"Не удалось прочитать STEP: {e}")
        return None
    finally:
        if step_path and os.path.exists(step_path):
            try:
                os.unlink(step_path)
            except OSError:
                pass


def get_region_mask(mesh, load_params):
    coords = mesh.vertices[:, :3]
    if load_params.get("region_mode") != "Точка с радиусом":
        return np.ones(len(coords), dtype=bool)
    selected_points = load_params.get("selection_points", [])
    radius = float(load_params.get("selection_radius", 1.0))
    if selected_points:
        pts = np.asarray(selected_points, dtype=float).reshape(-1, 3)
        mask = np.zeros(len(coords), dtype=bool)
        for i in range(0, len(pts), 40):  # порциями, чтобы не раздувать память
            chunk = pts[i:i + 40]
            dist = np.linalg.norm(coords[:, None, :] - chunk[None, :, :], axis=2)
            mask |= np.any(dist <= radius, axis=1)
        if np.any(mask):
            return mask
    return np.ones(len(coords), dtype=bool)


def get_region_area_mm2(mesh, load_params):
    mask = get_region_mask(mesh, load_params)
    if not np.any(mask):
        return 0.0
    face_mask = mask[mesh.faces].all(axis=1)
    if not np.any(face_mask):
        return 0.0
    return float(mesh.area_faces[face_mask].sum())


def get_region_centroid(mesh, mask):
    coords = mesh.vertices[:, :3]
    if not np.any(mask):
        return coords.mean(axis=0)
    face_mask = mask[mesh.faces].all(axis=1)
    if np.any(face_mask):
        centers = mesh.triangles_center[face_mask]
        areas = mesh.area_faces[face_mask]
        total = float(areas.sum())
        if total > 1e-12:
            return (centers * areas[:, None]).sum(axis=0) / total
    return coords[mask].mean(axis=0)


def get_constraint_zone(mesh, face_option, zone_frac=0.05):
    coords = mesh.vertices[:, :3]
    axis_idx, side = CONSTRAINT_FACE_OPTIONS.get(face_option, (2, "min"))
    lo = float(coords[:, axis_idx].min())
    hi = float(coords[:, axis_idx].max())
    depth = max((hi - lo) * float(zone_frac), 1e-6)
    if side == "min":
        mask = coords[:, axis_idx] <= lo + depth
    else:
        mask = coords[:, axis_idx] >= hi - depth
    if not np.any(mask):
        mask = np.ones(len(coords), dtype=bool)
    return mask, get_region_centroid(mesh, mask)


def get_mass_properties(mesh, density_kg_m3):
    coords = mesh.vertices[:, :3]
    extents = coords.max(axis=0) - coords.min(axis=0)
    is_exact = bool(mesh.is_watertight)
    try:
        volume_mm3 = float(abs(mesh.volume))
    except Exception:
        volume_mm3 = 0.0
    if volume_mm3 <= 1e-9:
        try:
            volume_mm3 = float(abs(mesh.convex_hull.volume))
        except Exception:
            volume_mm3 = float(np.prod(np.maximum(extents, 1e-6)))
        is_exact = False
    try:
        com = np.asarray(mesh.center_mass if is_exact else mesh.centroid, dtype=float)
    except Exception:
        com = coords.mean(axis=0)
    return {
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_mm3 / 1000.0,
        "mass_kg": float(density_kg_m3) * volume_mm3 * 1e-9,
        "center_of_mass": com,
        "extents_mm": extents,
        "is_exact": is_exact,
    }


def get_direction_vector(direction):
    m = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
         "+Z": (0, 0, 1), "-Z": (0, 0, -1)}
    return np.array(m.get(direction, (0, 0, -1)), dtype=float)


def _extent_along(coords, direction):
    proj = coords @ direction
    return float(proj.max() - proj.min())


# ---------------------------------------------------------------------------
# Аналитический статический решатель (экспресс-оценка, как уровень 1 VibeCAE)
# ---------------------------------------------------------------------------

def solve_static(mesh, material_data, params):
    coords = mesh.vertices[:, :3]
    density = float(material_data.get("density", 7850.0))
    mass_props = get_mass_properties(mesh, density)
    temperature = float(params.get("temperature", 20))
    sigma_yield_t = yield_strength_at_temp(material_data, temperature)

    total_mass_kg = mass_props["mass_kg"] + max(float(params.get("contents_mass_kg", 0.0)), 0.0)
    direction_label = params.get("direction", "-Z")
    d = get_direction_vector(direction_label)

    force_terms = []
    force_vec = np.zeros(3)
    if params.get("include_gravity", True) and total_mass_kg > 0:
        f_g = total_mass_kg * GRAVITY_MS2
        force_vec += f_g * np.array([0.0, 0.0, -1.0])
        force_terms.append({"name": "Вес (модель + содержимое)", "value_n": f_g, "direction": "-Z"})
    point_force = float(params.get("point_force_n", 0.0))
    if abs(point_force) > 1e-9:
        force_vec += point_force * d
        force_terms.append({"name": "Сосредоточенная сила", "value_n": point_force, "direction": direction_label})
    pressure = float(params.get("pressure_mpa", 0.0))
    region_area_mm2 = get_region_area_mm2(mesh, params)
    if pressure > 1e-9 and region_area_mm2 > 0:
        f_p = pressure * region_area_mm2
        force_vec += f_p * d
        force_terms.append({"name": f"Давление {pressure:g} МПа × {region_area_mm2:,.0f} мм²",
                            "value_n": f_p, "direction": direction_label})

    force_total = float(np.linalg.norm(force_vec))
    constraint_mask, constraint_centroid = get_constraint_zone(
        mesh, params.get("constraint_face"), params.get("constraint_zone_frac", 0.05))
    region_mask = get_region_mask(mesh, params)
    load_center = get_region_centroid(mesh, region_mask)

    volume_mm3 = mass_props["volume_mm3"]
    sigma_membrane = 0.0
    sigma_bending = 0.0
    if force_total > 1e-9 and volume_mm3 > 0:
        force_dir = force_vec / force_total
        length_along_force = max(_extent_along(coords, force_dir), 1e-3)
        area_section_mm2 = max(volume_mm3 / length_along_force, 1e-6)
        sigma_membrane = force_total / area_section_mm2

        lever_vec = load_center - constraint_centroid
        moment_nmm = float(np.linalg.norm(np.cross(lever_vec, force_vec)))
        if moment_nmm > 1e-9:
            lever_len = float(np.linalg.norm(lever_vec))
            beam_dir = lever_vec / max(lever_len, 1e-9)
            length_beam = max(_extent_along(coords, beam_dir), 1e-3)
            area_beam_mm2 = max(volume_mm3 / length_beam, 1e-6)
            f_perp = force_vec - float(force_vec @ beam_dir) * beam_dir
            f_perp_norm = float(np.linalg.norm(f_perp))
            if f_perp_norm > 1e-9:
                h_dir = f_perp / f_perp_norm
            else:
                ref = np.array([0.0, 0.0, 1.0]) if abs(beam_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                h_dir = np.cross(beam_dir, ref)
                h_dir /= max(np.linalg.norm(h_dir), 1e-9)
            h_mm = max(_extent_along(coords, h_dir), 1e-3)
            section_modulus_mm3 = max(area_beam_mm2 * h_mm / 6.0, 1e-6)
            sigma_bending = moment_nmm / section_modulus_mm3

    return {
        "analysis_type": "Статический",
        "mass_props": mass_props,
        "total_mass_kg": total_mass_kg,
        "force_terms": force_terms,
        "force_total_n": force_total,
        "sigma_membrane": sigma_membrane,
        "sigma_bending": sigma_bending,
        "sigma_total": sigma_membrane + sigma_bending,
        "sigma_yield_t": sigma_yield_t,
        "load_center": load_center,
        "constraint_centroid": constraint_centroid,
        "region_area_mm2": region_area_mm2,
    }


def build_display_stress_field(mesh, result):
    """Качественное распределение: мембранная часть равномерна, изгибная растёт к заделке."""
    coords = mesh.vertices[:, :3]
    dist_con = np.linalg.norm(coords - result["constraint_centroid"], axis=1)
    d_min = float(dist_con.min())
    span = max(float(dist_con.max()) - d_min, 1e-6)
    bend_shape = 1.0 - (dist_con - d_min) / span
    return result["sigma_membrane"] + result["sigma_bending"] * bend_shape


def sample_for_display(coords, values=None, max_points=60_000):
    n = len(coords)
    if n <= max_points:
        return coords, values
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return coords[idx], (values[idx] if values is not None else None)


# ---------------------------------------------------------------------------
# Интерактивный выбор мышью: сегментация модели на грани / рёбра / вершины
# ---------------------------------------------------------------------------

class _UnionFind:
    """Система непересекающихся множеств (без зависимости от scipy)."""

    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        p = self.parent
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri

    def labels(self):
        roots = np.array([self.find(i) for i in range(len(self.parent))])
        _, ids = np.unique(roots, return_inverse=True)
        return ids


def _connected_labels(pairs, node_count):
    """Метки связных компонент графа (аналог trimesh.graph.connected_component_labels)."""
    uf = _UnionFind(node_count)
    for a, b in pairs:
        uf.union(int(a), int(b))
    return uf.labels()


def _group_edge_chains(fe_edges, fe_face_pairs, face_labels):
    """Объединяет острые рёбра-сегменты в цепочки (одно «ребро CAD-модели»):
    сегменты одной цепочки разделяют вершину и одну и ту же пару гладких граней."""
    m = len(fe_edges)
    uf = _UnionFind(m)
    keys = [tuple(sorted((int(face_labels[a]), int(face_labels[b])))) for a, b in fe_face_pairs]
    vert_map = defaultdict(list)
    for i, (v0, v1) in enumerate(fe_edges):
        vert_map[(keys[i], int(v0))].append(i)
        vert_map[(keys[i], int(v1))].append(i)
    for lst in vert_map.values():
        for j in lst[1:]:
            uf.union(lst[0], j)
    chain_ids = uf.labels()
    return chain_ids, (int(chain_ids.max()) + 1 if m else 0)


def build_pick_data(mesh, max_faces=25_000, angle_deg=PICK_ANGLE_DEG):
    """Готовит данные для выбора мышью: упрощённая сетка, гладкие грани,
    цепочки острых рёбер, вершины."""
    pick = mesh
    if len(mesh.faces) > max_faces:
        try:
            pick = mesh.simplify_quadric_decimation(face_count=max_faces)
            if len(pick.faces) == 0:
                pick = mesh
        except BaseException:
            pick = mesh

    coords = np.asarray(pick.vertices, dtype=float)
    faces = np.asarray(pick.faces)
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    thr = np.radians(angle_deg)

    adj = pick.face_adjacency if len(pick.face_adjacency) else np.zeros((0, 2), dtype=int)
    ang = pick.face_adjacency_angles if len(adj) else np.zeros(0)

    # Гладкие грани: связные компоненты по смежности с малым двугранным углом
    smooth_pairs = adj[ang < thr] if len(adj) else np.zeros((0, 2), dtype=int)
    face_labels = _connected_labels(smooth_pairs, len(faces))
    n_regions = int(face_labels.max()) + 1 if len(face_labels) else 0
    region_areas = np.bincount(face_labels, weights=pick.area_faces, minlength=n_regions)

    # Острые рёбра и их цепочки
    sharp = ang >= thr
    fe_edges = pick.face_adjacency_edges[sharp] if len(adj) else np.zeros((0, 2), dtype=int)
    fe_pairs = adj[sharp] if len(adj) else np.zeros((0, 2), dtype=int)
    if len(fe_edges):
        edge_chain_ids, n_chains = _group_edge_chains(fe_edges, fe_pairs, face_labels)
    else:
        edge_chain_ids, n_chains = np.zeros(0, dtype=int), 0
    seg_len = (np.linalg.norm(coords[fe_edges[:, 0]] - coords[fe_edges[:, 1]], axis=1)
               if len(fe_edges) else np.zeros(0))
    chain_lengths = np.bincount(edge_chain_ids, weights=seg_len, minlength=n_chains) if n_chains else np.zeros(0)

    return {
        "mesh": pick,
        "coords": coords,
        "faces": faces,
        "tri_centers": np.asarray(pick.triangles_center, dtype=float),
        "tri_areas": np.asarray(pick.area_faces, dtype=float),
        "face_normals": np.asarray(pick.face_normals, dtype=float),
        "vertex_normals": np.asarray(pick.vertex_normals, dtype=float),
        "face_labels": face_labels,
        "n_regions": n_regions,
        "region_areas": region_areas,
        "fe_edges": fe_edges,
        "edge_chain_ids": edge_chain_ids,
        "n_chains": n_chains,
        "chain_lengths": chain_lengths,
        "diag": diag,
    }


def _sample_indices(n, limit):
    if n <= limit:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).astype(int)


def _l(a):
    """numpy -> list: компонент plotly_events собран со старым plotly.js,
    который не понимает бинарную сериализацию массивов (bdata) plotly 6."""
    return np.asarray(a).tolist()


def build_pick_figure(pd_, mode, selection):
    """Модель + «активный» слой для кликов. Возвращает (fig, номер кликабельной кривой,
    массив идентификаторов: клик по точке i -> сущность pick_ids[i])."""
    coords, faces, diag = pd_["coords"], pd_["faces"], pd_["diag"]
    eps = 0.004 * max(diag, 1e-6)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=_l(coords[:, 0]), y=_l(coords[:, 1]), z=_l(coords[:, 2]),
        i=_l(faces[:, 0]), j=_l(faces[:, 1]), k=_l(faces[:, 2]),
        color="#c3c9d1", flatshading=True, hoverinfo="skip",
        lighting=dict(ambient=0.55, diffuse=0.85), showlegend=False,
    ))

    pick_ids = np.zeros(0, dtype=int)
    if mode == "point":
        idx = _sample_indices(len(coords), 5000)
        pts = coords[idx] + pd_["vertex_normals"][idx] * eps
        text = [f"Вершина {int(i)}: ({coords[i, 0]:.1f}; {coords[i, 1]:.1f}; {coords[i, 2]:.1f})" for i in idx]
        fig.add_trace(go.Scatter3d(
            x=_l(pts[:, 0]), y=_l(pts[:, 1]), z=_l(pts[:, 2]), mode="markers",
            marker=dict(size=2.6, color="rgba(31, 88, 160, 0.85)"),
            hovertext=text, hoverinfo="text", showlegend=False,
        ))
        pick_curve = 1
        pick_ids = idx
    elif mode == "edge":
        fe_edges, chain_ids = pd_["fe_edges"], pd_["edge_chain_ids"]
        if len(fe_edges):
            # линии всех острых рёбер (некликабельные)
            xl = np.empty(3 * len(fe_edges)); yl = np.empty(3 * len(fe_edges)); zl = np.empty(3 * len(fe_edges))
            for k in range(3):
                arr = (xl, yl, zl)[k]
                arr[0::3] = coords[fe_edges[:, 0], k]
                arr[1::3] = coords[fe_edges[:, 1], k]
                arr[2::3] = np.nan
            fig.add_trace(go.Scatter3d(
                x=_l(xl), y=_l(yl), z=_l(zl), mode="lines",
                line=dict(color="rgba(25, 60, 110, 0.9)", width=4),
                hoverinfo="skip", showlegend=False,
            ))
            # кликабельные маркеры — середины сегментов
            idx = _sample_indices(len(fe_edges), 6000)
            mids = 0.5 * (coords[fe_edges[idx, 0]] + coords[fe_edges[idx, 1]])
            nrm = 0.5 * (pd_["vertex_normals"][fe_edges[idx, 0]] + pd_["vertex_normals"][fe_edges[idx, 1]])
            mids = mids + nrm * eps
            text = [f"Ребро {int(chain_ids[i]) + 1} · L ≈ {pd_['chain_lengths'][chain_ids[i]]:.1f} мм" for i in idx]
            fig.add_trace(go.Scatter3d(
                x=_l(mids[:, 0]), y=_l(mids[:, 1]), z=_l(mids[:, 2]), mode="markers",
                marker=dict(size=3.5, color="rgba(31, 88, 160, 0.6)"),
                hovertext=text, hoverinfo="text", showlegend=False,
            ))
            pick_curve = 2
            pick_ids = chain_ids[idx]
        else:
            pick_curve = None
    else:  # face
        labels = pd_["face_labels"]
        centers = pd_["tri_centers"]
        idx = _sample_indices(len(centers), 6000)
        # гарантируем хотя бы одну точку на каждую грань
        first_of_label = np.unique(labels, return_index=True)[1]
        idx = np.unique(np.concatenate([idx, first_of_label]))
        pts = centers[idx] + pd_["face_normals"][idx] * eps
        text = [f"Грань {int(labels[i]) + 1} · S ≈ {pd_['region_areas'][labels[i]]:,.0f} мм²" for i in idx]
        fig.add_trace(go.Scatter3d(
            x=_l(pts[:, 0]), y=_l(pts[:, 1]), z=_l(pts[:, 2]), mode="markers",
            marker=dict(size=3, color="rgba(31, 88, 160, 0.45)"),
            hovertext=text, hoverinfo="text", showlegend=False,
        ))
        pick_curve = 1
        pick_ids = labels[idx]

    # Подсветка текущего выбора (красным)
    if selection and selection.get("mode") == mode:
        sid = int(selection["id"])
        if mode == "point" and sid < len(coords):
            p = coords[sid] + pd_["vertex_normals"][sid] * eps * 2
            fig.add_trace(go.Scatter3d(
                x=[float(p[0])], y=[float(p[1])], z=[float(p[2])], mode="markers",
                marker=dict(size=8, color="#d0021b", symbol="circle"),
                hoverinfo="skip", showlegend=False,
            ))
        elif mode == "edge" and pd_["n_chains"] and sid < pd_["n_chains"]:
            seg = pd_["fe_edges"][pd_["edge_chain_ids"] == sid]
            xl = np.empty(3 * len(seg)); yl = np.empty(3 * len(seg)); zl = np.empty(3 * len(seg))
            for k in range(3):
                arr = (xl, yl, zl)[k]
                arr[0::3] = coords[seg[:, 0], k]
                arr[1::3] = coords[seg[:, 1], k]
                arr[2::3] = np.nan
            fig.add_trace(go.Scatter3d(
                x=_l(xl), y=_l(yl), z=_l(zl), mode="lines",
                line=dict(color="#d0021b", width=9),
                hoverinfo="skip", showlegend=False,
            ))
        elif mode == "face" and sid < pd_["n_regions"]:
            f_sel = faces[pd_["face_labels"] == sid]
            used = np.unique(f_sel)
            remap = np.full(len(coords), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            v = coords[used] + pd_["vertex_normals"][used] * eps * 0.6
            f = remap[f_sel]
            fig.add_trace(go.Mesh3d(
                x=_l(v[:, 0]), y=_l(v[:, 1]), z=_l(v[:, 2]),
                i=_l(f[:, 0]), j=_l(f[:, 1]), k=_l(f[:, 2]),
                color="#d0021b", opacity=0.85, flatshading=True,
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=560,
        uirevision="keep",  # не сбрасывать камеру при перерисовке
    )
    return fig, pick_curve, pick_ids


def selection_to_region(pd_, selection, point_radius_mm, fem_size_mm):
    """Преобразует выбранную сущность в параметры области нагрузки
    (region_mode «Точка с радиусом»: точки + радиус захвата узлов)."""
    coords = pd_["coords"]
    mode = selection["mode"]
    sid = int(selection["id"])
    if mode == "point":
        pts = coords[[sid]]
        radius = float(point_radius_mm)
        desc = f"Точка ({pts[0][0]:.1f}; {pts[0][1]:.1f}; {pts[0][2]:.1f}), R = {radius:g} мм"
    elif mode == "edge":
        seg = pd_["fe_edges"][pd_["edge_chain_ids"] == sid]
        v_idx = np.unique(seg)
        pts = coords[v_idx[_sample_indices(len(v_idx), 60)]]
        seg_len = np.linalg.norm(coords[seg[:, 0]] - coords[seg[:, 1]], axis=1)
        radius = max(1.6 * float(np.median(seg_len)) if len(seg_len) else 1.0,
                     float(point_radius_mm))
        desc = f"Ребро {sid + 1}, L ≈ {pd_['chain_lengths'][sid]:.1f} мм, R = {radius:.1f} мм"
    else:  # face
        in_region = pd_["face_labels"] == sid
        centers = pd_["tri_centers"][in_region]
        areas = pd_["tri_areas"][in_region]
        area_total = float(areas.sum())
        idx = _sample_indices(len(centers), 80)
        pts = centers[idx]
        spacing = np.sqrt(area_total / max(len(pts), 1))
        radius = max(1.5 * spacing, float(fem_size_mm), 0.5)
        desc = f"Грань {sid + 1}, S ≈ {area_total:,.0f} мм², {len(pts)} опорных точек, R = {radius:.1f} мм"
    return {
        "region_mode": "Точка с радиусом",
        "selection_points": [[float(x), float(y), float(z)] for x, y, z in pts],
        "selection_radius": float(radius),
    }, desc


def build_setup_figure(pd_, params):
    """Схема нагружения на модели: область нагрузки (оранжевый), зона закрепления (красный),
    стрелки силы/давления и собственного веса."""
    pm = pd_["mesh"]
    coords, faces, diag = pd_["coords"], pd_["faces"], pd_["diag"]

    fixed_mask, fixed_center = get_constraint_zone(
        pm, params.get("constraint_face"), params.get("constraint_zone_frac", 0.05))
    load_mask = get_region_mask(pm, params)
    load_center = get_region_centroid(pm, load_mask)

    inten = np.zeros(len(coords))
    inten[load_mask] = 1.0
    inten[fixed_mask] = 2.0  # закрепление поверх области нагрузки

    colorscale = [
        [0.0, "#c3c9d1"], [1 / 3, "#c3c9d1"],
        [1 / 3, "#f5a623"], [2 / 3, "#f5a623"],
        [2 / 3, "#d0021b"], [1.0, "#d0021b"],
    ]
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=inten, cmin=0.0, cmax=2.0,
        colorscale=colorscale, showscale=False,
        flatshading=True, hoverinfo="skip",
    ))

    arrow_len = max(diag * 0.30, 1.0)

    def _surface_tip(center, d, pts):
        """Сдвигает точку вдоль −d на поверхность тела, чтобы стрелка не пряталась внутри модели."""
        center = np.asarray(center, dtype=float)
        proj = pts @ d
        return center + d * (float(proj.min()) - float(center @ d))

    def _add_arrow(tip, d, color, label):
        tip = np.asarray(tip, dtype=float)
        d = np.asarray(d, dtype=float)
        tail = tip - d * arrow_len
        fig.add_trace(go.Scatter3d(
            x=[tail[0], tip[0]], y=[tail[1], tip[1]], z=[tail[2], tip[2]],
            mode="lines+text", text=[label, ""], textposition="top center",
            textfont=dict(color=color, size=13),
            line=dict(color=color, width=7), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[d[0]], v=[d[1]], w=[d[2]],
            sizemode="absolute", sizeref=arrow_len * 0.25, anchor="tip",
            colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip",
        ))

    if float(params.get("point_force_n", 0.0)) > 0 or float(params.get("pressure_mpa", 0.0)) > 0:
        d_load = get_direction_vector(params.get("direction", "-Z"))
        load_pts = coords[load_mask] if np.any(load_mask) else coords
        _add_arrow(_surface_tip(load_center, d_load, load_pts), d_load, "#e07b00", "нагрузка")
    if params.get("include_gravity", True):
        d_g = np.array([0.0, 0.0, -1.0])
        g_tip = _surface_tip(pm.centroid, d_g, coords)
        # разводим со стрелкой нагрузки, если они совпадают
        g_tip = g_tip + np.array([0.12, 0.0, 0.0]) * diag
        _add_arrow(g_tip, d_g, "#2b6cb0", "вес")

    # подпись зоны закрепления — с отступом наружу от грани
    axis_idx, side = CONSTRAINT_FACE_OPTIONS.get(params.get("constraint_face"), (2, "min"))
    label_pos = np.asarray(fixed_center, dtype=float).copy()
    label_pos[axis_idx] += (-1.0 if side == "min" else 1.0) * diag * 0.08
    fig.add_trace(go.Scatter3d(
        x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
        mode="text", text=["закрепление"], textposition="middle center",
        textfont=dict(color="#d0021b", size=13), hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=440,
        uirevision="setup",
    )
    return fig


# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------

st.title("VibCAELight")
st.caption("Облегчённая версия VibeCAE: только статический расчёт. Область приложения "
           "нагрузки выбирается мышью на модели — грань, ребро или точка (как в ANSYS).")

with st.sidebar:
    selected_material = st.selectbox("Материал конструкции (ГОСТ)", list(MATERIALS_GOST.keys()))
    material_data = MATERIALS_GOST[selected_material]
    st.caption(f"_{material_data['desc']}_")
    st.caption(f"σт(20 °C) = {material_data['yield_strength']} МПа | E = {material_data['elastic_modulus']} ГПа | "
               f"ρ = {material_data['density']} кг/м³")

    st.markdown("---")
    norm_name = st.selectbox("Нормативный коэффициент запаса", list(SAFETY_NORMS.keys()))
    if SAFETY_NORMS[norm_name]["n_yield"] is None:
        norm_coef = st.number_input("Коэффициент запаса по σт", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    else:
        norm_coef = SAFETY_NORMS[norm_name]["n_yield"]
        st.caption(SAFETY_NORMS[norm_name]["desc"])
    st.caption(f"Допускаемое напряжение: [σ] = σт(T) / {norm_coef:g}")

    st.markdown("---")
    if FEM_AVAILABLE:
        st.markdown("**Решатель: МКЭ (CalculiX)**")
        fem_mesh_size = st.slider("Размер КЭ (мм)", 2.0, 20.0, 8.0, 0.5,
                                  help="Тетраэдры 2-го порядка C3D10. Требуется STEP-файл.")
    else:
        fem_mesh_size = 8.0
        st.markdown("*Решатель: экспресс-оценка (CalculiX не найден)*")

# --- 1. Загрузка модели ---
st.header("1. Модель")
uploaded_file = st.file_uploader(
    "Перетащите CAD-модель (.stp, .step, .stl)",
    type=["stp", "step", "stl"],
    key="light_uploader",
)

if uploaded_file and st.session_state.get("light_name") != uploaded_file.name:
    try:
        if uploaded_file.name.lower().endswith(".stl"):
            mesh = trimesh.load(BytesIO(uploaded_file.getvalue()), file_type="stl", force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.to_mesh() if hasattr(mesh, "to_mesh") else mesh.dump(concatenate=True)
            st.session_state["light_mesh"] = mesh
            st.session_state["light_step_bytes"] = None
        else:
            step_bytes = uploaded_file.getvalue()
            mesh = load_step_to_trimesh(step_bytes, linear_deflection=0.5)
            st.session_state["light_mesh"] = mesh
            st.session_state["light_step_bytes"] = step_bytes if mesh is not None else None
        st.session_state["light_name"] = uploaded_file.name
        st.session_state.pop("pick_selection", None)
        st.session_state.pop("light_pick_key", None)
        st.session_state.pop("light_result", None)
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")

mesh = st.session_state.get("light_mesh")
if mesh is None:
    st.info("Загрузите модель, чтобы выбрать область приложения нагрузки и выполнить расчёт.")
    st.stop()

density = float(material_data["density"])
mass_props = get_mass_properties(mesh, density)
ext = mass_props["extents_mm"]
st.caption(f"Файл: {st.session_state.get('light_name')} | Габариты: {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} мм | "
           f"Масса: {mass_props['mass_kg']:.2f} кг | Треугольников: {len(mesh.faces):,}")

# --- 2. Выбор области приложения нагрузки мышью ---
st.header("2. Область приложения нагрузки")

pick_key = (st.session_state.get("light_name"), PICK_ANGLE_DEG)
if st.session_state.get("light_pick_key") != pick_key:
    with st.spinner("Подготовка модели к интерактивному выбору..."):
        st.session_state["light_pick_data"] = build_pick_data(mesh)
        st.session_state["light_pick_key"] = pick_key
pick_data = st.session_state["light_pick_data"]

col_mode, col_clear = st.columns([3, 1])
with col_mode:
    mode_label = st.radio("Режим выбора", list(PICK_MODES.keys()), horizontal=True, key="pick_mode")
mode = PICK_MODES[mode_label]
with col_clear:
    st.write("")
    if st.button("Сбросить выбор", width="stretch"):
        st.session_state.pop("pick_selection", None)
        st.session_state["pick_nonce"] = st.session_state.get("pick_nonce", 0) + 1
        st.rerun()

selection = st.session_state.get("pick_selection")
if selection and selection.get("mode") != mode:
    selection = None  # смена режима сбрасывает выбор

point_radius_mm = st.slider(
    "Радиус зоны захвата (мм)", 0.5, max(pick_data["diag"] * 0.2, 2.0),
    float(st.session_state.get("pick_radius", max(round(pick_data["diag"] * 0.02, 1), 1.0))),
    0.5, key="pick_radius",
    help="Для точки и ребра: узлы КЭ-сетки в пределах радиуса от выбранной сущности войдут в область нагрузки.",
)

fig, pick_curve, pick_ids = build_pick_figure(pick_data, mode, selection)
if PLOTLY_EVENTS_AVAILABLE:
    # st.plotly_chart(on_select=...) не поддерживает клики по 3D-сценам —
    # используем компонент plotly_events (событие plotly_click работает в 3D).
    clicked_points = plotly_events(
        fig, click_event=True, select_event=False, hover_event=False,
        override_height=560, override_width="100%",
        key=f"pick_chart_{mode}_{st.session_state.get('pick_nonce', 0)}",
    ) or []
else:
    st.plotly_chart(fig, key="pick_chart")
    clicked_points = []
    st.warning("Компонент streamlit-plotly-events не установлен — выбор мышью недоступен. "
               "Установите: pip install streamlit-plotly-events")

if clicked_points and pick_curve is not None:
    p = clicked_points[-1]
    if p.get("curveNumber") == pick_curve:
        pn = p.get("pointNumber", p.get("pointIndex"))
        if pn is not None and 0 <= int(pn) < len(pick_ids):
            new_sel = {"mode": mode, "id": int(pick_ids[int(pn)])}
            if new_sel != st.session_state.get("pick_selection"):
                st.session_state["pick_selection"] = new_sel
                st.rerun()

selection = st.session_state.get("pick_selection")
if selection and selection.get("mode") == mode:
    region_params, region_desc = selection_to_region(pick_data, selection, point_radius_mm, fem_mesh_size)
    st.success(f"Выбрано: {region_desc}")
else:
    region_params, region_desc = None, None
    st.info("Кликните по синему маркеру на модели: в режиме «Грань» — любую точку поверхности, "
            "в режиме «Ребро» — маркер на линии ребра, в режиме «Точка» — вершину. "
            "Пока ничего не выбрано, нагрузка распределяется по всей поверхности.")

if mode == "edge" and pick_data["n_chains"] == 0:
    st.warning("На модели не найдено острых рёбер (порог 30°). Используйте выбор грани или точки.")

# --- 3. Нагрузки и закрепление ---
st.header("3. Нагрузки и закрепление")
col_load, col_fix = st.columns(2)
with col_load:
    st.subheader("Нагрузка")
    point_force_n = st.number_input("Сосредоточенная сила (Н)", min_value=0.0, value=1000.0, step=100.0)
    pressure_mpa = st.number_input("Давление на область (МПа)", min_value=0.0, value=0.0, step=0.05, format="%.3f")
    direction = st.selectbox("Направление силы/давления", ["-Z", "+Z", "-X", "+X", "-Y", "+Y"])
    include_gravity = st.checkbox("Учитывать собственный вес", value=True)
    contents_mass_kg = st.number_input("Масса содержимого (кг)", min_value=0.0, value=0.0, step=1.0,
                                       help="Прикладывается в области нагрузки.")
    temperature = st.number_input("Температура (°C, для σт(T))", min_value=20.0, max_value=800.0, value=20.0, step=10.0)
with col_fix:
    st.subheader("Закрепление")
    constraint_face = st.selectbox("Грань закрепления", list(CONSTRAINT_FACE_OPTIONS.keys()))
    constraint_zone = st.slider("Глубина зоны закрепления (% габарита)", 1, 20, 5)

params = {
    "analysis_type": "Статический",
    "load_type": "Статическая",
    "direction": direction,
    "temperature": float(temperature),
    "include_gravity": bool(include_gravity),
    "contents_mass_kg": float(contents_mass_kg),
    "point_force_n": float(point_force_n),
    "pressure_mpa": float(pressure_mpa),
    "seismic_g": 0.0,
    "constraint_face": constraint_face,
    "constraint_type": "Жёсткая заделка",
    "constraint_zone_frac": float(constraint_zone) / 100.0,
}
if region_params:
    params.update(region_params)
else:
    params["region_mode"] = "Автоматическая зона"

show_setup = st.toggle("Схема нагружения на модели", value=True,
                       help="Оранжевый — область приложения нагрузки, красный — зона закрепления, "
                            "стрелки — направления силы и собственного веса.")
if show_setup:
    setup_fig = build_setup_figure(pick_data, params)
    st.plotly_chart(setup_fig, key="setup_chart", config={"displayModeBar": False})

use_fem = FEM_AVAILABLE and st.session_state.get("light_step_bytes") is not None
if FEM_AVAILABLE and st.session_state.get("light_step_bytes") is None:
    st.warning("Для МКЭ-расчёта нужен STEP-файл. Для STL выполняется экспресс-оценка.")

# --- 4. Расчёт ---
st.header("4. Статический расчёт")
if st.button("Выполнить расчёт", type="primary", width="stretch"):
    result = None
    if use_fem:
        fem_key = (st.session_state.get("light_name"), float(fem_mesh_size))
        fem_mesh = None
        if st.session_state.get("light_fem_key") == fem_key:
            fem_mesh = st.session_state.get("light_fem_mesh")
        if fem_mesh is None:
            with st.spinner(f"Построение КЭ-сетки (gmsh, {fem_mesh_size:g} мм)..."):
                try:
                    fem_mesh = fem_solver.build_fem_mesh_subprocess(
                        st.session_state["light_step_bytes"], mesh_size_mm=float(fem_mesh_size))
                    st.session_state["light_fem_mesh"] = fem_mesh
                    st.session_state["light_fem_key"] = fem_key
                except Exception as exc:
                    st.error(f"Ошибка построения КЭ-сетки: {exc}")
                    fem_mesh = None
        if fem_mesh is not None:
            with st.spinner("КЭ-расчёт CalculiX..."):
                try:
                    result = fem_solver.solve_scenario_fem(fem_mesh, material_data, params)
                except Exception as exc:
                    st.error(f"Ошибка КЭ-расчёта: {exc}")
    if result is None:
        result = solve_static(mesh, material_data, params)
    st.session_state["light_result"] = (result, params)

stored = st.session_state.get("light_result")
if stored:
    result, run_params = stored
    is_fem = bool(result.get("fem"))
    sigma_total = float(result["sigma_total"])
    sigma_yield_t = float(result["sigma_yield_t"])
    sigma_allow = sigma_yield_t / norm_coef
    safety = sigma_yield_t / sigma_total if sigma_total > 0 else float("inf")

    col_info, col_plot = st.columns([1, 2])
    with col_info:
        if is_fem:
            st.metric("Макс. напряжение (МКЭ, по Мизесу)", f"{sigma_total:.1f} МПа")
            st.caption(f"95-й перцентиль: {result.get('sigma_p95', 0):.1f} МПа | "
                       f"Макс. перемещение: {result.get('max_disp_mm', 0) * 1000:.1f} мкм")
            st.caption(f"КЭ-модель: {result.get('n_nodes', 0):,} узлов, {result.get('n_elements', 0):,} C3D10")
            if result.get("reactions_n"):
                rf = result["reactions_n"]
                st.caption(f"Реакции опор: ({rf[0]:,.0f}; {rf[1]:,.0f}; {rf[2]:,.0f}) Н")
        else:
            st.metric("Макс. напряжение (оценка)", f"{sigma_total:.1f} МПа")
            st.caption(f"σ мембранное: {result['sigma_membrane']:.1f} | σ изгибное: {result['sigma_bending']:.1f} МПа")

        st.caption(f"Суммарная нагрузка: {result['force_total_n']:,.0f} Н | "
                   f"Масса: {result['total_mass_kg']:.1f} кг | "
                   f"Площадь области: {result.get('region_area_mm2', 0):,.0f} мм²")
        for term in result.get("force_terms", []):
            st.caption(f"• {term['name']}: {term['value_n']:,.0f} Н ({term['direction']})")

        st.caption(f"σт({run_params['temperature']:g} °C) = {sigma_yield_t:.0f} МПа | "
                   f"[σ] = {sigma_allow:.1f} МПа (n = {norm_coef:g})")
        safety_text = f"{safety:.2f}" if np.isfinite(safety) else "∞"
        passed = safety >= norm_coef
        st.metric("Запас прочности по σт(T)", safety_text,
                  delta="Пройдён" if passed else "Не пройдён",
                  delta_color="normal" if passed else "inverse")
        if safety < 1.0:
            st.error("Напряжения превышают предел текучести: требуется пересмотр конструкции или нагрузки.")
        elif not passed:
            st.warning(f"Запас ниже требуемого {norm_coef:g}: требуется уточнение.")
        else:
            st.success("Статический сценарий допустим.")

    with col_plot:
        if is_fem and result.get("viz_field") is not None:
            vc, vt = result["viz_coords"], result["viz_tris"]
            fig_r = go.Figure(go.Mesh3d(
                x=vc[:, 0], y=vc[:, 1], z=vc[:, 2],
                i=vt[:, 0], j=vt[:, 1], k=vt[:, 2],
                intensity=result["viz_field"], colorscale="Jet",
                cmin=0.0, cmax=max(sigma_total, 1e-6), showscale=True,
                colorbar=dict(title="σ, МПа"),
                lighting=dict(ambient=0.6, diffuse=0.8),
            ))
            fig_r.update_layout(title="Напряжения по Мизесу (МКЭ)",
                                scene=dict(aspectmode="data"),
                                height=520, margin=dict(l=0, r=0, b=0, t=40))
        else:
            field = build_display_stress_field(mesh, result)
            plot_coords, plot_field = sample_for_display(mesh.vertices[:, :3], field)
            fig_r = go.Figure(go.Scatter3d(
                x=plot_coords[:, 0], y=plot_coords[:, 1], z=plot_coords[:, 2],
                mode="markers",
                marker=dict(size=3, color=plot_field, colorscale="Jet",
                            cmin=0, cmax=max(sigma_total, 1e-6), showscale=True),
            ))
            fig_r.update_layout(title="Оценочное распределение напряжений",
                                scene=dict(aspectmode="data"),
                                height=520, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_r)
        if not is_fem:
            st.caption("Распределение качественное: мембранная часть равномерна, изгибная нарастает к зоне закрепления.")
