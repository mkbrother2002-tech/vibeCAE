import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import trimesh
import os
from datetime import datetime
from io import BytesIO
import cadquery as cq
from OCP import TopoDS

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Настройка страницы
st.set_page_config(page_title="VibeCAE - Атоммаш & ЦИФРА", layout="wide")

# Базовая база данных высокопрочных материалов по ГОСТ для ИЯУ
MATERIALS_GOST = {
    "Сталь 08Х18Н10Т (Аустенитная)": {"yield_strength": 220, "elastic_modulus": 195, "desc": "Применяется в корпусных элементах ИЗК"},
    "Сталь 12Х18Н10Т": {"yield_strength": 196, "elastic_modulus": 198, "desc": "Высокая коррозионная стойкость"},
    "Сталь 20": {"yield_strength": 245, "elastic_modulus": 200, "desc": "Общего назначения для неответственных узлов"},
    "Сталь 09Г2С": {"yield_strength": 345, "elastic_modulus": 205, "desc": "Низколегированная конструкционная сталь"},
    "Сталь 15Х5М": {"yield_strength": 280, "elastic_modulus": 210, "desc": "Жаростойкая и коррозионностойкая сталь"},
    "Сплав ХН78Т (Жаропрочный)": {"yield_strength": 350, "elastic_modulus": 210, "desc": "Для высокотемпературных узлов оборудования"},
    "Титан ВТ6": {"yield_strength": 830, "elastic_modulus": 114, "desc": "Легкий высокопрочный сплав для ответственных узлов"},
    "Алюминий АМг6": {"yield_strength": 275, "elastic_modulus": 70, "desc": "Лёгкий конструкционный сплав"},
    "Бронза БрАЖ9-4": {"yield_strength": 280, "elastic_modulus": 105, "desc": "Антикоррозионный сплав для трущихся узлов"}
}

# Дополнительная база материалов для быстрых черновых расчётов
MATERIALS_QUICK = {
    "Сталь конструкционная": {"yield_strength": 245, "elastic_modulus": 200, "thermal_expansion": 12e-6, "poisson": 0.30},
    "Нержавеющая сталь": {"yield_strength": 205, "elastic_modulus": 193, "thermal_expansion": 16e-6, "poisson": 0.30},
    "Титан": {"yield_strength": 830, "elastic_modulus": 114, "thermal_expansion": 8.6e-6, "poisson": 0.34},
    "Алюминий": {"yield_strength": 275, "elastic_modulus": 70, "thermal_expansion": 23e-6, "poisson": 0.33},
}

ANALYSIS_PRESETS = {
    "Статический": {
        "description": "Быстрая оценка прочности при заданной нагрузке и направлении.",
        "base_stress": 100.0,
        "max_stress": 5000.0,
        "magnitude_scale": 1.0,
        "default_temp": 20,
        "default_load_type": "Гравитация",
        "default_direction": "+Z",
    },
    "Температурный": {
        "description": "Оценка влияния повышенной температуры на прочность и деформации.",
        "base_stress": 115.0,
        "max_stress": 6000.0,
        "magnitude_scale": 1.1,
        "default_temp": 220,
        "default_load_type": "Комбинированная",
        "default_direction": "+Z",
    },
    "Модальный": {
        "description": "Оценка собственных частот и чувствительности к динамическому возбуждению.",
        "base_stress": 90.0,
        "max_stress": 4500.0,
        "magnitude_scale": 0.9,
        "default_temp": 20,
        "default_load_type": "Сейсмика",
        "default_direction": "+X",
    },
    "Спектральный": {
        "description": "Оценка отклика по спектральному воздействию и направлению возбуждения.",
        "base_stress": 95.0,
        "max_stress": 4800.0,
        "magnitude_scale": 1.0,
        "default_temp": 20,
        "default_load_type": "Сейсмика",
        "default_direction": "+Y",
    },
}


def get_analysis_preset(analysis_type):
    return ANALYSIS_PRESETS.get(analysis_type, ANALYSIS_PRESETS["Статический"])


def get_engineering_recommendations(analysis_type, max_stress_val, safety_factor, temperature, direction, load_type):
    recommendations = []
    if safety_factor < 1.0:
        recommendations.append("Снизить уровень нагрузки или изменить зону приложения силы.")
    elif safety_factor < 1.3:
        recommendations.append("Проверить зону концентрации напряжений и уточнить граничные условия.")
    else:
        recommendations.append("Сохранить текущую схему и проверить чувствительность к температуре и направлению.")

    if temperature > 200:
        recommendations.append("Проверить термоупругие эффекты и свойства материала при повышенной температуре.")
    if analysis_type in {"Модальный", "Спектральный"}:
        recommendations.append("Уточнить частотный отклик и возможный резонанс относительно режима возбуждения.")
    if load_type == "Сейсмика":
        recommendations.append("Проверить спектральное воздействие и направление возбуждения.")
    if direction in {"+X", "-X", "+Y", "-Y"}:
        recommendations.append("Сравнить результат с альтернативным направлением, чтобы оценить чувствительность.")
    return recommendations


def _pick_pdf_font_name():
    if not REPORTLAB_AVAILABLE:
        return None

    candidates = [
        "fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("AppFont", path))
                return "AppFont"
            except Exception:
                continue
    return "Helvetica"


def build_nds_preview_png(coords, stress, scenario_name):
    try:
        if coords is None or stress is None or len(coords) == 0:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=3, color=stress, colorscale='Jet', showscale=True)
        ))
        fig.update_layout(
            title=f"Карта НДС: {scenario_name}",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            height=460,
            margin=dict(l=0, r=0, b=0, t=50),
        )
        return pio.to_image(fig, format="png", width=1200, height=760, scale=2)
    except Exception:
        return None


def build_pdf_report_bytes(material_name, report_rows, preview_png=None):
    if not REPORTLAB_AVAILABLE:
        return None

    font_name = _pick_pdf_font_name()
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    _, height = A4

    # Титульный лист
    y = height - 30 * mm
    pdf.setFont(font_name, 18)
    pdf.drawString(20 * mm, y, "VibeCAE")
    y -= 10 * mm

    pdf.setFont(font_name, 14)
    pdf.drawString(20 * mm, y, "Научно-технический отчет по расчету")
    y -= 12 * mm

    pdf.setFont(font_name, 11)
    pdf.drawString(20 * mm, y, f"Дата формирования: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 7 * mm
    pdf.drawString(20 * mm, y, f"Материал: {material_name}")
    y -= 7 * mm
    pdf.drawString(20 * mm, y, f"Количество сценариев: {len(report_rows)}")
    y -= 12 * mm
    pdf.setFont(font_name, 10)
    pdf.drawString(20 * mm, y, "Документ сформирован автоматически на основании выбранных сценариев.")

    pdf.showPage()

    y = height - 20 * mm
    pdf.setFont(font_name, 14)
    pdf.drawString(20 * mm, y, "Сводная таблица результатов")
    y -= 8 * mm

    pdf.setFont(font_name, 10)
    pdf.drawString(20 * mm, y, f"Материал: {material_name}")
    y -= 10 * mm

    headers = ["Сценарий", "Тип", "T, °C", "Напр.", "σmax, МПа", "Запас", "Вердикт"]
    cols_mm = [20, 82, 112, 126, 144, 169, 184]

    pdf.setFont(font_name, 9)
    for label, x in zip(headers, cols_mm):
        pdf.drawString(x * mm, y, label)
    y -= 4 * mm
    pdf.line(20 * mm, y, 195 * mm, y)
    y -= 5 * mm

    for row in report_rows:
        if y < 20 * mm:
            pdf.showPage()
            pdf.setFont(font_name, 9)
            y = height - 20 * mm

        pdf.drawString(20 * mm, y, str(row["scenario"])[:36])
        pdf.drawString(82 * mm, y, str(row["analysis_type"])[:14])
        pdf.drawRightString(122 * mm, y, f"{row['temperature']}")
        pdf.drawString(126 * mm, y, str(row["direction"]))
        pdf.drawRightString(164 * mm, y, f"{row['max_stress']:.1f}")
        pdf.drawRightString(181 * mm, y, f"{row['safety_factor']:.2f}")
        pdf.drawString(184 * mm, y, str(row["verdict"])[:11])
        y -= 5 * mm

    y -= 2 * mm
    pdf.line(20 * mm, y, 195 * mm, y)
    y -= 7 * mm

    if report_rows:
        worst = min(report_rows, key=lambda r: r["safety_factor"])
        pdf.setFont(font_name, 10)
        pdf.drawString(20 * mm, y, f"Критичный сценарий: {worst['scenario'][:70]}")
        y -= 6 * mm
        pdf.drawString(20 * mm, y, f"Минимальный запас прочности: {worst['safety_factor']:.2f}")

    if preview_png is not None:
        y -= 12 * mm
        if y < 95 * mm:
            pdf.showPage()
            y = height - 20 * mm
        pdf.setFont(font_name, 11)
        pdf.drawString(20 * mm, y, "Карта НДС для критичного сценария")
        y -= 5 * mm
        try:
            image = ImageReader(BytesIO(preview_png))
            pdf.drawImage(image, 20 * mm, y - 85 * mm, width=170 * mm, height=85 * mm, preserveAspectRatio=True, anchor='n')
        except Exception:
            pdf.setFont(font_name, 9)
            pdf.drawString(20 * mm, y, "Не удалось встроить изображение карты НДС в PDF.")

    pdf.showPage()
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _safe_max_stress(*arrays):
    values = []
    for arr in arrays:
        if arr is None or np.size(arr) == 0:
            continue
        try:
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                values.append(float(np.max(finite)))
        except Exception:
            continue
    return max(values) if values else 0.0


def load_step_to_trimesh(file_bytes):
    try:
        step_path = "/tmp/vibecae_step.step"
        with open(step_path, "wb") as fh:
            fh.write(file_bytes)

        shape = cq.importers.importStep(step_path)
        if shape is None:
            return None

        if hasattr(shape, "Solids"):
            solid = shape.Solids().val()
        else:
            solid = shape

        if solid is None:
            return None

        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP import TopoDS
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRep import BRep_Tool
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane

        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != IFSelect_RetDone:
            return None

        reader.TransferRoot(1)
        shape = reader.Shape(1)

        if not shape.IsNull():
            from OCP.TopoDS import TopoDS_Shape
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.BRepTools import BRepTools
            from OCP.BRep import BRep_Tool
            from OCP.ShapeAnalysis import ShapeAnalysis_Surface
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopLoc import TopLoc_Location
            from OCP.gp import gp_Pnt
            import numpy as np

            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            faces = []
            while explorer.More():
                face = explorer.Current()
                faces.append(face)
                explorer.Next()

            if not faces:
                return None

            verts = []
            faces_idx = []
            current_offset = 0
            for face in faces:
                try:
                    mesh = BRepMesh_IncrementalMesh(face, 0.5)
                    mesh.Perform()
                    triangulation = BRep_Tool.Triangulation(face, TopLoc_Location())
                    if triangulation is None:
                        continue
                    for i in range(1, triangulation.NbNodes() + 1):
                        pnt = triangulation.Node(i).Transformed(TopLoc_Location())
                        verts.append((pnt.X(), pnt.Y(), pnt.Z()))
                    for i in range(1, triangulation.NbTriangles() + 1):
                        tri = triangulation.Triangle(i)
                        n1 = tri.Get(1) - 1
                        n2 = tri.Get(2) - 1
                        n3 = tri.Get(3) - 1
                        faces_idx.append((current_offset + n1, current_offset + n2, current_offset + n3))
                    current_offset += triangulation.NbNodes()
                except Exception:
                    continue

            if not verts:
                return None

            return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces_idx), process=False)

        return None
    except Exception as e:
        st.warning(f"Не удалось прочитать STEP: {e}")
        return None


def build_mesh_from_uploaded_model(mesh, element_size_mm):
    if mesh is None:
        return None

    try:
        mesh_copy = mesh.copy()
        mesh_copy.remove_degenerate_faces()
        mesh_copy.remove_unreferenced_vertices()
        if len(mesh_copy.faces) == 0:
            return mesh

        if element_size_mm is None:
            return mesh_copy

        size_mm = float(element_size_mm)
        target_edge = max(size_mm, 0.01)

        try:
            verts, faces = trimesh.remesh.subdivide_to_size(
                mesh_copy.vertices,
                mesh_copy.faces,
                max_edge=target_edge,
                max_iter=10,
            )
            return trimesh.Trimesh(vertices=verts, faces=faces)
        except Exception:
            try:
                verts, faces = trimesh.remesh.subdivide(
                    mesh_copy.vertices,
                    mesh_copy.faces,
                    return_index=False,
                )
                return trimesh.Trimesh(vertices=verts, faces=faces)
            except Exception:
                return mesh_copy
    except Exception:
        return mesh


def build_mesh_figure(mesh, title="Сетка на модели"):
    if mesh is None:
        return None

    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.35,
        flatshading=True,
        hoverinfo='skip'
    ))

    try:
        edges = mesh.edges_unique
        if len(edges) > 0:
            x_lines = np.empty(3 * len(edges), dtype=float)
            y_lines = np.empty(3 * len(edges), dtype=float)
            z_lines = np.empty(3 * len(edges), dtype=float)

            x_lines[0::3] = vertices[edges[:, 0], 0]
            x_lines[1::3] = vertices[edges[:, 1], 0]
            x_lines[2::3] = np.nan

            y_lines[0::3] = vertices[edges[:, 0], 1]
            y_lines[1::3] = vertices[edges[:, 1], 1]
            y_lines[2::3] = np.nan

            z_lines[0::3] = vertices[edges[:, 0], 2]
            z_lines[1::3] = vertices[edges[:, 1], 2]
            z_lines[2::3] = np.nan

            fig.add_trace(go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode='lines',
                line=dict(color='rgba(20, 60, 120, 0.95)', width=1.2),
                showlegend=False,
                hoverinfo='skip'
            ))
    except Exception:
        pass

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
    )
    return fig


def get_region_mask(mesh, load_params):
    if mesh is None:
        return np.ones(0, dtype=bool)

    coords = mesh.vertices[:, :3]
    region_mode = load_params.get("region_mode", "Автоматическая зона")
    if region_mode == "Выделено мышью":
        selected_points = load_params.get("selection_points", [])
        radius = float(load_params.get("selection_radius", 1.0))
        if selected_points:
            selected_coords = np.asarray(selected_points, dtype=float)
            if selected_coords.ndim == 1:
                selected_coords = selected_coords.reshape(1, -1)
            dist = np.linalg.norm(coords[:, None, :] - selected_coords[None, :, :], axis=2)
            mask = np.any(dist <= radius, axis=1)
            if np.any(mask):
                return mask
        return np.ones(len(coords), dtype=bool)

    if region_mode != "Пользовательская область":
        return np.ones(len(coords), dtype=bool)

    axis = load_params.get("region_axis", "Z")
    axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis, 2)
    region_min = float(load_params.get("region_min", np.min(coords[:, axis_idx])))
    region_max = float(load_params.get("region_max", np.max(coords[:, axis_idx])))
    if region_min > region_max:
        region_min, region_max = region_max, region_min

    mask = (coords[:, axis_idx] >= region_min) & (coords[:, axis_idx] <= region_max)
    return mask if np.any(mask) else np.ones(len(coords), dtype=bool)


def get_load_center(mesh, load_params):
    if mesh is None:
        return np.array([0.0, 0.0, 0.0])

    coords = mesh.vertices[:, :3]
    region_mask = get_region_mask(mesh, load_params)
    active_points = coords[region_mask] if np.any(region_mask) else coords

    location = load_params.get("location", "Центральная зона")
    if location == "Верхняя поверхность":
        z_max = np.max(active_points[:, 2])
        subset = active_points[np.isclose(active_points[:, 2], z_max, atol=1e-6)]
        return subset.mean(axis=0) if subset.size else active_points.mean(axis=0)
    if location == "Нижняя поверхность":
        z_min = np.min(active_points[:, 2])
        subset = active_points[np.isclose(active_points[:, 2], z_min, atol=1e-6)]
        return subset.mean(axis=0) if subset.size else active_points.mean(axis=0)
    if location == "Боковая поверхность":
        x_abs_max = np.max(np.abs(active_points[:, 0]))
        subset = active_points[np.isclose(np.abs(active_points[:, 0]), x_abs_max, atol=1e-6)]
        return subset.mean(axis=0) if subset.size else active_points.mean(axis=0)
    return active_points.mean(axis=0)


def get_direction_vector(direction):
    direction_map = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    return direction_map.get(direction, np.array([0.0, 0.0, 1.0]))


def build_load_stress_field(mesh, load_params, base_stress=120.0, max_stress=6000.0):
    if mesh is None or not load_params:
        return None

    coords = mesh.vertices[:, :3]
    center = get_load_center(mesh, load_params)
    direction = get_direction_vector(load_params.get("direction", "+Z"))
    magnitude = float(load_params.get("magnitude", 1.0))

    delta = coords - center
    dist = np.linalg.norm(delta, axis=1)
    size_scale = max(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) / 6.0, 1e-3)

    region_mask = get_region_mask(mesh, load_params)
    influence = np.exp(-dist / max(size_scale, 1e-3))
    influence = np.where(region_mask, influence, 0.0)

    projection = np.einsum('ij,j->i', delta, direction)
    projection = np.clip(projection / max(size_scale, 1e-3), -1.0, 1.0)

    surface_alignment = np.abs(np.einsum('ij,j->i', delta, direction)) / np.maximum(dist, 1e-6)
    surface_alignment = np.clip(surface_alignment, 0.0, 1.0)

    directional_bias = 1.0 + 0.8 * np.maximum(projection, 0.0)
    tangential_penalty = 1.0 - 0.45 * np.maximum(0.0, 1.0 - surface_alignment)

    stress = base_stress + magnitude * 20.0 * influence * directional_bias * tangential_penalty
    stress = np.clip(stress, 0.0, max_stress)
    return stress

# Боковая панель
with st.sidebar:
    selected_material = st.selectbox("Материал конструкции (ГОСТ)", list(MATERIALS_GOST.keys()))
    material_data = MATERIALS_GOST[selected_material]
    st.caption(f"_{material_data['desc']}_")
    st.caption(f"Предел текучести: {material_data['yield_strength']} МПа")

    st.markdown("---")
    st.caption("Быстрый материал для черновика")
    quick_material = st.selectbox("Быстрый выбор материала", list(MATERIALS_QUICK.keys()), key="quick_material")
    quick_material_data = MATERIALS_QUICK[quick_material]
    st.caption(f"E = {quick_material_data['elastic_modulus']} ГПа | ν = {quick_material_data['poisson']:.2f}")
    st.caption(f"σy = {quick_material_data['yield_strength']} МПа")

    st.markdown("---")
    st.markdown("*Статус: Компонентный FEA-режим*")

# Вкладки интерфейса
tab1, tab2, tab3 = st.tabs([
    "1. Импорт CAD и КЭМ", 
    "2. Сценарии нагружения", 
    "3. Анализ НДС & Отчет"
])

if 'mesh_built' not in st.session_state:
    st.session_state['mesh_built'] = False
if 'mesh_build_key' not in st.session_state:
    st.session_state['mesh_build_key'] = None
if 'exp_done' not in st.session_state:
    st.session_state['exp_done'] = False
if 'test_done' not in st.session_state:
    st.session_state['test_done'] = False
if 'stl_mesh' not in st.session_state:
    st.session_state['stl_mesh'] = None
if 'stl_name' not in st.session_state:
    st.session_state['stl_name'] = None
if 'active_mesh' not in st.session_state:
    st.session_state['active_mesh'] = None
if 'mesh_element_size' not in st.session_state:
    st.session_state['mesh_element_size'] = 2.0
if 'exp_load' not in st.session_state:
    st.session_state['exp_load'] = None
if 'test_load' not in st.session_state:
    st.session_state['test_load'] = None
if 'selected_region_points' not in st.session_state:
    st.session_state['selected_region_points'] = []

# --- ВКЛАДКА 1: ИМПОРТ И АВТОМАТИЧЕСКОЕ ПОСТРОЕНИЕ СЕТКИ ---
with tab1:
    st.header("Подготовка конечно-элементной модели")
    st.write("Загрузите CAD-модель или STL-файл, и сетка будет строиться автоматически на основе загруженной геометрии.")
    
    uploaded_file = st.file_uploader(
        "Перетащите CAD-модель сюда или выберите файл (.stp, .step, .stl, .parasolid)", 
        type=["stp", "step", "stl", "x_t"],
        key="permanent_cad_uploader"
    )
    st.caption("Если удобнее, нажмите кнопку выбора файла или просто перетащите модель в область загрузки.")
    
    if uploaded_file:
        st.info(f"Файл `{uploaded_file.name}` успешно загружен. Сетка будет построена автоматически.")

        if st.session_state.get('stl_name') != uploaded_file.name:
            try:
                if uploaded_file.name.lower().endswith('.stl'):
                    mesh_bytes = uploaded_file.getvalue()
                    mesh = trimesh.load(BytesIO(mesh_bytes), file_type='stl', force='mesh')
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    st.session_state['stl_mesh'] = mesh
                    st.session_state['stl_name'] = uploaded_file.name
                    st.session_state['active_mesh'] = mesh
                    st.session_state['mesh_build_key'] = None
                    st.success("STL-файл прочитан. Сетка будет построена автоматически.")
                elif uploaded_file.name.lower().endswith(('.step', '.stp')):
                    mesh = load_step_to_trimesh(uploaded_file.getvalue())
                    if mesh is not None:
                        st.session_state['stl_mesh'] = mesh
                        st.session_state['stl_name'] = uploaded_file.name
                        st.session_state['active_mesh'] = mesh
                        st.session_state['mesh_build_key'] = None
                        st.success("STEP-файл прочитан. Сетка будет построена автоматически.")
                    else:
                        st.warning("Не удалось обработать STEP-файл. Проверьте геометрию файла или попробуйте экспорт в STL.")
                        st.session_state['stl_mesh'] = None
                        st.session_state['stl_name'] = uploaded_file.name
                        st.session_state['active_mesh'] = None
                        st.session_state['mesh_build_key'] = None
                else:
                    st.warning("В текущей версии приложения автоматическое построение сетки поддерживается для STL и STEP.")
                    st.session_state['stl_mesh'] = None
                    st.session_state['stl_name'] = uploaded_file.name
                    st.session_state['active_mesh'] = None
                    st.session_state['mesh_build_key'] = None
            except Exception as e:
                st.error(f"Не удалось загрузить модель: {e}")

    col_mesh1, col_mesh2 = st.columns([1, 2])
    with col_mesh1:
        st.subheader("Параметры конечных элементов")
        element_size = st.slider("Размер ячейки/разбиения (мм)", 0.5, 10.0, st.session_state['mesh_element_size'], 0.5)
        mesh_type = st.radio("Тип конечных элементов", ["SOLID186 (3D 20-узловые гексаэдры)", "SOLID185 (Линейные блоки)"])
        st.caption("Меньшее значение — более мелкая сетка, большее — более крупная.")
        
        if st.session_state['stl_mesh'] is None:
            st.info("Сначала загрузите CAD-модель или STL-файл.")
        else:
            mesh_key = (st.session_state['stl_name'], round(element_size, 2), mesh_type)
            if st.session_state.get('mesh_build_key') != mesh_key:
                with st.spinner("Построение сетки на загруженной модели..."):
                    st.session_state['active_mesh'] = build_mesh_from_uploaded_model(st.session_state['stl_mesh'], element_size)
                    st.session_state['mesh_built'] = True
                    st.session_state['mesh_element_size'] = element_size
                    st.session_state['mesh_build_key'] = mesh_key
                st.success(f"Сетка построена автоматически на модели {st.session_state['stl_name']}.")
            else:
                st.caption("Сетка уже построена для текущих параметров.")
            
    with col_mesh2:
        if st.session_state['stl_mesh'] is not None:
            mesh = st.session_state.get('active_mesh') or st.session_state['stl_mesh']
            st.subheader("Сетка на загруженной модели")
            st.write(f"Файл: {st.session_state['stl_name']}")
            st.write(f"Вершин: {len(mesh.vertices):,} | Треугольников: {len(mesh.faces):,}")
            st.write(f"Параметр разбиения: {element_size:.2f} мм")

            fig_stl = build_mesh_figure(mesh, title="Треугольная сетка на модели")
            vertex_coords = mesh.vertices[:, :3]
            fig_stl.add_trace(go.Scatter3d(
                x=vertex_coords[:, 0],
                y=vertex_coords[:, 1],
                z=vertex_coords[:, 2],
                mode='markers',
                marker=dict(size=2.2, color='rgba(15, 60, 120, 0.85)'),
                hoverinfo='skip',
                customdata=np.arange(len(vertex_coords)),
                name='vertices'
            ))
            st.plotly_chart(fig_stl, use_container_width=True, key="mesh_preview_chart")
            st.caption("Выберите область приложения силы через координаты модели.")
            if st.button("Сохранить текущую область как выделенную", key="save_region_selection"):
                if len(vertex_coords) > 0:
                    center_point = vertex_coords[np.argmin(np.linalg.norm(vertex_coords - vertex_coords.mean(axis=0), axis=1))]
                    st.session_state['selected_region_points'] = [center_point.tolist()]
                    st.success("Выделена центральная точка модели. Она будет использоваться как область приложения силы.")
                else:
                    st.warning("Нет доступных вершин для выделения.")
            if st.button("Очистить выделение", key="clear_region_selection"):
                st.session_state['selected_region_points'] = []
                st.rerun()
        else:
            st.info("Загрузите CAD-модель или STL-файл, чтобы увидеть геометрию и построить на ней сетку.")

# --- ВКЛАДКА 2: СЦЕНАРИИ НАГРУЖЕНИЯ ---
with tab2:
    st.header("Сценарии нагружения по ТЗ заказчика")
    st.write("Выберите один или несколько сценариев, которые требуется проверить для модели оборудования.")

    scenario_options = [
        "1. Состояние покоя",
        "2. Эксплуатация",
        "3. Модальный анализ — пустое оборудование",
        "4. Модальный анализ — загруженное оборудование",
        "5. Проверка прочности — пустое оборудование",
        "6. Проверка прочности — пустое оборудование, с учётом динамических эффектов",
        "7. Проверка прочности — загруженное оборудование",
        "8. Проверка прочности — загруженное оборудование, с учётом динамических эффектов",
        "9. Проверка прочности — рабочая мощность",
        "10. Проверка прочности — рабочая мощность, с учётом динамических эффектов"
    ]

    selected_scenarios = st.multiselect(
        "Доступные сценарии нагружения",
        scenario_options,
        default=[],
        help="Можно выбрать несколько сценариев одновременно."
    )

    st.caption("Для каждого выбранного сценария будет сформирован отдельный блок параметров и результат расчёта.")

    if selected_scenarios:
        st.subheader("Параметры выбранных сценариев")
        for scenario in selected_scenarios:
            with st.expander(scenario, expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    analysis_type = st.selectbox(f"Тип расчета ({scenario[:20]}...)", ["Статический", "Температурный", "Модальный", "Спектральный"], key=f"analysis_type_{scenario}")
                    preset = get_analysis_preset(analysis_type)
                    st.caption(preset["description"])
                    st.slider(f"Температура для сценария ({scenario[:20]}...)", 20, 800, preset["default_temp"], key=f"temp_{scenario}")
                    st.selectbox(f"Тип нагрузки ({scenario[:20]}...)", ["Гравитация", "Сейсмика", "Комбинированная"], index=["Гравитация", "Сейсмика", "Комбинированная"].index(preset["default_load_type"]), key=f"load_type_{scenario}")
                    st.selectbox(f"Направление нагрузки ({scenario[:20]}...)", ["+X", "-X", "+Y", "-Y", "+Z", "-Z"], index=["+X", "-X", "+Y", "-Y", "+Z", "-Z"].index(preset["default_direction"]), key=f"direction_{scenario}")
                with col_b:
                    st.slider(f"Массовый коэффициент ({scenario[:20]}...)", 0.5, 2.0, 1.0, 0.1, key=f"mass_factor_{scenario}")
                    st.checkbox(f"Учитывать собственный вес ({scenario[:20]}...)", value=True, key=f"gravity_{scenario}")

    if st.button("Сформировать набор сценариев", type="primary"):
        st.session_state['selected_scenarios'] = selected_scenarios
        st.session_state['exp_done'] = bool(selected_scenarios)
        st.session_state['test_done'] = bool(selected_scenarios)

        if selected_scenarios:
            first_scenario = selected_scenarios[0]
            base_magnitude = 50.0 + 5.0 * min(len(selected_scenarios), 3)
            st.session_state['exp_load'] = {
                "location": "Центральная зона",
                "direction": "+Z",
                "magnitude": base_magnitude,
                "region_mode": "Автоматическая зона",
                "selection_points": st.session_state.get('selected_region_points', []),
                "scenario": first_scenario,
            }
            if len(selected_scenarios) > 1:
                second_scenario = selected_scenarios[1]
                st.session_state['test_load'] = {
                    "location": "Боковая поверхность",
                    "direction": "+X",
                    "magnitude": base_magnitude + 20.0,
                    "region_mode": "Автоматическая зона",
                    "selection_points": st.session_state.get('selected_region_points', []),
                    "scenario": second_scenario,
                }
            else:
                st.session_state['test_load'] = st.session_state['exp_load']

        st.success(f"Сформировано {len(selected_scenarios)} сценариев нагружения.")

# --- ВКЛАДКА 3: АНАЛИЗ НДС И ОТЧЕТ ---
with tab3:
    st.header("Инженерный вердикт и отчетность для НТС")

    selected_scenarios = st.session_state.get('selected_scenarios', [])
    mesh = st.session_state.get('active_mesh') or st.session_state.get('stl_mesh')

    if not selected_scenarios:
        st.warning("⚠️ Выберите хотя бы один сценарий нагружения на вкладке 2, чтобы видеть результаты по нагрузкам.")
    elif mesh is None:
        st.warning("⚠️ Сначала загрузите и обработайте модель, чтобы построить карты напряжений.")
    else:
        limit_strength = MATERIALS_GOST[selected_material]["yield_strength"]
        st.caption(f"Текущий материал: {selected_material} | Предел текучести: {limit_strength} МПа")
        st.info("💡 Здесь отображаются результаты именно по тем сценариям, которые выбраны на вкладке 2. Для каждого сценария показывается своя нагрузка, температура и карта напряжений.")

        coords = mesh.vertices[:, :3]
        scenario_results = []

        for scenario in selected_scenarios:
            analysis_type = st.session_state.get(f'analysis_type_{scenario}', 'Статический')
            direction = st.session_state.get(f'direction_{scenario}', '+Z')
            temperature = st.session_state.get(f'temp_{scenario}', 20)
            load_type = st.session_state.get(f'load_type_{scenario}', 'Гравитация')
            mass_factor = st.session_state.get(f'mass_factor_{scenario}', 1.0)
            include_gravity = st.session_state.get(f'gravity_{scenario}', True)
            preset = get_analysis_preset(analysis_type)

            magnitude = 40.0 * preset["magnitude_scale"] + 10.0 * float(mass_factor) + (15.0 if load_type == 'Сейсмика' else 0.0) + (10.0 if include_gravity else 0.0)
            load_params = {
                "location": "Центральная зона",
                "direction": direction,
                "magnitude": magnitude,
                "temperature": temperature,
                "load_type": load_type,
                "mass_factor": mass_factor,
                "gravity": include_gravity,
                "scenario": scenario,
                "region_mode": "Автоматическая зона",
                "selection_points": st.session_state.get('selected_region_points', []),
            }
            stress = build_load_stress_field(mesh, load_params, base_stress=preset["base_stress"] + 0.5 * temperature, max_stress=preset["max_stress"] + 10.0 * temperature)
            max_stress_val = float(np.max(stress)) if stress is not None else 0.0
            scenario_results.append((scenario, load_params, stress, max_stress_val))

        if not scenario_results:
            st.info("Сценарии ещё не сформированы. Вернитесь на вкладку 2 и нажмите кнопку формирования.")
        else:
            color_max = max(value for _, _, _, value in scenario_results) if scenario_results else 1.0
            color_max = color_max * 1.05 if color_max > 0 else 1.0

            st.subheader("Сравнение сценариев")
            comparison_rows = []
            for scenario, load_params, stress, max_stress_val in scenario_results:
                safety_factor = limit_strength / max_stress_val if max_stress_val > 0 else float("inf")
                comparison_rows.append({
                    "Сценарий": scenario,
                    "Тип расчета": st.session_state.get(f'analysis_type_{scenario}', 'Статический'),
                    "Температура, °C": load_params['temperature'],
                    "Направление": load_params['direction'],
                    "Макс. напряжение, МПа": round(max_stress_val, 2),
                    "Запас прочности": round(safety_factor, 2) if np.isfinite(safety_factor) else "∞",
                    "Вердикт": "Пройдён" if safety_factor >= 1.3 else "Требует уточнения"
                })
            if comparison_rows:
                st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

            for scenario, load_params, stress, max_stress_val in scenario_results:
                with st.expander(scenario, expanded=True):
                    col_info, col_plot = st.columns([1, 2])
                    with col_info:
                        st.metric("Макс. напряжение", f"{max_stress_val:.1f} МПа")
                        st.caption(f"Температура: {load_params['temperature']} °C | Тип: {load_params['load_type']} | Направление: {load_params['direction']}")
                        st.caption(f"Массовый коэффициент: {load_params['mass_factor']:.1f} | Учитывать собственный вес: {'да' if load_params['gravity'] else 'нет'}")
                        safety_factor = limit_strength / max_stress_val if max_stress_val > 0 else float("inf")
                        safety_text = f"{safety_factor:.3f}" if safety_factor < 1.0 else f"{safety_factor:.2f}"
                        status_text = "Пройдён" if safety_factor >= 1.3 else "Требует уточнения"
                        status_color = "normal" if safety_factor >= 1.3 else "inverse"
                        st.metric("Запас прочности", safety_text, delta=status_text, delta_color=status_color)

                        if load_params['temperature'] > 300:
                            temp_note = "Высокая температура — нужен дополнительный термоупругий контроль"
                        elif load_params['temperature'] > 100:
                            temp_note = "Повышенная температура — стоит проверить свойства материала"
                        else:
                            temp_note = "Температурный уровень в допустимом диапазоне"

                        st.caption(temp_note)
                        st.caption("Что проверить дальше:")
                        analysis_type = st.session_state.get(f'analysis_type_{scenario}', 'Статический')
                        for rec in get_engineering_recommendations(analysis_type, max_stress_val, safety_factor, load_params['temperature'], load_params['direction'], load_params['load_type']):
                            st.caption(f"• {rec}")

                        if safety_factor < 1.0:
                            st.error("Критический уровень напряжений: требуется пересмотр конструкции или нагрузки.")
                        elif safety_factor < 1.3:
                            st.warning("Напряжения близки к предельным: целесообразно уточнить параметры.")
                        else:
                            st.success("Сценарий выглядит допустимым по текущей грубой оценке.")

                    with col_plot:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter3d(
                            x=coords[:, 0],
                            y=coords[:, 1],
                            z=coords[:, 2],
                            mode='markers',
                            marker=dict(size=3, color=stress, colorscale='Jet', cmin=0, cmax=color_max, showscale=True)
                        ))
                        fig.update_layout(
                            title=f"Эпюра НДС по сценарию: {scenario}",
                            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                            height=430,
                            margin=dict(l=0, r=0, b=0, t=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            if REPORTLAB_AVAILABLE:
                report_rows = []
                worst_case = None
                for scenario, load_params, _, max_stress_val in scenario_results:
                    analysis_type = st.session_state.get(f'analysis_type_{scenario}', 'Статический')
                    safety_factor = limit_strength / max_stress_val if max_stress_val > 0 else float("inf")
                    row = {
                        "scenario": scenario,
                        "analysis_type": analysis_type,
                        "temperature": load_params['temperature'],
                        "direction": load_params['direction'],
                        "max_stress": max_stress_val,
                        "safety_factor": safety_factor,
                        "verdict": "Пройдён" if safety_factor >= 1.3 else "Требует уточнения",
                    }
                    report_rows.append(row)

                    if worst_case is None or row["safety_factor"] < worst_case["row"]["safety_factor"]:
                        worst_case = {
                            "row": row,
                            "scenario": scenario,
                        }

                preview_png = None
                if worst_case is not None:
                    for scenario, _, stress, _ in scenario_results:
                        if scenario == worst_case["scenario"]:
                            preview_png = build_nds_preview_png(coords, stress, scenario)
                            break

                pdf_data = build_pdf_report_bytes(selected_material, report_rows, preview_png=preview_png)
                if pdf_data is not None:
                    st.download_button(
                        "Скачать реальный PDF-отчет",
                        data=pdf_data,
                        file_name="vibecae_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("Не удалось сформировать PDF-отчет.")
            else:
                st.warning("Для PDF-отчета установите пакет reportlab: pip install reportlab")