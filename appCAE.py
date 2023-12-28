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
