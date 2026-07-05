import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import trimesh
import os
import tempfile
from datetime import datetime
from io import BytesIO

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

# База данных материалов по ГОСТ для ИЯУ.
# yield_strength — предел текучести при 20 °C (МПа), elastic_modulus — модуль упругости (ГПа),
# density — плотность (кг/м³), poisson — коэффициент Пуассона, thermal_expansion — КЛТР (1/°C),
# yield_temp_curve — снижение предела текучести с температурой (°C -> МПа).
# Значения справочные: для отчётной документации уточнять по сертификату и нормам.
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

# Нормативные коэффициенты запаса по пределу текучести
SAFETY_NORMS = {
    "ПНАЭ Г-7-002-86 (оборудование ИЯУ, НУЭ)": {"n_yield": 1.5, "desc": "Запас по пределу текучести для нормальных условий эксплуатации."},
    "ГОСТ 34233.1 (сосуды и аппараты)": {"n_yield": 1.5, "desc": "Общий коэффициент запаса по пределу текучести."},
    "Общемашиностроительный": {"n_yield": 1.4, "desc": "Типовой запас для неответственных конструкций."},
    "Пользовательский": {"n_yield": None, "desc": "Коэффициент запаса задаётся вручную."},
}

GRAVITY_MS2 = 9.81
SEISMIC_BAND_HZ = (0.5, 33.0)  # типовой диапазон сейсмического возбуждения

CONSTRAINT_FACE_OPTIONS = {
    "Нижняя грань (Z min)": (2, "min"),
    "Верхняя грань (Z max)": (2, "max"),
    "Грань X min": (0, "min"),
    "Грань X max": (0, "max"),
    "Грань Y min": (1, "min"),
    "Грань Y max": (1, "max"),
}

ANALYSIS_PRESETS = {
    "Статический": {
        "description": "Статическая прочность: вес, содержимое, сила, давление. σ = мембранная + изгибная составляющие.",
        "default_temp": 20,
        "default_load_type": "Гравитация",
        "default_direction": "-Z",
    },
    "Температурный": {
        "description": "Статика + верхняя оценка температурных напряжений E·α·ΔT при стеснённом расширении; σт снижается с температурой.",
        "default_temp": 220,
        "default_load_type": "Комбинированная",
        "default_direction": "-Z",
    },
    "Модальный": {
        "description": "Оценка первой собственной частоты по балочной модели (метод Рэлея) и проверка на сейсмический диапазон 0.5–33 Гц.",
        "default_temp": 20,
        "default_load_type": "Сейсмика",
        "default_direction": "+X",
    },
    "Спектральный": {
        "description": "Линейно-спектральный метод: сейсмическое ускорение учитывается как эквивалентная статическая нагрузка.",
        "default_temp": 20,
        "default_load_type": "Сейсмика",
        "default_direction": "+X",
    },
}


def get_analysis_preset(analysis_type):
    return ANALYSIS_PRESETS.get(analysis_type, ANALYSIS_PRESETS["Статический"])


def yield_strength_at_temp(material_data, temperature):
    """Предел текучести при заданной температуре (линейная интерполяция по кривой)."""
    curve = material_data.get("yield_temp_curve")
    base = float(material_data["yield_strength"])
    if not curve:
        return base
    temps = np.array(sorted(curve.keys()), dtype=float)
    values = np.array([curve[t] for t in sorted(curve.keys())], dtype=float)
    return float(np.interp(float(temperature), temps, values))


def get_engineering_recommendations(result, params, norm_coef):
    recs = []
    if result["analysis_type"] == "Модальный":
        f1 = result["first_frequency_hz"]
        if f1 is not None and SEISMIC_BAND_HZ[0] <= f1 <= SEISMIC_BAND_HZ[1]:
            recs.append("Первая частота попадает в сейсмический диапазон 0.5–33 Гц: требуется спектральный расчёт и/или повышение жёсткости.")
        else:
            recs.append("Первая частота вне сейсмического диапазона: допустима квазистатическая оценка сейсмики.")
        recs.append("Балочная оценка частоты грубая: для ответственных узлов выполнить модальный КЭ-анализ.")
        return recs

    sigma_total = result["sigma_total"]
    safety = result["sigma_yield_t"] / sigma_total if sigma_total > 0 else float("inf")
    if safety < 1.0:
        recs.append("Напряжения превышают предел текучести: пересмотреть конструкцию, материал или схему закрепления.")
    elif safety < norm_coef:
        recs.append(f"Запас ниже нормативного n = {norm_coef:g}: уточнить расчёт по КЭ-модели или снизить нагрузку.")
    else:
        recs.append("Запас достаточен по экспресс-оценке; для НТС подтвердить поверочным КЭ-расчётом.")

    if result["sigma_bending"] > result["sigma_membrane"]:
        recs.append("Преобладает изгиб: проверить плечо от зоны закрепления до зоны нагрузки и жёсткость сечения.")
    if float(params.get("temperature", 20)) > 150:
        recs.append(f"σт снижен по температуре до {result['sigma_yield_t']:.0f} МПа: проверить свойства материала по сертификату.")
    if result["sigma_thermal"] > 0:
        recs.append("Температурная составляющая — верхняя оценка при полном стеснении расширения; при свободном расширении она ниже.")
    if params.get("load_type") in ("Сейсмика", "Комбинированная") and float(params.get("seismic_g", 0.0)) <= 0:
        recs.append("Задано сейсмическое нагружение, но ускорение 0 g — укажите ускорение по спектру площадки.")
    return recs


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

        coords, stress = sample_for_display(coords, stress)

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


def build_pdf_report_bytes(material_name, norm_name, norm_coef, report_rows, preview_png=None):
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
    pdf.drawString(20 * mm, y, "Научно-технический отчет по экспресс-оценке прочности")
    y -= 12 * mm

    pdf.setFont(font_name, 11)
    pdf.drawString(20 * mm, y, f"Дата формирования: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 7 * mm
    pdf.drawString(20 * mm, y, f"Материал: {material_name}")
    y -= 7 * mm
    pdf.drawString(20 * mm, y, f"Норма запаса: {norm_name} (n = {norm_coef:g})")
    y -= 7 * mm
    pdf.drawString(20 * mm, y, f"Количество сценариев: {len(report_rows)}")
    y -= 12 * mm
    pdf.setFont(font_name, 10)
    pdf.drawString(20 * mm, y, "Оценка выполнена аналитическими формулами (без КЭ-решателя) и носит предварительный характер.")

    pdf.showPage()

    y = height - 20 * mm
    pdf.setFont(font_name, 14)
    pdf.drawString(20 * mm, y, "Сводная таблица результатов")
    y -= 8 * mm

    pdf.setFont(font_name, 10)
    pdf.drawString(20 * mm, y, f"Материал: {material_name} | [σ] = σт(T) / {norm_coef:g}")
    y -= 10 * mm

    headers = ["Сценарий", "Тип", "T,°C", "Результат", "Допуск", "Запас", "Вердикт"]
    cols_mm = [20, 74, 100, 110, 138, 162, 174]

    pdf.setFont(font_name, 9)
    for label, x in zip(headers, cols_mm):
        pdf.drawString(x * mm, y, label)
    y -= 4 * mm
    pdf.line(20 * mm, y, 200 * mm, y)
    y -= 5 * mm

    for row in report_rows:
        if y < 20 * mm:
            pdf.showPage()
            pdf.setFont(font_name, 9)
            y = height - 20 * mm

        pdf.drawString(20 * mm, y, str(row["scenario"])[:32])
        pdf.drawString(74 * mm, y, str(row["analysis_type"])[:13])
        pdf.drawRightString(107 * mm, y, f"{row['temperature']}")
        pdf.drawString(110 * mm, y, str(row["result_text"])[:16])
        pdf.drawString(138 * mm, y, str(row["allow_text"])[:14])
        pdf.drawRightString(171 * mm, y, str(row["safety_text"]))
        pdf.drawString(174 * mm, y, str(row["verdict"])[:18])
        y -= 5 * mm

    y -= 2 * mm
    pdf.line(20 * mm, y, 200 * mm, y)
    y -= 7 * mm

    finite_rows = [r for r in report_rows if np.isfinite(r.get("safety_sort", float("inf")))]
    if finite_rows:
        worst = min(finite_rows, key=lambda r: r["safety_sort"])
        pdf.setFont(font_name, 10)
        pdf.drawString(20 * mm, y, f"Критичный сценарий: {worst['scenario'][:70]}")
        y -= 6 * mm
        pdf.drawString(20 * mm, y, f"Минимальный запас прочности: {worst['safety_text']} (норматив n = {norm_coef:g})")

    if preview_png is not None:
        y -= 12 * mm
        if y < 95 * mm:
            pdf.showPage()
            y = height - 20 * mm
        pdf.setFont(font_name, 11)
        pdf.drawString(20 * mm, y, "Оценочное распределение напряжений для критичного сценария")
        y -= 5 * mm
        try:
            image = ImageReader(BytesIO(preview_png))
            pdf.drawImage(image, 20 * mm, y - 85 * mm, width=170 * mm, height=85 * mm, preserveAspectRatio=True, anchor='n')
        except Exception:
            pdf.setFont(font_name, 9)
            pdf.drawString(20 * mm, y, "Не удалось встроить изображение карты НДС в PDF.")

    # Методика и допущения
    pdf.showPage()
    y = height - 20 * mm
    pdf.setFont(font_name, 14)
    pdf.drawString(20 * mm, y, "Методика и допущения")
    y -= 10 * mm
    pdf.setFont(font_name, 10)
    methodology_lines = [
        "1. Оценка выполнена аналитическими формулами без КЭ-решателя (экспресс-метод).",
        "2. Мембранные напряжения: σм = F / Aср, где Aср = V / L — средняя площадь сечения вдоль силы.",
        "3. Изгибные напряжения: σи = M / W по балочной модели «зона закрепления → зона нагрузки», W ≈ A·h/6.",
        "4. Для шарнирного опирания изгибающий момент принят M ≈ F·L/4.",
        "5. Температурные напряжения: верхняя оценка E·α·ΔT при полностью стеснённом расширении.",
        "6. Предел текучести σт(T) интерполирован по справочной кривой снижения с температурой.",
        "7. Первая собственная частота — балочная модель (метод Рэлея), консервативно по наименьшему габариту сечения.",
        "8. Сейсмика учтена линейно-спектральным методом как эквивалентная статическая нагрузка m·a.",
        "9. Концентрация напряжений (отверстия, галтели, сварные швы) не учитывается.",
        "10. Результаты предназначены для предварительной оценки и не заменяют поверочный расчёт по КЭ-модели.",
    ]
    for line in methodology_lines:
        if y < 20 * mm:
            pdf.showPage()
            pdf.setFont(font_name, 10)
            y = height - 20 * mm
        pdf.drawString(20 * mm, y, line)
        y -= 6 * mm

    pdf.showPage()
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


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

        # Триангуляция всей формы целиком с заданным отклонением от геометрии
        BRepMesh_IncrementalMesh(shape, max(float(linear_deflection), 0.01))

        verts = []
        faces_idx = []
        current_offset = 0

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


def build_mesh_from_uploaded_model(mesh, element_size_mm, max_faces=400_000):
    if mesh is None:
        return None

    try:
        mesh_copy = mesh.copy()
        mesh_copy.update_faces(mesh_copy.nondegenerate_faces())
        mesh_copy.remove_unreferenced_vertices()
        if len(mesh_copy.faces) == 0:
            return mesh

        if element_size_mm is None:
            return mesh_copy

        size_mm = float(element_size_mm)
        target_edge = max(size_mm, 0.01)

        # Оценка числа треугольников после измельчения: каждое деление учетверяет грань.
        # Если прогноз превышает лимит, увеличиваем целевое ребро, чтобы не подвесить интерфейс.
        triangles = mesh_copy.triangles
        longest_edge = np.linalg.norm(np.roll(triangles, -1, axis=1) - triangles, axis=2).max(axis=1)

        def estimate_faces(edge):
            splits = np.ceil(np.log2(np.maximum(longest_edge / edge, 1.0)))
            return float(np.sum(4.0 ** splits))

        for _ in range(40):
            if estimate_faces(target_edge) <= max_faces:
                break
            target_edge *= 1.15

        try:
            verts, faces = trimesh.remesh.subdivide_to_size(
                mesh_copy.vertices,
                mesh_copy.faces,
                max_edge=target_edge,
                max_iter=10,
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
        # На крупных сетках линии рёбер делают браузер неотзывчивым — пропускаем их.
        if 0 < len(edges) <= 120_000:
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


def sample_for_display(coords, values=None, max_points=60_000):
    """Равномерно прореживает точки для отрисовки, чтобы не подвешивать браузер."""
    n = len(coords)
    if n <= max_points:
        return coords, values
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return coords[idx], (values[idx] if values is not None else None)


def get_region_mask(mesh, load_params):
    if mesh is None:
        return np.ones(0, dtype=bool)

    coords = mesh.vertices[:, :3]
    region_mode = load_params.get("region_mode", "Автоматическая зона")
    if region_mode == "Точка с радиусом":
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


def get_region_area_mm2(mesh, load_params):
    """Площадь поверхности выбранной области приложения нагрузки (мм²)."""
    if mesh is None:
        return 0.0
    mask = get_region_mask(mesh, load_params)
    if not np.any(mask):
        return 0.0
    face_mask = mask[mesh.faces].all(axis=1)
    if not np.any(face_mask):
        return 0.0
    return float(mesh.area_faces[face_mask].sum())


def get_region_centroid(mesh, mask):
    """Площадно-взвешенный центроид области (не зависит от плотности разбиения)."""
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
    """Маска вершин и центроид зоны закрепления (грань габарита + глубина зоны)."""
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
    """Объём, масса и центр масс модели (координаты сетки — в мм)."""
    if mesh is None:
        return None
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
        if is_exact:
            com = np.asarray(mesh.center_mass, dtype=float)
        else:
            com = np.asarray(mesh.centroid, dtype=float)  # площадно-взвешенный — не зависит от плотности разбиения
    except Exception:
        com = coords.mean(axis=0)
    return {
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_mm3 / 1000.0,
        "mass_kg": float(density_kg_m3) * volume_mm3 * 1e-9,
        "center_of_mass": com,
        "extents_mm": extents,
        "surface_area_cm2": float(mesh.area) / 100.0,
        "is_exact": is_exact,
    }


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


def _extent_along(coords, direction):
    proj = coords @ direction
    return float(proj.max() - proj.min())


def estimate_first_frequency_hz(mesh, material_data, mass_props, total_mass_kg, constraint_centroid, constraint_type):
    """Оценка первой собственной частоты по балочной модели (метод Рэлея).

    Балка направлена от зоны закрепления к центру масс; сечение — среднее (V/L),
    момент инерции — по наименьшему поперечному габариту (консервативно).
    """
    try:
        coords = mesh.vertices[:, :3]
        com = mass_props["center_of_mass"]
        beam_vec = com - constraint_centroid
        beam_len = float(np.linalg.norm(beam_vec))
        if beam_len < 1e-6:
            extents = coords.max(axis=0) - coords.min(axis=0)
            beam_dir = np.eye(3)[int(np.argmax(extents))]
        else:
            beam_dir = beam_vec / beam_len

        length_mm = max(_extent_along(coords, beam_dir), 1e-3)
        ref = np.array([0.0, 0.0, 1.0]) if abs(beam_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(beam_dir, ref)
        t1 /= max(np.linalg.norm(t1), 1e-9)
        t2 = np.cross(beam_dir, t1)
        h_mm = max(min(_extent_along(coords, t1), _extent_along(coords, t2)), 1e-3)

        area_m2 = (mass_props["volume_mm3"] / length_mm) * 1e-6
        inertia_m4 = area_m2 * (h_mm * 1e-3) ** 2 / 12.0
        length_m = length_mm * 1e-3
        mass_per_m = max(float(total_mass_kg) / length_m, 1e-9)
        e_pa = float(material_data["elastic_modulus"]) * 1e9
        lam = 1.875 if constraint_type == "Жёсткая заделка" else np.pi  # консоль / шарнирное опирание
        freq = (lam ** 2 / (2.0 * np.pi)) * np.sqrt(e_pa * inertia_m4 / (mass_per_m * length_m ** 4))
        return float(freq)
    except Exception:
        return None


def solve_scenario(mesh, material_data, params):
    """Аналитическая экспресс-оценка (уровень 1).

    Интерфейс решателя: solve(mesh, material, params) -> dict с полем напряжений и метриками.
    При переходе на КЭ-решатель (уровень 2) заменяется только эта функция.

    Методика:
      • мембранные напряжения σм = F / Aср, Aср = V / L (среднее сечение вдоль силы);
      • изгибные σи = M / W по балочной модели «закрепление → зона нагрузки», W ≈ A·h/6;
      • температурные σт.напр = E·α·ΔT — верхняя оценка при стеснённом расширении;
      • первая частота — балочная модель (метод Рэлея).
    """
    coords = mesh.vertices[:, :3]
    density = float(material_data.get("density", 7850.0))
    mass_props = get_mass_properties(mesh, density)
    temperature = float(params.get("temperature", 20))
    sigma_yield_t = yield_strength_at_temp(material_data, temperature)

    total_mass_kg = mass_props["mass_kg"] + max(float(params.get("contents_mass_kg", 0.0)), 0.0)
    direction_label = params.get("direction", "-Z")
    d = get_direction_vector(direction_label)
    load_type = params.get("load_type", "Гравитация")

    force_terms = []
    force_vec = np.zeros(3)
    if params.get("include_gravity", True) and total_mass_kg > 0:
        f_g = total_mass_kg * GRAVITY_MS2
        force_vec += f_g * np.array([0.0, 0.0, -1.0])
        force_terms.append({"name": "Вес (модель + содержимое)", "value_n": f_g, "direction": "-Z"})
    seismic_g = float(params.get("seismic_g", 0.0))
    if load_type in ("Сейсмика", "Комбинированная") and seismic_g > 0 and total_mass_kg > 0:
        f_s = total_mass_kg * GRAVITY_MS2 * seismic_g
        force_vec += f_s * d
        force_terms.append({"name": f"Сейсмическая инерционная ({seismic_g:.2f} g)", "value_n": f_s, "direction": direction_label})
    point_force = float(params.get("point_force_n", 0.0))
    if abs(point_force) > 1e-9:
        force_vec += point_force * d
        force_terms.append({"name": "Сосредоточенная сила", "value_n": point_force, "direction": direction_label})
    pressure = float(params.get("pressure_mpa", 0.0))
    region_area_mm2 = get_region_area_mm2(mesh, params)
    if pressure > 1e-9 and region_area_mm2 > 0:
        f_p = pressure * region_area_mm2
        force_vec += f_p * d
        force_terms.append({"name": f"Давление {pressure:g} МПа × {region_area_mm2:,.0f} мм²", "value_n": f_p, "direction": direction_label})

    force_total = float(np.linalg.norm(force_vec))

    constraint_mask, constraint_centroid = get_constraint_zone(
        mesh, params.get("constraint_face"), params.get("constraint_zone_frac", 0.05)
    )
    constraint_type = params.get("constraint_type", "Жёсткая заделка")

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
        if constraint_type != "Жёсткая заделка":
            moment_nmm *= 0.25  # шарнирное опирание: M ≈ F·L/4 вместо консольного F·L
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

    analysis_type = params.get("analysis_type", "Статический")
    sigma_thermal = 0.0
    if analysis_type == "Температурный" and temperature > 20:
        alpha = float(material_data.get("thermal_expansion", 12e-6))
        e_mpa = float(material_data["elastic_modulus"]) * 1000.0
        restraint = 1.0 if constraint_type == "Жёсткая заделка" else 0.3
        sigma_thermal = e_mpa * alpha * (temperature - 20.0) * restraint

    first_frequency_hz = estimate_first_frequency_hz(
        mesh, material_data, mass_props, total_mass_kg, constraint_centroid, constraint_type
    )

    return {
        "analysis_type": analysis_type,
        "mass_props": mass_props,
        "total_mass_kg": total_mass_kg,
        "force_terms": force_terms,
        "force_total_n": force_total,
        "sigma_membrane": sigma_membrane,
        "sigma_bending": sigma_bending,
        "sigma_thermal": sigma_thermal,
        "sigma_total": sigma_membrane + sigma_bending + sigma_thermal,
        "sigma_yield_t": sigma_yield_t,
        "first_frequency_hz": first_frequency_hz,
        "load_center": load_center,
        "constraint_centroid": constraint_centroid,
        "region_area_mm2": region_area_mm2,
    }


def build_display_stress_field(mesh, result):
    """Распределение напряжений для визуализации: мембранная и температурная части
    равномерны, изгибная нарастает к зоне закрепления. Максимум поля равен σ_total."""
    coords = mesh.vertices[:, :3]
    dist_con = np.linalg.norm(coords - result["constraint_centroid"], axis=1)
    d_min = float(dist_con.min())
    span = max(float(dist_con.max()) - d_min, 1e-6)
    bend_shape = 1.0 - (dist_con - d_min) / span
    return (result["sigma_membrane"] + result["sigma_thermal"]) + result["sigma_bending"] * bend_shape


def build_mode_shape_field(mesh, result):
    """Качественная форма первого тона (относительные перемещения 0..1)."""
    coords = mesh.vertices[:, :3]
    dist_con = np.linalg.norm(coords - result["constraint_centroid"], axis=1)
    d_min = float(dist_con.min())
    span = max(float(dist_con.max()) - d_min, 1e-6)
    return ((dist_con - d_min) / span) ** 2

# Боковая панель
with st.sidebar:
    selected_material = st.selectbox("Материал конструкции (ГОСТ)", list(MATERIALS_GOST.keys()))
    material_data = MATERIALS_GOST[selected_material]
    st.caption(f"_{material_data['desc']}_")
    st.caption(f"σт(20 °C) = {material_data['yield_strength']} МПа | E = {material_data['elastic_modulus']} ГПа")
    st.caption(f"ρ = {material_data['density']} кг/м³ | ν = {material_data['poisson']:.2f} | α = {material_data['thermal_expansion'] * 1e6:.1f}·10⁻⁶ 1/°C")
    st.caption("Свойства справочные — для отчётной документации уточнять по сертификату.")

    st.markdown("---")
    norm_name = st.selectbox("Нормативный коэффициент запаса", list(SAFETY_NORMS.keys()))
    if SAFETY_NORMS[norm_name]["n_yield"] is None:
        norm_coef = st.number_input("Коэффициент запаса по σт", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    else:
        norm_coef = SAFETY_NORMS[norm_name]["n_yield"]
        st.caption(SAFETY_NORMS[norm_name]["desc"])
    st.caption(f"Допускаемое напряжение: [σ] = σт(T) / {norm_coef:g}")

    st.markdown("---")
    st.markdown("*Режим: экспресс-оценка (аналитические формулы, без КЭ-решателя)*")

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
if 'stl_mesh' not in st.session_state:
    st.session_state['stl_mesh'] = None
if 'stl_name' not in st.session_state:
    st.session_state['stl_name'] = None
if 'source_type' not in st.session_state:
    st.session_state['source_type'] = None
if 'step_bytes' not in st.session_state:
    st.session_state['step_bytes'] = None
if 'active_mesh' not in st.session_state:
    st.session_state['active_mesh'] = None
if 'mesh_element_size' not in st.session_state:
    st.session_state['mesh_element_size'] = 2.0

# --- ВКЛАДКА 1: ИМПОРТ И АВТОМАТИЧЕСКОЕ ПОСТРОЕНИЕ СЕТКИ ---
with tab1:
    st.header("Подготовка конечно-элементной модели")
    st.write("Загрузите CAD-модель или STL-файл, и сетка будет строиться автоматически на основе загруженной геометрии.")
    
    uploaded_file = st.file_uploader(
        "Перетащите CAD-модель сюда или выберите файл (.stp, .step, .stl)",
        type=["stp", "step", "stl"],
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
                        mesh = mesh.to_mesh() if hasattr(mesh, 'to_mesh') else mesh.dump(concatenate=True)
                    st.session_state['stl_mesh'] = mesh
                    st.session_state['stl_name'] = uploaded_file.name
                    st.session_state['source_type'] = 'stl'
                    st.session_state['step_bytes'] = None
                    st.session_state['active_mesh'] = mesh
                    st.session_state['mesh_build_key'] = None
                    st.success("STL-файл прочитан. Сетка будет построена автоматически.")
                elif uploaded_file.name.lower().endswith(('.step', '.stp')):
                    step_bytes = uploaded_file.getvalue()
                    deflection = max(st.session_state.get('mesh_element_size', 2.0) / 4.0, 0.05)
                    mesh = load_step_to_trimesh(step_bytes, linear_deflection=deflection)
                    if mesh is not None:
                        st.session_state['stl_mesh'] = mesh
                        st.session_state['stl_name'] = uploaded_file.name
                        st.session_state['source_type'] = 'step'
                        st.session_state['step_bytes'] = step_bytes
                        st.session_state['active_mesh'] = mesh
                        st.session_state['mesh_build_key'] = None
                        st.success("STEP-файл прочитан. Сетка будет построена автоматически.")
                    else:
                        st.warning("Не удалось обработать STEP-файл. Проверьте геометрию файла или попробуйте экспорт в STL.")
                        st.session_state['stl_mesh'] = None
                        st.session_state['stl_name'] = uploaded_file.name
                        st.session_state['source_type'] = None
                        st.session_state['step_bytes'] = None
                        st.session_state['active_mesh'] = None
                        st.session_state['mesh_build_key'] = None
            except Exception as e:
                st.error(f"Не удалось загрузить модель: {e}")

    col_mesh1, col_mesh2 = st.columns([1, 2])
    with col_mesh1:
        st.subheader("Параметры сетки")
        element_size = st.slider("Размер ячейки/разбиения (мм)", 0.5, 10.0, st.session_state['mesh_element_size'], 0.5)
        st.caption("Поверхностная треугольная сетка. Для STEP размер влияет в обе стороны (геометрия перетриангулируется), для STL возможно только измельчение исходной сетки. Число треугольников ограничено ~400 тыс. для отзывчивости интерфейса.")
        
        if st.session_state['stl_mesh'] is None:
            st.info("Сначала загрузите CAD-модель или STL-файл.")
        else:
            mesh_key = (st.session_state['stl_name'], round(element_size, 2))
            if st.session_state.get('mesh_build_key') != mesh_key:
                with st.spinner("Построение сетки на загруженной модели..."):
                    base_mesh = st.session_state['stl_mesh']
                    if st.session_state.get('source_type') == 'step' and st.session_state.get('step_bytes'):
                        retessellated = load_step_to_trimesh(
                            st.session_state['step_bytes'],
                            linear_deflection=max(element_size / 4.0, 0.05),
                        )
                        if retessellated is not None:
                            base_mesh = retessellated
                            st.session_state['stl_mesh'] = retessellated
                    st.session_state['active_mesh'] = build_mesh_from_uploaded_model(base_mesh, element_size)
                    st.session_state['mesh_built'] = True
                    st.session_state['mesh_element_size'] = element_size
                    st.session_state['mesh_build_key'] = mesh_key
                st.success(f"Сетка построена автоматически на модели {st.session_state['stl_name']}.")
            else:
                st.caption("Сетка уже построена для текущих параметров.")
            
    with col_mesh2:
        if st.session_state['stl_mesh'] is not None:
            mesh = st.session_state.get('active_mesh')
            if mesh is None:
                mesh = st.session_state['stl_mesh']
            st.subheader("Сетка на загруженной модели")
            st.write(f"Файл: {st.session_state['stl_name']}")
            st.write(f"Вершин: {len(mesh.vertices):,} | Треугольников: {len(mesh.faces):,}")
            st.write(f"Параметр разбиения: {element_size:.2f} мм")

            mass_props = get_mass_properties(mesh, material_data["density"])
            if mass_props is not None:
                ext = mass_props["extents_mm"]
                com = mass_props["center_of_mass"]
                st.write(f"Габариты: {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} мм")
                st.write(f"Объём: {mass_props['volume_cm3']:.1f} см³ | Масса ({selected_material}): {mass_props['mass_kg']:.2f} кг")
                com_note = f"Центр масс: X={com[0]:.1f}, Y={com[1]:.1f}, Z={com[2]:.1f} мм"
                if not mass_props["is_exact"]:
                    com_note += " | сетка не замкнута — объём и масса оценочные"
                st.caption(com_note)

            fig_stl = build_mesh_figure(mesh, title="Треугольная сетка на модели")
            vertex_coords = mesh.vertices[:, :3]
            display_coords, _ = sample_for_display(vertex_coords)
            fig_stl.add_trace(go.Scatter3d(
                x=display_coords[:, 0],
                y=display_coords[:, 1],
                z=display_coords[:, 2],
                mode='markers',
                marker=dict(size=2.2, color='rgba(15, 60, 120, 0.85)'),
                hoverinfo='skip',
                name='vertices'
            ))
            st.plotly_chart(fig_stl, key="mesh_preview_chart")
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
                    load_type = st.selectbox(f"Тип нагрузки ({scenario[:20]}...)", ["Гравитация", "Сейсмика", "Комбинированная"], index=["Гравитация", "Сейсмика", "Комбинированная"].index(preset["default_load_type"]), key=f"load_type_{scenario}")
                    st.selectbox(f"Направление нагрузки ({scenario[:20]}...)", ["+X", "-X", "+Y", "-Y", "+Z", "-Z"], index=["+X", "-X", "+Y", "-Y", "+Z", "-Z"].index(preset["default_direction"]), key=f"direction_{scenario}")
                with col_b:
                    st.checkbox(f"Учитывать собственный вес ({scenario[:20]}...)", value=True, key=f"gravity_{scenario}")
                    st.number_input(f"Масса содержимого, кг ({scenario[:20]}...)", min_value=0.0, value=0.0, step=10.0, key=f"contents_mass_{scenario}", help="Масса среды, оснастки или груза, добавляемая к массе модели.")
                    st.number_input(f"Сосредоточенная сила, Н ({scenario[:20]}...)", min_value=0.0, value=0.0, step=100.0, key=f"point_force_{scenario}", help="Прикладывается в центре выбранной области нагрузки по заданному направлению.")
                    st.number_input(f"Давление на область, МПа ({scenario[:20]}...)", min_value=0.0, value=0.0, step=0.05, format="%.2f", key=f"pressure_{scenario}", help="Равнодействующая = давление × площадь выбранной области.")
                    if load_type in ("Сейсмика", "Комбинированная"):
                        st.number_input(f"Сейсмическое ускорение, g ({scenario[:20]}...)", min_value=0.0, max_value=5.0, value=0.5, step=0.1, key=f"seismic_{scenario}", help="Ускорение по спектру площадки; инерционная сила = m·a.")

                st.caption("Закрепление модели для этого сценария")
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.selectbox(f"Зона закрепления ({scenario[:20]}...)", list(CONSTRAINT_FACE_OPTIONS.keys()), key=f"constraint_face_{scenario}")
                with cc2:
                    st.selectbox(f"Тип закрепления ({scenario[:20]}...)", ["Жёсткая заделка", "Шарнирное опирание"], key=f"constraint_type_{scenario}")
                with cc3:
                    st.slider(f"Глубина зоны, % габарита ({scenario[:20]}...)", 1, 30, 5, key=f"constraint_zone_{scenario}")

                st.caption("Область приложения нагрузки для этого сценария (давление и сила действуют на неё)")
                region_mode = st.selectbox(
                    f"Режим области ({scenario[:20]}...)",
                    ["Автоматическая зона", "Пользовательская область", "Точка с радиусом"],
                    key=f"region_mode_{scenario}",
                    help="Автоматическая зона — нагрузка по всей модели. Пользовательская область — диапазон по оси. Точка с радиусом — сфера вокруг заданной точки.",
                )

                if region_mode == "Пользовательская область":
                    region_axis = st.selectbox(
                        f"Ось области ({scenario[:20]}...)",
                        ["X", "Y", "Z"],
                        index=2,
                        key=f"region_axis_{scenario}",
                    )
                    mesh_for_bounds = st.session_state.get('active_mesh')
                    if mesh_for_bounds is None:
                        mesh_for_bounds = st.session_state.get('stl_mesh')
                    if mesh_for_bounds is not None and len(mesh_for_bounds.vertices) > 0:
                        bounds_min = mesh_for_bounds.vertices[:, :3].min(axis=0)
                        bounds_max = mesh_for_bounds.vertices[:, :3].max(axis=0)
                        axis_idx = {"X": 0, "Y": 1, "Z": 2}[region_axis]
                        lo = float(bounds_min[axis_idx])
                        hi = float(bounds_max[axis_idx])
                        if hi - lo < 1e-9:
                            hi = lo + 1e-3
                        st.slider(
                            f"Диапазон по оси {region_axis} ({scenario[:20]}...)",
                            lo, hi, (lo, hi),
                            key=f"region_range_{scenario}",
                        )
                    else:
                        st.info("Диапазон по оси станет доступен после загрузки модели.")

                elif region_mode == "Точка с радиусом":
                    mesh_for_point = st.session_state.get('active_mesh')
                    if mesh_for_point is None:
                        mesh_for_point = st.session_state.get('stl_mesh')
                    if mesh_for_point is not None and len(mesh_for_point.vertices) > 0:
                        coords = mesh_for_point.vertices[:, :3]
                        center_default = coords.mean(axis=0)
                        bounds_min = coords.min(axis=0)
                        bounds_max = coords.max(axis=0)
                        diag = float(np.linalg.norm(bounds_max - bounds_min))
                    else:
                        center_default = np.array([0.0, 0.0, 0.0])
                        diag = 1.0

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.number_input(f"X точки ({scenario[:20]}...)", value=float(center_default[0]), key=f"region_px_{scenario}")
                    with c2:
                        st.number_input(f"Y точки ({scenario[:20]}...)", value=float(center_default[1]), key=f"region_py_{scenario}")
                    with c3:
                        st.number_input(f"Z точки ({scenario[:20]}...)", value=float(center_default[2]), key=f"region_pz_{scenario}")
                    st.slider(
                        f"Радиус области ({scenario[:20]}...)",
                        max(diag / 200.0, 0.1), max(diag, 1.0), max(diag / 10.0, 0.5),
                        key=f"region_radius_{scenario}",
                    )

    if st.button("Сформировать набор сценариев", type="primary"):
        st.session_state['selected_scenarios'] = selected_scenarios
        st.session_state.pop('pdf_report', None)
        st.success(f"Сформировано {len(selected_scenarios)} сценариев нагружения.")

# --- ВКЛАДКА 3: АНАЛИЗ НДС И ОТЧЕТ ---
with tab3:
    st.header("Инженерный вердикт и отчетность для НТС")

    selected_scenarios = st.session_state.get('selected_scenarios', [])
    mesh = st.session_state.get('active_mesh')
    if mesh is None:
        mesh = st.session_state.get('stl_mesh')

    if not selected_scenarios:
        st.warning("⚠️ Выберите хотя бы один сценарий нагружения на вкладке 2, чтобы видеть результаты по нагрузкам.")
    elif mesh is None:
        st.warning("⚠️ Сначала загрузите и обработайте модель, чтобы построить карты напряжений.")
    else:
        sigma_yield_20 = MATERIALS_GOST[selected_material]["yield_strength"]
        st.caption(f"Материал: {selected_material} | σт(20 °C) = {sigma_yield_20} МПа | Норма: {norm_name} (n = {norm_coef:g})")
        st.info("💡 Экспресс-оценка аналитическими формулами: мембранные + изгибные + температурные напряжения, первая частота по методу Рэлея. Результаты — для предварительного сравнения сценариев, поверочный КЭ-расчёт они не заменяют.")

        coords = mesh.vertices[:, :3]
        scenario_results = []

        for scenario in selected_scenarios:
            analysis_type = st.session_state.get(f'analysis_type_{scenario}', 'Статический')
            direction = st.session_state.get(f'direction_{scenario}', '-Z')
            temperature = st.session_state.get(f'temp_{scenario}', 20)
            load_type = st.session_state.get(f'load_type_{scenario}', 'Гравитация')
            include_gravity = st.session_state.get(f'gravity_{scenario}', True)

            region_mode = st.session_state.get(f"region_mode_{scenario}", "Автоматическая зона")
            region_settings = {
                "region_mode": region_mode,
                "region_axis": st.session_state.get(f"region_axis_{scenario}", "Z"),
                "selection_points": [],
                "selection_radius": st.session_state.get(f"region_radius_{scenario}", 1.0),
            }
            if region_mode == "Пользовательская область":
                region_range = st.session_state.get(f"region_range_{scenario}")
                if region_range is not None:
                    region_settings["region_min"] = float(region_range[0])
                    region_settings["region_max"] = float(region_range[1])
            elif region_mode == "Точка с радиусом":
                region_settings["selection_points"] = [[
                    float(st.session_state.get(f"region_px_{scenario}", 0.0)),
                    float(st.session_state.get(f"region_py_{scenario}", 0.0)),
                    float(st.session_state.get(f"region_pz_{scenario}", 0.0)),
                ]]

            params = {
                "analysis_type": analysis_type,
                "direction": direction,
                "temperature": temperature,
                "load_type": load_type,
                "include_gravity": include_gravity,
                "contents_mass_kg": st.session_state.get(f'contents_mass_{scenario}', 0.0),
                "point_force_n": st.session_state.get(f'point_force_{scenario}', 0.0),
                "pressure_mpa": st.session_state.get(f'pressure_{scenario}', 0.0),
                "seismic_g": st.session_state.get(f'seismic_{scenario}', 0.5 if load_type in ("Сейсмика", "Комбинированная") else 0.0),
                "constraint_face": st.session_state.get(f'constraint_face_{scenario}', "Нижняя грань (Z min)"),
                "constraint_type": st.session_state.get(f'constraint_type_{scenario}', "Жёсткая заделка"),
                "constraint_zone_frac": float(st.session_state.get(f'constraint_zone_{scenario}', 5)) / 100.0,
                "scenario": scenario,
                **region_settings,
            }
            result = solve_scenario(mesh, material_data, params)
            if result["analysis_type"] == "Модальный":
                field = build_mode_shape_field(mesh, result)
            else:
                field = build_display_stress_field(mesh, result)
            scenario_results.append((scenario, params, result, field))

        if not scenario_results:
            st.info("Сценарии ещё не сформированы. Вернитесь на вкладку 2 и нажмите кнопку формирования.")
        else:
            strength_maxes = [r["sigma_total"] for _, _, r, _ in scenario_results if r["analysis_type"] != "Модальный"]
            color_max = max(strength_maxes) * 1.05 if strength_maxes and max(strength_maxes) > 0 else 1.0

            st.subheader("Сравнение сценариев")
            comparison_rows = []
            for scenario, params, result, field in scenario_results:
                sigma_allow = result["sigma_yield_t"] / norm_coef
                if result["analysis_type"] == "Модальный":
                    f1 = result["first_frequency_hz"]
                    in_band = f1 is not None and SEISMIC_BAND_HZ[0] <= f1 <= SEISMIC_BAND_HZ[1]
                    comparison_rows.append({
                        "Сценарий": scenario,
                        "Тип расчета": result["analysis_type"],
                        "Температура, °C": params['temperature'],
                        "σт(T), МПа": round(result["sigma_yield_t"], 0),
                        "Результат": f"f₁ ≈ {f1:.1f} Гц" if f1 is not None else "f₁: н/д",
                        "Критерий": "вне 0.5–33 Гц",
                        "Запас": "—",
                        "Вердикт": "Возможен резонанс" if in_band else "Резонанс маловероятен",
                    })
                else:
                    sigma_total = result["sigma_total"]
                    safety_factor = result["sigma_yield_t"] / sigma_total if sigma_total > 0 else float("inf")
                    comparison_rows.append({
                        "Сценарий": scenario,
                        "Тип расчета": result["analysis_type"],
                        "Температура, °C": params['temperature'],
                        "σт(T), МПа": round(result["sigma_yield_t"], 0),
                        "Результат": f"σmax ≈ {sigma_total:.1f} МПа",
                        "Критерий": f"[σ] = {sigma_allow:.1f} МПа",
                        "Запас": round(safety_factor, 2) if np.isfinite(safety_factor) else "∞",
                        "Вердикт": "Пройдён" if safety_factor >= norm_coef else "Не пройдён",
                    })
            if comparison_rows:
                st.dataframe(comparison_rows, width='stretch', hide_index=True)

            for scenario, params, result, field in scenario_results:
                with st.expander(scenario, expanded=True):
                    col_info, col_plot = st.columns([1, 2])
                    is_modal = result["analysis_type"] == "Модальный"
                    with col_info:
                        if is_modal:
                            f1 = result["first_frequency_hz"]
                            st.metric("Первая собственная частота", f"≈ {f1:.1f} Гц" if f1 is not None else "н/д")
                            st.caption(f"Балочная модель, {params['constraint_type'].lower()} | Масса: {result['total_mass_kg']:.1f} кг (модель {result['mass_props']['mass_kg']:.1f} кг)")
                            if f1 is not None and SEISMIC_BAND_HZ[0] <= f1 <= SEISMIC_BAND_HZ[1]:
                                st.error(f"Частота в сейсмическом диапазоне {SEISMIC_BAND_HZ[0]}–{SEISMIC_BAND_HZ[1]} Гц: возможен резонанс.")
                            else:
                                st.success("Частота вне типового сейсмического диапазона 0.5–33 Гц.")
                        else:
                            st.metric("Макс. напряжение (оценка)", f"{result['sigma_total']:.1f} МПа")
                            st.caption(f"σ мембранное: {result['sigma_membrane']:.1f} | σ изгибное: {result['sigma_bending']:.1f} | σ температурное: {result['sigma_thermal']:.1f} МПа")
                            st.caption(f"Суммарная нагрузка: {result['force_total_n']:,.0f} Н | Масса: {result['total_mass_kg']:.1f} кг (модель {result['mass_props']['mass_kg']:.1f} кг)")
                            if result["force_terms"]:
                                st.caption("Составляющие нагрузки:")
                                for term in result["force_terms"]:
                                    st.caption(f"• {term['name']}: {term['value_n']:,.0f} Н ({term['direction']})")
                            else:
                                st.warning("Нагрузки не заданы: укажите вес, силу, давление или ускорение на вкладке 2.")

                            sigma_allow = result["sigma_yield_t"] / norm_coef
                            st.caption(f"σт({params['temperature']} °C) = {result['sigma_yield_t']:.0f} МПа | [σ] = {sigma_allow:.0f} МПа (n = {norm_coef:g})")
                            safety_factor = result["sigma_yield_t"] / result["sigma_total"] if result["sigma_total"] > 0 else float("inf")
                            safety_text = f"{safety_factor:.2f}" if np.isfinite(safety_factor) else "∞"
                            status_text = "Пройдён" if safety_factor >= norm_coef else "Не пройдён"
                            status_color = "normal" if safety_factor >= norm_coef else "inverse"
                            st.metric("Запас прочности по σт(T)", safety_text, delta=status_text, delta_color=status_color)
                            if result["first_frequency_hz"] is not None:
                                st.caption(f"Первая частота (справочно): ≈ {result['first_frequency_hz']:.1f} Гц")

                            if safety_factor < 1.0:
                                st.error("Напряжения превышают предел текучести: требуется пересмотр конструкции или нагрузки.")
                            elif safety_factor < norm_coef:
                                st.warning(f"Запас ниже нормативного n = {norm_coef:g}: требуется уточнение.")
                            else:
                                st.success("Сценарий допустим по экспресс-оценке.")

                        st.caption("Что проверить дальше:")
                        for rec in get_engineering_recommendations(result, params, norm_coef):
                            st.caption(f"• {rec}")

                    with col_plot:
                        plot_coords, plot_field = sample_for_display(coords, field)
                        fig = go.Figure()
                        if is_modal:
                            marker = dict(size=3, color=plot_field, colorscale='Viridis', cmin=0, cmax=1, showscale=True)
                            plot_title = f"Форма 1-го тона (отн. перемещения): {scenario}"
                        else:
                            marker = dict(size=3, color=plot_field, colorscale='Jet', cmin=0, cmax=color_max, showscale=True)
                            plot_title = f"Оценочное распределение напряжений: {scenario}"
                        fig.add_trace(go.Scatter3d(
                            x=plot_coords[:, 0],
                            y=plot_coords[:, 1],
                            z=plot_coords[:, 2],
                            mode='markers',
                            marker=marker
                        ))
                        fig.update_layout(
                            title=plot_title,
                            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                            height=430,
                            margin=dict(l=0, r=0, b=0, t=40),
                        )
                        st.plotly_chart(fig)
                        if not is_modal:
                            st.caption("Распределение качественное: мембранная и температурная части равномерны, изгибная нарастает к зоне закрепления. Максимум карты равен расчётному σmax.")

            with st.expander("Методика и допущения экспресс-оценки"):
                st.markdown(
                    "- Мембранные напряжения: σм = F / Aср, где Aср = V / L — средняя площадь сечения вдоль направления равнодействующей.\n"
                    "- Изгибные напряжения: σи = M / W по балочной модели «зона закрепления → зона нагрузки», W ≈ A·h/6; для шарнирного опирания M ≈ F·L/4.\n"
                    "- Температурные напряжения: верхняя оценка E·α·ΔT при полностью стеснённом расширении (для шарнирного опирания — коэффициент 0.3).\n"
                    "- Предел текучести σт(T) интерполируется по справочной кривой снижения с температурой.\n"
                    "- Первая собственная частота — балочная модель (метод Рэлея) по наименьшему поперечному габариту (консервативно).\n"
                    "- Сейсмика учитывается линейно-спектральным методом как эквивалентная статическая нагрузка m·a.\n"
                    "- Концентрация напряжений (отверстия, галтели) не учитывается. Результаты не заменяют поверочный КЭ-расчёт."
                )

            st.markdown("---")
            if REPORTLAB_AVAILABLE:
                if st.button("Сформировать PDF-отчет", width='stretch'):
                    report_rows = []
                    worst_case = None
                    for scenario, params, result, field in scenario_results:
                        if result["analysis_type"] == "Модальный":
                            f1 = result["first_frequency_hz"]
                            in_band = f1 is not None and SEISMIC_BAND_HZ[0] <= f1 <= SEISMIC_BAND_HZ[1]
                            row = {
                                "scenario": scenario,
                                "analysis_type": result["analysis_type"],
                                "temperature": params['temperature'],
                                "result_text": f"f1 = {f1:.1f} Гц" if f1 is not None else "f1: н/д",
                                "allow_text": "вне 0.5-33 Гц",
                                "safety_text": "—",
                                "verdict": "Возможен резонанс" if in_band else "Резонанс маловероятен",
                                "safety_sort": float("inf"),
                            }
                        else:
                            sigma_total = result["sigma_total"]
                            safety_factor = result["sigma_yield_t"] / sigma_total if sigma_total > 0 else float("inf")
                            sigma_allow = result["sigma_yield_t"] / norm_coef
                            row = {
                                "scenario": scenario,
                                "analysis_type": result["analysis_type"],
                                "temperature": params['temperature'],
                                "result_text": f"{sigma_total:.1f} МПа",
                                "allow_text": f"{sigma_allow:.1f} МПа",
                                "safety_text": f"{safety_factor:.2f}" if np.isfinite(safety_factor) else "∞",
                                "verdict": "Пройдён" if safety_factor >= norm_coef else "Не пройдён",
                                "safety_sort": safety_factor,
                            }
                        report_rows.append(row)

                        if row["safety_sort"] != float("inf") and (worst_case is None or row["safety_sort"] < worst_case["safety_sort"]):
                            worst_case = {"scenario": scenario, "safety_sort": row["safety_sort"]}

                    with st.spinner("Формирование PDF-отчета..."):
                        preview_png = None
                        if worst_case is not None:
                            for scenario, _, result, field in scenario_results:
                                if scenario == worst_case["scenario"]:
                                    preview_png = build_nds_preview_png(coords, field, scenario)
                                    break

                        pdf_data = build_pdf_report_bytes(selected_material, norm_name, norm_coef, report_rows, preview_png=preview_png)
                    if pdf_data is not None:
                        st.session_state['pdf_report'] = pdf_data
                    else:
                        st.session_state.pop('pdf_report', None)
                        st.warning("Не удалось сформировать PDF-отчет.")

                if st.session_state.get('pdf_report'):
                    st.download_button(
                        "Скачать PDF-отчет",
                        data=st.session_state['pdf_report'],
                        file_name="vibecae_report.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )
            else:
                st.info("ReportLab недоступен. Установите reportlab для возможности экспорта PDF-отчётов.")
