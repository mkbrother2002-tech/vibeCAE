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

try:
    import fem_solver
    FEM_AVAILABLE = fem_solver.fem_available()
except Exception:
    fem_solver = None
    FEM_AVAILABLE = False

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
        "description": "Линейно-спектральный метод (МКЭ): отклик тонов по огибающей поэтажного спектра, комбинация SRSS + статика НУЭ. В экспресс-режиме — эквивалентная статическая нагрузка m·a.",
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
    is_fem = bool(result.get("fem"))
    if result["analysis_type"] == "Модальный":
        f1 = result["first_frequency_hz"]
        if f1 is not None and SEISMIC_BAND_HZ[0] <= f1 <= SEISMIC_BAND_HZ[1]:
            recs.append("Первая частота попадает в сейсмический диапазон 0.5–33 Гц: требуется спектральный расчёт и/или повышение жёсткости.")
        else:
            recs.append("Первая частота вне сейсмического диапазона: допустима квазистатическая оценка сейсмики.")
        if is_fem:
            te = result.get("total_effective_mass_kg")
            mm_kg = result["mass_props"]["mass_kg"]
            if te and mm_kg > 0 and min(te) / mm_kg < 0.9:
                recs.append("Сумма эффективных масс 10 тонов < 90% по одной из осей: для спектрального расчёта увеличить число тонов.")
        else:
            recs.append("Балочная оценка частоты грубая: для ответственных узлов выполнить модальный КЭ-анализ (режим МКЭ).")
        return recs

    sigma_total = result["sigma_total"]
    safety = result["sigma_yield_t"] / sigma_total if sigma_total > 0 else float("inf")
    if safety < 1.0:
        recs.append("Напряжения превышают предел текучести: пересмотреть конструкцию, материал или схему закрепления.")
    elif safety < norm_coef:
        recs.append(f"Запас ниже нормативного n = {norm_coef:g}: уточнить расчёт или снизить нагрузку.")
    elif is_fem:
        recs.append("Запас достаточен по КЭ-расчёту; проверить сходимость по сетке (уменьшить размер КЭ и сравнить σmax).")
    else:
        recs.append("Запас достаточен по экспресс-оценке; для НТС подтвердить поверочным КЭ-расчётом (режим МКЭ).")

    if is_fem and result.get("sigma_p95") is not None and sigma_total > 3.0 * result["sigma_p95"] and sigma_total > 0:
        recs.append("σmax значительно выше 95-го перцентиля — локальная концентрация (возможно, сингулярность у закрепления): оценить линеаризацией или сгущением сетки.")
    if is_fem and result["analysis_type"] == "Спектральный":
        te = result.get("total_effective_mass_kg")
        mm_kg = result["mass_props"]["mass_kg"]
        axis_i = {"X": 0, "Y": 1, "Z": 2}.get(result.get("spectrum_axis", "X"), 0)
        if te and mm_kg > 0 and te[axis_i] / mm_kg < 0.9:
            recs.append(f"Эффективная масса по оси {result.get('spectrum_axis')} = {te[axis_i] / mm_kg * 100:.0f}% < 90%: часть отклика не учтена — увеличить число тонов или добавить поправку на остаточную массу.")
    if not is_fem and result["sigma_bending"] > result["sigma_membrane"]:
        recs.append("Преобладает изгиб: проверить плечо от зоны закрепления до зоны нагрузки и жёсткость сечения.")
    if float(params.get("temperature", 20)) > 150:
        recs.append(f"σт снижен по температуре до {result['sigma_yield_t']:.0f} МПа: проверить свойства материала по сертификату.")
    if not is_fem and result["sigma_thermal"] > 0:
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


def build_pdf_report_bytes(material_name, norm_name, norm_coef, report_rows, preview_png=None, use_fem=False):
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
    if use_fem:
        pdf.drawString(20 * mm, y, "Расчёт выполнен методом конечных элементов (CalculiX, тетраэдры C3D10 2-го порядка).")
    else:
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
    if use_fem:
        methodology_lines = [
            "1. Расчёт выполнен методом конечных элементов: решатель CalculiX, сетка gmsh из STEP-геометрии.",
            "2. Элементы — тетраэдры 2-го порядка (C3D10); линейно-упругая постановка.",
            "3. Гравитация и сейсмика — объёмные инерционные нагрузки; сейсмика задана квазистатически (ускорение в долях g).",
            "4. Сосредоточенная сила и давление распределены по узлам области пропорционально площадям.",
            "5. Масса содержимого — точечные массы на узлах области (участвуют в инерционных нагрузках и модальном анализе).",
            "6. Температурный расчёт — равномерный нагрев от 20 °C с учётом реального стеснения закреплениями.",
            "7. Модальный анализ: 10 тонов, частоты и эффективные модальные массы по осям X/Y/Z.",
            "8. Оценка прочности — по максимальным узловым напряжениям по Мизесу; в зонах закреплений возможны особенности решения.",
            "9. Предел текучести σт(T) интерполирован по справочной кривой; свойства материала уточнять по сертификату.",
            "10. Контакты и сварные швы не моделируются: тела в STEP сшиты жёстко (общие узлы на границах).",
        ]
    else:
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


def build_scenario_setup_figure(mesh, setup, max_faces=30_000):
    """Схема сценария на модели: зона закрепления (красный), область нагрузки (оранжевый),
    стрелки направления нагрузки и гравитации."""
    if mesh is None:
        return None
    coords = mesh.vertices[:, :3]
    faces = mesh.faces
    if len(faces) == 0 or len(coords) == 0:
        return None
    # упрощаем сетку, чтобы схема не подвешивала вкладку сценариев
    if len(faces) > max_faces:
        cache_key = (id(mesh), max_faces)
        if st.session_state.get('_setup_preview_key') == cache_key:
            preview = st.session_state.get('_setup_preview_mesh')
        else:
            try:
                preview = mesh.simplify_quadric_decimation(face_count=max_faces)
                if len(preview.faces) == 0:
                    preview = None
            except BaseException:
                preview = None
            st.session_state['_setup_preview_key'] = cache_key
            st.session_state['_setup_preview_mesh'] = preview
        if preview is not None:
            mesh = preview
            coords = mesh.vertices[:, :3]
            faces = mesh.faces
        else:
            # запасной вариант: прореживание треугольников (возможны «дыры» на схеме)
            faces = faces[np.linspace(0, len(faces) - 1, max_faces).astype(int)]
    used = np.unique(faces)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    v = coords[used]
    f = remap[faces]

    fixed_mask, fixed_center = get_constraint_zone(mesh, setup.get("constraint_face"), setup.get("constraint_zone_frac", 0.05))
    load_mask = get_region_mask(mesh, setup)
    load_center = get_region_centroid(mesh, load_mask)

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
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=inten[used], cmin=0.0, cmax=2.0,
        colorscale=colorscale, showscale=False,
        flatshading=True, hoverinfo='skip',
    ))

    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
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
            mode='lines+text', text=[label, ''], textposition='top center',
            textfont=dict(color=color, size=13),
            line=dict(color=color, width=7), hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[d[0]], v=[d[1]], w=[d[2]],
            sizemode='absolute', sizeref=arrow_len * 0.25, anchor='tip',
            colorscale=[[0, color], [1, color]], showscale=False, hoverinfo='skip',
        ))

    if setup.get("show_load_arrow", True):
        d_load = get_direction_vector(setup.get("direction", "-Z"))
        load_pts = coords[load_mask] if np.any(load_mask) else coords
        _add_arrow(_surface_tip(load_center, d_load, load_pts), d_load, "#e07b00", "нагрузка")
    if setup.get("show_gravity_arrow", False):
        d_g = np.array([0.0, 0.0, -1.0])
        g_tip = _surface_tip(mesh.centroid, d_g, coords)
        # разводим со стрелкой нагрузки, если они совпадают
        g_tip = g_tip + np.array([0.12, 0.0, 0.0]) * diag
        _add_arrow(g_tip, d_g, "#2b6cb0", "вес")

    # подпись зоны закрепления — с отступом наружу от грани
    axis_idx, side = CONSTRAINT_FACE_OPTIONS.get(setup.get("constraint_face"), (2, "min"))
    label_pos = np.asarray(fixed_center, dtype=float).copy()
    label_pos[axis_idx] += (-1.0 if side == "min" else 1.0) * diag * 0.08
    fig.add_trace(go.Scatter3d(
        x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
        mode='text', text=['закрепление'], textposition='middle center',
        textfont=dict(color="#d0021b", size=13), hoverinfo='skip', showlegend=False,
    ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=10),
        height=420,
    )
    return fig


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
    if FEM_AVAILABLE:
        solver_mode = st.radio(
            "Режим расчёта",
            ["МКЭ (CalculiX)", "Экспресс-оценка (аналитика)"],
            help="МКЭ — полноценный КЭ-расчёт: поля НДС, собственные частоты и эффективные массы. Требует STEP-файл.",
        )
        if solver_mode.startswith("МКЭ"):
            fem_mesh_size = st.slider("Размер КЭ для МКЭ (мм)", 2.0, 20.0, 8.0, 0.5,
                                      help="Тетраэдры 2-го порядка (C3D10). Меньше — точнее, но дольше счёт.")
        else:
            fem_mesh_size = 8.0
    else:
        solver_mode = "Экспресс-оценка (аналитика)"
        fem_mesh_size = 8.0
        st.markdown("*Режим: экспресс-оценка (CalculiX не найден — установите ccx для МКЭ)*")

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
                    _atype_opts = ["Статический", "Температурный", "Модальный", "Спектральный"]
                    _atype_default = 2 if "Модальный анализ" in scenario else 0
                    analysis_type = st.selectbox(f"Тип расчета ({scenario[:20]}...)", _atype_opts, index=_atype_default, key=f"analysis_type_{scenario}")
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
                    if analysis_type == "Спектральный":
                        st.selectbox(
                            f"Сочетание нагрузок ({scenario[:20]}...)",
                            ["НУЭ+ПЗ ([σ]×1.2)", "НУЭ+МРЗ ([σ]×1.4)"],
                            key=f"seism_combo_{scenario}",
                            help="Категория допускаемых по ПНАЭ Г-7-002-86: для сочетаний с ПЗ допускаемые напряжения повышаются в 1.2 раза, с МРЗ — в 1.4 раза.")
                        st.text_area(
                            f"Спектр ответа: f(Гц) Sa(g) по строке ({scenario[:20]}...)",
                            value="0.5 0.1\n2 0.5\n10 0.5\n33 0.2",
                            height=120, key=f"spectrum_{scenario}",
                            help="Огибающая поэтажного спектра ответа (ПЗ/МРЗ). Между точками — интерполяция по log f, за пределами — ближайшее значение. Ось возбуждения — «Направление нагрузки».")

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

                mesh_preview = st.session_state.get('active_mesh')
                if mesh_preview is None:
                    mesh_preview = st.session_state.get('stl_mesh')
                if mesh_preview is None:
                    st.info("Схема закреплений и нагрузок появится после загрузки модели на вкладке 1.")
                elif st.toggle(f"Схема закреплений и нагрузок ({scenario[:20]}...)", value=True, key=f"setup_viz_{scenario}"):
                    setup = {
                        "constraint_face": st.session_state.get(f"constraint_face_{scenario}", list(CONSTRAINT_FACE_OPTIONS.keys())[0]),
                        "constraint_zone_frac": float(st.session_state.get(f"constraint_zone_{scenario}", 5)) / 100.0,
                        "region_mode": region_mode,
                        "direction": st.session_state.get(f"direction_{scenario}", preset["default_direction"]),
                        "show_gravity_arrow": bool(st.session_state.get(f"gravity_{scenario}", True)),
                    }
                    if region_mode == "Пользовательская область":
                        setup["region_axis"] = st.session_state.get(f"region_axis_{scenario}", "Z")
                        rng = st.session_state.get(f"region_range_{scenario}")
                        if rng is not None:
                            setup["region_min"], setup["region_max"] = float(rng[0]), float(rng[1])
                    elif region_mode == "Точка с радиусом":
                        setup["selection_points"] = [[
                            float(st.session_state.get(f"region_px_{scenario}", 0.0)),
                            float(st.session_state.get(f"region_py_{scenario}", 0.0)),
                            float(st.session_state.get(f"region_pz_{scenario}", 0.0)),
                        ]]
                        setup["selection_radius"] = float(st.session_state.get(f"region_radius_{scenario}", 1.0))
                    fig_setup = build_scenario_setup_figure(mesh_preview, setup)
                    if fig_setup is not None:
                        st.plotly_chart(fig_setup, key=f"setup_chart_{scenario}")
                        st.caption("🔴 зона закрепления | 🟠 область приложения силы/давления | стрелка «нагрузка» — направление силы/сейсмики, «вес» — гравитация")

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

        use_fem = FEM_AVAILABLE and solver_mode.startswith("МКЭ") and st.session_state.get('step_bytes') is not None
        if FEM_AVAILABLE and solver_mode.startswith("МКЭ") and st.session_state.get('step_bytes') is None:
            st.warning("Режим МКЭ требует STEP-файл (для STL доступна только экспресс-оценка). Использована аналитика.")

        fem_mesh = None
        if use_fem:
            st.info("💡 Полноценный КЭ-расчёт (CalculiX): поля напряжений по Мизесу, перемещения, собственные частоты и эффективные массы.")
            fem_key = (st.session_state.get('stl_name'), round(fem_mesh_size, 2))
            if st.session_state.get('fem_mesh_key') != fem_key:
                with st.spinner(f"Построение КЭ-сетки C3D10 (размер {fem_mesh_size:g} мм)..."):
                    try:
                        st.session_state['fem_mesh'] = fem_solver.build_fem_mesh_subprocess(
                            st.session_state['step_bytes'], mesh_size_mm=fem_mesh_size)
                        st.session_state['fem_mesh_key'] = fem_key
                        st.session_state['fem_results_cache'] = {}
                    except Exception as exc:
                        st.error(f"Не удалось построить КЭ-сетку: {exc}")
                        use_fem = False
            fem_mesh = st.session_state.get('fem_mesh')
            if fem_mesh is not None and use_fem:
                st.caption(f"КЭ-сетка: {fem_mesh['n_nodes']:,} узлов, {fem_mesh['n_elements']:,} тетраэдров C3D10 (2-й порядок)")
        if not use_fem:
            st.info("💡 Экспресс-оценка аналитическими формулами: мембранные + изгибные + температурные напряжения, первая частота по методу Рэлея. Результаты — для предварительного сравнения сценариев, поверочный КЭ-расчёт они не заменяют.")

        coords = mesh.vertices[:, :3]
        scenario_results = []
        fem_cache = st.session_state.setdefault('fem_results_cache', {})

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
            if analysis_type == "Спектральный":
                combo = st.session_state.get(f'seism_combo_{scenario}', 'НУЭ+ПЗ ([σ]×1.2)')
                params["seism_combo"] = "НУЭ+МРЗ" if "МРЗ" in combo else "НУЭ+ПЗ"
                params["allow_factor"] = 1.4 if "МРЗ" in combo else 1.2
                spec_pts = []
                for ln in str(st.session_state.get(f'spectrum_{scenario}', '')).splitlines():
                    parts = ln.replace(',', ' ').replace(';', ' ').split()
                    if len(parts) >= 2:
                        try:
                            spec_pts.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                if spec_pts:
                    params["spectrum_points"] = spec_pts
            result = None
            if use_fem and fem_mesh is not None:
                cache_key = (scenario, selected_material, repr(sorted((k, str(v)) for k, v in params.items())))
                result = fem_cache.get(cache_key)
                if result is None:
                    with st.spinner(f"КЭ-расчёт CalculiX: {scenario}..."):
                        try:
                            result = fem_solver.solve_scenario_fem(fem_mesh, material_data, params)
                            fem_cache[cache_key] = result
                        except Exception as exc:
                            st.error(f"Ошибка КЭ-расчёта «{scenario}»: {exc}")
                            result = None
            if result is None:
                result = solve_scenario(mesh, material_data, params)
            if result.get("fem"):
                field = result.get("viz_field")
            elif result["analysis_type"] == "Модальный":
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
                allow_k = float(params.get("allow_factor", 1.0))
                norm_eff = norm_coef / allow_k
                sigma_allow = result["sigma_yield_t"] / norm_eff
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
                        "Критерий": f"[σ] = {sigma_allow:.1f} МПа" + (f" ({params['seism_combo']})" if allow_k > 1.0 else ""),
                        "Запас": round(safety_factor, 2) if np.isfinite(safety_factor) else "∞",
                        "Вердикт": "Пройдён" if safety_factor >= norm_eff else "Не пройдён",
                    })
            if comparison_rows:
                st.dataframe(comparison_rows, width='stretch', hide_index=True)

            for scenario, params, result, field in scenario_results:
                with st.expander(scenario, expanded=True):
                    col_info, col_plot = st.columns([1, 2])
                    is_modal = result["analysis_type"] == "Модальный"
                    is_fem = bool(result.get("fem"))
                    with col_info:
                        if is_modal:
                            f1 = result["first_frequency_hz"]
                            st.metric("Первая собственная частота", f"{f1:.1f} Гц" if f1 is not None else "н/д")
                            if is_fem:
                                st.caption(f"КЭ модальный анализ (CalculiX) | Масса: {result['total_mass_kg']:.1f} кг (модель {result['mass_props']['mass_kg']:.1f} кг)")
                            else:
                                st.caption(f"Балочная модель, {params['constraint_type'].lower()} | Масса: {result['total_mass_kg']:.1f} кг (модель {result['mass_props']['mass_kg']:.1f} кг)")
                            if f1 is not None and SEISMIC_BAND_HZ[0] <= f1 <= SEISMIC_BAND_HZ[1]:
                                st.error(f"Частота в сейсмическом диапазоне {SEISMIC_BAND_HZ[0]}–{SEISMIC_BAND_HZ[1]} Гц: возможен резонанс.")
                            else:
                                st.success("Частота вне типового сейсмического диапазона 0.5–33 Гц.")
                            if is_fem and result.get("modes"):
                                st.caption("Собственные частоты и эффективные массы:")
                                mode_rows = [{
                                    "Тон": m["mode"],
                                    "f, Гц": round(m["f_hz"], 1),
                                    "mx, кг": round(m["effmass_x_kg"], 3),
                                    "my, кг": round(m["effmass_y_kg"], 3),
                                    "mz, кг": round(m["effmass_z_kg"], 3),
                                } for m in result["modes"]]
                                st.dataframe(mode_rows, hide_index=True, height=240)
                                te = result.get("total_effective_mass_kg")
                                if te:
                                    mm_kg = result["mass_props"]["mass_kg"]
                                    st.caption(f"Σ эфф. масс (10 тонов): X {te[0]:.2f} / Y {te[1]:.2f} / Z {te[2]:.2f} кг из {mm_kg:.2f} кг ({te[0]/mm_kg*100:.0f}% / {te[1]/mm_kg*100:.0f}% / {te[2]/mm_kg*100:.0f}%)")
                        else:
                            if is_fem:
                                st.metric("Макс. напряжение (МКЭ, по Мизесу)", f"{result['sigma_total']:.1f} МПа")
                                st.caption(f"95-й перцентиль: {result['sigma_p95']:.1f} МПа | Макс. перемещение: {result['max_disp_mm'] * 1000:.1f} мкм")
                                if result["analysis_type"] == "Спектральный" and result.get("sigma_spectral") is not None:
                                    st.caption(f"Линейно-спектральный метод (SRSS, {len(result.get('modes', []))} тонов, ось {result.get('spectrum_axis', '?')}): σ спектр {result['sigma_spectral']:.1f} + σ НУЭ {result['sigma_static_part']:.1f} МПа")
                                    if result.get("modes"):
                                        sa_rows = [{"Тон": m["mode"], "f, Гц": round(m["f_hz"], 1), "Sa, g": round(m.get("sa_g", 0.0), 3)} for m in result["modes"]]
                                        with st.expander("Sa по тонам"):
                                            st.dataframe(sa_rows, hide_index=True, height=220)
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

                            allow_k = float(params.get("allow_factor", 1.0))
                            norm_eff = norm_coef / allow_k
                            sigma_allow = result["sigma_yield_t"] / norm_eff
                            combo_note = f" | сочетание {params['seism_combo']}: [σ]×{allow_k:g}" if allow_k > 1.0 else ""
                            st.caption(f"σт({params['temperature']} °C) = {result['sigma_yield_t']:.0f} МПа | [σ] = {sigma_allow:.0f} МПа (n = {norm_coef:g}){combo_note}")
                            safety_factor = result["sigma_yield_t"] / result["sigma_total"] if result["sigma_total"] > 0 else float("inf")
                            safety_text = f"{safety_factor:.2f}" if np.isfinite(safety_factor) else "∞"
                            status_text = "Пройдён" if safety_factor >= norm_eff else "Не пройдён"
                            status_color = "normal" if safety_factor >= norm_eff else "inverse"
                            st.metric("Запас прочности по σт(T)", safety_text, delta=status_text, delta_color=status_color)
                            if result["first_frequency_hz"] is not None:
                                st.caption(f"Первая частота (справочно): ≈ {result['first_frequency_hz']:.1f} Гц")
                            if is_fem and result.get("reactions_n"):
                                rf = result["reactions_n"]
                                st.caption(f"Реакции опор: ({rf[0]:,.0f}, {rf[1]:,.0f}, {rf[2]:,.0f}) Н")

                            if safety_factor < 1.0:
                                st.error("Напряжения превышают предел текучести: требуется пересмотр конструкции или нагрузки.")
                            elif safety_factor < norm_eff:
                                st.warning(f"Запас ниже требуемого {norm_eff:.2f}: требуется уточнение.")
                            else:
                                st.success("Сценарий допустим по экспресс-оценке.")

                        st.caption("Что проверить дальше:")
                        for rec in get_engineering_recommendations(result, params, norm_coef / float(params.get("allow_factor", 1.0))):
                            st.caption(f"• {rec}")

                    with col_plot:
                        if is_fem and field is not None:
                            vc = result["viz_coords"]
                            vt = result["viz_tris"]
                            if is_modal:
                                colorscale, cmin, cmax = 'Viridis', 0.0, 1.0
                                plot_title = f"Форма 1-го тона (МКЭ): {scenario}"
                                cbar_title = "|U| отн."
                            else:
                                colorscale, cmin, cmax = 'Jet', 0.0, max(color_max, 1e-6)
                                plot_title = f"Напряжения по Мизесу (МКЭ): {scenario}"
                                cbar_title = "σ, МПа"
                            fig = go.Figure(go.Mesh3d(
                                x=vc[:, 0], y=vc[:, 1], z=vc[:, 2],
                                i=vt[:, 0], j=vt[:, 1], k=vt[:, 2],
                                intensity=field, colorscale=colorscale,
                                cmin=cmin, cmax=cmax, showscale=True,
                                colorbar=dict(title=cbar_title),
                                lighting=dict(ambient=0.6, diffuse=0.8),
                            ))
                        else:
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
                        if not is_modal and not is_fem:
                            st.caption("Распределение качественное: мембранная и температурная части равномерны, изгибная нарастает к зоне закрепления. Максимум карты равен расчётному σmax.")

            with st.expander("Методика и допущения"):
                if use_fem:
                    st.markdown(
                        "- КЭ-расчёт: CalculiX (ccx), тетраэдры 2-го порядка C3D10, сетка gmsh из STEP-геометрии.\n"
                        "- Статика: линейно-упругий расчёт; гравитация и сейсмика — объёмные инерционные нагрузки; сила и давление распределяются по узлам области пропорционально площадям.\n"
                        "- Масса содержимого — точечные массы на узлах области нагрузки (участвуют в гравитации, сейсмике и модальном анализе).\n"
                        "- Температурный расчёт: равномерный нагрев от 20 °C до заданной T с закреплением — реальное стеснение расширения (не верхняя оценка).\n"
                        "- Модальный анализ: 10 тонов, частоты и эффективные модальные массы по X/Y/Z.\n"
                        "- Оценка прочности — по максимальным узловым напряжениям по Мизесу; в зонах закрепления возможны сингулярности — см. 95-й перцентиль и рекомендации.\n"
                        "- Спектральный тип: линейно-спектральный метод — отклик тона q = Γ·Sa(f)/ω² по огибающей поэтажного спектра, комбинация тонов SRSS, сложение с НУЭ — по полям Мизеса (консервативно); контролируйте полноту эффективных масс по оси возбуждения.\n"
                        "- Допускаемые для сейсмических сочетаний (ПНАЭ Г-7-002-86): НУЭ+ПЗ — [σ]×1.2, НУЭ+МРЗ — [σ]×1.4.\n"
                        "- Сейсмика в статических сценариях — квазистатически (ускорение в g).\n"
                        "- Сейсмика в статических сценариях — квазистатически (ускорение в g).\n"
                        "- Контакты и сварные швы не моделируются: несколько тел в STEP сшиваются жёстко (общие узлы)."
                    )
                else:
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
                            allow_k = float(params.get("allow_factor", 1.0))
                            norm_eff = norm_coef / allow_k
                            sigma_allow = result["sigma_yield_t"] / norm_eff
                            row = {
                                "scenario": scenario,
                                "analysis_type": result["analysis_type"],
                                "temperature": params['temperature'],
                                "result_text": f"{sigma_total:.1f} МПа",
                                "allow_text": f"{sigma_allow:.1f} МПа" + (f" ({params['seism_combo']})" if allow_k > 1.0 else ""),
                                "safety_text": f"{safety_factor:.2f}" if np.isfinite(safety_factor) else "∞",
                                "verdict": "Пройдён" if safety_factor >= norm_eff else "Не пройдён",
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
                                    preview_coords = result["viz_coords"] if result.get("fem") else coords
                                    preview_png = build_nds_preview_png(preview_coords, field, scenario)
                                    break

                        pdf_data = build_pdf_report_bytes(selected_material, norm_name, norm_coef, report_rows, preview_png=preview_png, use_fem=use_fem)
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
