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
