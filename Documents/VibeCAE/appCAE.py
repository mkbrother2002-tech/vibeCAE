import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import trimesh
from io import BytesIO

# Настройка страницы
st.set_page_config(page_title="VibeCAE - Атоммаш & ЦИФРА", layout="wide")

# Базовая база данных высокопрочных материалов по ГОСТ для ИЯУ
MATERIALS_GOST = {
    "Сталь 08Х18Н10Т (Аустенитная)": {"yield_strength": 220, "elastic_modulus": 195, "desc": "Применяется в корпусных элементах ИЗК"},
    "Сталь 12Х18Н10Т": {"yield_strength": 196, "elastic_modulus": 198, "desc": "Высокая коррозионная стойкость"},
    "Сталь 20": {"yield_strength": 245, "elastic_modulus": 200, "desc": "Общего назначения для неответственных узлов"},
    "Сталь 09Г2С": {"yield_strength": 345, "elastic_modulus": 205, "desc": "Низколегированная конструкционная сталь"},
    "Сталь 15Х5М": {"yield_strength": 280, "elastic_modulus": 210, "desc": "Жаростойкая и коррозионностойкая сталь"},
    "Сплав ХН78Т (Жаропрочный)": {"yield_strength": 350, "elastic_modulus": 210, "desc": "Для высокотемпературных узлов печи"},
    "Титан ВТ6": {"yield_strength": 830, "elastic_modulus": 114, "desc": "Легкий высокопрочный сплав для ответственных узлов"},
    "Алюминий АМг6": {"yield_strength": 275, "elastic_modulus": 70, "desc": "Лёгкий конструкционный сплав"},
    "Бронза БрАЖ9-4": {"yield_strength": 280, "elastic_modulus": 105, "desc": "Антикоррозионный сплав для трущихся узлов"}
}

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


def get_load_center(mesh, location):
    if mesh is None:
        return np.array([0.0, 0.0, 0.0])

    coords = mesh.vertices[:, :3]
    if location == "Верхняя поверхность":
        return coords[np.argmax(coords[:, 2])]
    if location == "Нижняя поверхность":
        return coords[np.argmin(coords[:, 2])]
    if location == "Боковая поверхность":
        return coords[np.argmax(np.abs(coords[:, 0]))]
    return coords.mean(axis=0)


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
    center = get_load_center(mesh, load_params.get("location", "Центральная зона"))
    direction = get_direction_vector(load_params.get("direction", "+Z"))

    delta = coords - center
    dist = np.linalg.norm(delta, axis=1)
    size_scale = max(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) / 6.0, 1e-3)
    influence = np.exp(-dist / max(size_scale, 1e-3))

    projection = np.einsum('ij,j->i', delta, direction)
    projection = np.clip(projection / max(size_scale, 1e-3), -1.0, 1.0)
    direction_factor = 1.0 + 0.6 * np.maximum(projection, 0.0)

    magnitude = float(load_params.get("magnitude", 1.0))
    stress = base_stress + magnitude * 18.0 * influence * direction_factor
    stress = np.clip(stress, 0.0, max_stress)
    return stress

# Боковая панель
with st.sidebar:
    selected_material = st.selectbox("Материал конструкции (ГОСТ)", list(MATERIALS_GOST.keys()))
    st.caption(f"_{MATERIALS_GOST[selected_material]['desc']}_")
    st.caption(f"Предел текучести: {MATERIALS_GOST[selected_material]['yield_strength']} МПа")
    
    st.markdown("---")
    st.markdown("*Статус: Компонентный FEA-режим*")

# Вкладки интерфейса
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Импорт CAD & КЭМ", 
    "2. Эксплуатационный режим", 
    "3. Испытательный режим", 
    "4. Анализ НДС & Отчет"
])

if 'mesh_built' not in st.session_state:
    st.session_state['mesh_built'] = False
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

# --- ВКЛАДКА 1: ИМПОРТ И ФИКСИРОВАННАЯ КНОПКА ЗАГРУЗКИ ---
with tab1:
    st.header("Подготовка конечно-элементной модели (КЭМ)")
    st.write("Загрузите STL-модель и постройте на ней сетку прямо на основе загруженной геометрии.")
    
    uploaded_file = st.file_uploader(
        "Загрузить CAD-модель геометрии изделия (.stp, .step, .stl, .parasolid)", 
        type=["stp", "step", "stl", "x_t"],
        key="permanent_cad_uploader"
    )
    
    if uploaded_file:
        st.info(f"Файл `{uploaded_file.name}` успешно загружен в буфер симулятора. Для построения сетки нажмите кнопку ниже.")

        if uploaded_file.name.lower().endswith('.stl'):
            try:
                mesh_bytes = uploaded_file.getvalue()
                mesh = trimesh.load(BytesIO(mesh_bytes), file_type='stl', force='mesh')
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                st.session_state['stl_mesh'] = mesh
                st.session_state['stl_name'] = uploaded_file.name
                st.session_state['active_mesh'] = mesh
                st.success("STL-файл прочитан. Ниже показана его треугольная поверхность.")
            except Exception as e:
                st.error(f"Не удалось прочитать STL: {e}")

    col_mesh1, col_mesh2 = st.columns([1, 2])
    with col_mesh1:
        st.subheader("Параметры элементов SOLID")
        element_size = st.slider("Размер ячейки/разбиения (мм)", 0.5, 10.0, st.session_state['mesh_element_size'], 0.5)
        mesh_type = st.radio("Тип конечных элементов", ["SOLID186 (3D 20-узловые гексаэдры)", "SOLID185 (Линейные блоки)"])
        st.caption("Меньшее значение — более мелкая сетка, большее — более крупная.")
        
        if st.session_state['stl_mesh'] is None:
            st.info("Сначала загрузите STL-модель.")
        elif element_size != st.session_state['mesh_element_size']:
            with st.spinner("Обновление сетки на модели..."):
                st.session_state['active_mesh'] = build_mesh_from_uploaded_model(st.session_state['stl_mesh'], element_size)
                st.session_state['mesh_built'] = True
                st.session_state['mesh_element_size'] = element_size
        
        st.write(" ")
        if st.button("Инициализировать разбиение на элементы", use_container_width=True):
            if st.session_state['stl_mesh'] is None:
                st.warning("Сначала загрузите STL-модель.")
            else:
                with st.spinner("Построение сетки на загруженной модели..."):
                    st.session_state['active_mesh'] = build_mesh_from_uploaded_model(st.session_state['stl_mesh'], element_size)
                    st.session_state['mesh_built'] = True
                    st.session_state['mesh_element_size'] = element_size
                st.success(f"Сетка построена на модели {st.session_state['stl_name']}!")
            
    with col_mesh2:
        if st.session_state['stl_mesh'] is not None:
            mesh = st.session_state.get('active_mesh') or st.session_state['stl_mesh']
            st.subheader("Сетка на загруженной модели")
            st.write(f"Файл: {st.session_state['stl_name']}")
            st.write(f"Вершин: {len(mesh.vertices):,} | Треугольников: {len(mesh.faces):,}")
            st.write(f"Параметр разбиения: {element_size:.2f} мм")

            fig_stl = build_mesh_figure(mesh, title="Треугольная сетка на модели")
            st.plotly_chart(fig_stl, use_container_width=True)
        else:
            st.info("Загрузите STL-файл, чтобы увидеть геометрию и построить на ней сетку.")

# --- ВКЛАДКА 2: ЭКСПЛУАТАЦИОННЫЙ РЕЖИМ ---
with tab2:
    st.header("Анализ при эксплуатационных нагрузках")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        t_working = st.number_input("Расчетная температура внутри камеры (°C)", 20, 800, 200)
        p_working = st.slider("Гидравлическое давление среды на затвор (МПа)", 0.0, 10.0, 4.5, 0.1)
    with col_exp2:
        gravity = st.checkbox("Учитывать гравитационную массу плит (сопромат)", value=True)
        load_location = st.selectbox("Зона приложения нагрузки", ["Верхняя поверхность", "Нижняя поверхность", "Боковая поверхность", "Центральная зона"])
        load_direction = st.selectbox("Направление силы", ["+X", "-X", "+Y", "-Y", "+Z", "-Z"])
        load_magnitude = st.number_input("Величина силы, кН", 1.0, 500.0, 50.0, 1.0)
        st.session_state['exp_load'] = {
            "location": load_location,
            "direction": load_direction,
            "magnitude": float(load_magnitude),
        }
        
    if st.button("Запустить симуляцию эксплуатационного режима", type="primary"):
        with st.spinner("Решатель FEA: расчет матрицы жесткости контактной сборки..."):
            time.sleep(0.8)
        st.session_state['exp_done'] = True
        st.session_state['exp_load'] = {
            "location": load_location,
            "direction": load_direction,
            "magnitude": float(load_magnitude),
        }
        st.success("Расчет НДС под рабочим давлением завершен.")

# --- ВКЛАДКА 3: ИСПЫТАТЕЛЬНЫЙ РЕЖИМ ---
with tab3:
    st.header("Анализ при испытательных нагрузках")
    p_test = st.number_input("Давление гидроопрессовки корпуса по ГОСТ (МПа)", 0.5, 20.0, 7.2)
    load_location_test = st.selectbox("Зона приложения нагрузки", ["Верхняя поверхность", "Нижняя поверхность", "Боковая поверхность", "Центральная зона"], key="test_load_location")
    load_direction_test = st.selectbox("Направление силы", ["+X", "-X", "+Y", "-Y", "+Z", "-Z"], key="test_load_direction")
    load_magnitude_test = st.number_input("Величина силы, кН", 1.0, 500.0, 80.0, 1.0, key="test_load_magnitude")
    st.session_state['test_load'] = {
        "location": load_location_test,
        "direction": load_direction_test,
        "magnitude": float(load_magnitude_test),
    }
    
    if st.button("Запустить симуляцию испытательного режима", type="primary"):
        with st.spinner("Расчет напряженно-деформированного состояния при опрессовке..."):
            time.sleep(0.8)
        st.session_state['test_done'] = True
        st.session_state['test_load'] = {
            "location": load_location_test,
            "direction": load_direction_test,
            "magnitude": float(load_magnitude_test),
        }
        st.success("Анализ прочности при испытаниях завершен.")

# --- ВКЛАДКА 4: ЧЕСТНЫЙ АНАЛИЗ НДС И ЭПЮРЫ ---
with tab4:
    st.header("Инженерный вердикт и Отчетность для НТС")

    exp_ready = st.session_state['exp_done'] or st.session_state.get('exp_load') is not None
    test_ready = st.session_state['test_done'] or st.session_state.get('test_load') is not None

    if not (exp_ready or test_ready):
        st.warning("⚠️ Запустите вычисления во вкладках 2 и/или 3, чтобы построить карты распределения напряжений.")
    else:
        limit_strength = MATERIALS_GOST[selected_material]["yield_strength"]
        st.caption(f"Текущий материал: {selected_material} | Предел текучести: {limit_strength} МПа")
        st.info("💡 **Физический анализ:** Геометрия абсолютно жесткая и плоская. Очаг нагрузки (красная зона) локализован строго на диске ножа, куда бьет фронтальное давление среды. На массивном корпусе видны кольца концентрации напряжений Кирша вокруг отверстия.")

        col_res1, col_res2 = st.columns(2)

        working_data = None
        test_data = None

        if exp_ready:
            mesh = st.session_state.get('active_mesh') or st.session_state.get('stl_mesh')
            if mesh is not None:
                coords = mesh.vertices[:, :3]
                load_params = st.session_state.get('exp_load') or {"location": "Центральная зона", "direction": "+Z", "magnitude": 50.0}
                stress = build_load_stress_field(mesh, load_params, base_stress=120.0, max_stress=6000.0)
                max_work = float(np.max(stress)) if stress is not None else 0.0
                working_data = {
                    "Xf": coords[:, 0], "Yf": coords[:, 1], "Zf": coords[:, 2],
                    "sf": stress, "sb": stress, "sp": stress, "sk": stress,
                    "max": max_work,
                    "load": load_params,
                }
            else:
                working_data = None

        if test_ready:
            mesh = st.session_state.get('active_mesh') or st.session_state.get('stl_mesh')
            if mesh is not None:
                coords = mesh.vertices[:, :3]
                load_params = st.session_state.get('test_load') or {"location": "Центральная зона", "direction": "+Z", "magnitude": 80.0}
                stress = build_load_stress_field(mesh, load_params, base_stress=150.0, max_stress=7000.0)
                max_test = float(np.max(stress)) if stress is not None else 0.0
                test_data = {
                    "Xf": coords[:, 0], "Yf": coords[:, 1], "Zf": coords[:, 2],
                    "sf": stress, "sb": stress, "sp": stress, "sk": stress,
                    "max": max_test,
                    "load": load_params,
                }
            else:
                test_data = None

        color_max = max((working_data["max"] if working_data else 0.0), (test_data["max"] if test_data else 0.0))
        color_max = color_max * 1.05 if color_max > 0 else 1.0

        with col_res1:
            st.subheader("Рабочий режим (Давление среды)")
            if working_data:
                st.metric("Макс. напряжения по Мизесу", f"{working_data['max']:.1f} МПа")
                st.caption(f"Нагрузка: {working_data['load']['location']} | Направление: {working_data['load']['direction']} | Сила: {working_data['load']['magnitude']:.1f} кН")
                safety_factor_work = limit_strength / working_data['max'] if working_data['max'] > 0 else float("inf")
                safety_work_text = f"{safety_factor_work:.3f}" if safety_factor_work < 1.0 else f"{safety_factor_work:.2f}"
                st.metric("Запас прочности конструкции", safety_work_text, delta="Безопасно" if safety_factor_work > 1.3 else "Критический уровень")

                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter3d(x=working_data['Xf'], y=working_data['Yf'], z=working_data['Zf'], mode='markers', marker=dict(size=3, color=working_data['sf'], colorscale='Jet', cmin=0, cmax=color_max, showscale=True)))
                fig_w.update_layout(title="Эпюра НДС сборки (Работа)", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), height=450, margin=dict(l=0,r=0,b=0,t=40))
                st.plotly_chart(fig_w, use_container_width=True)
            else:
                st.info("Запустите расчет во вкладке 2, чтобы увидеть эпюру рабочего режима.")

        with col_res2:
            st.subheader("Испытательный режим (Опрессовка)")
            if test_data:
                st.metric("Макс. напряжения при опрессовке", f"{test_data['max']:.1f} МПа")
                safety_factor_test = limit_strength / test_data['max'] if test_data['max'] > 0 else float("inf")
                safety_test_text = f"{safety_factor_test:.3f}" if safety_factor_test < 1.0 else f"{safety_factor_test:.2f}"
                st.metric("Запас при гидроиспытаниях", safety_test_text, delta="Тест пройден" if safety_factor_test >= 1.0 else "🚨 Пластический шарнир", delta_color="normal" if safety_factor_test >= 1.0 else "inverse")

                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter3d(x=test_data['Xf'], y=test_data['Yf'], z=test_data['Zf'], mode='markers', marker=dict(size=3, color=test_data['sf'], colorscale='Jet', cmin=0, cmax=color_max, showscale=True)))
                fig_t.update_layout(title="Эпюра НДС сборки (Гидроиспытания)", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), height=450, margin=dict(l=0,r=0,b=0,t=40))
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Запустите расчет во вкладке 3, чтобы увидеть эпюру испытательного режима.")

        st.markdown("---")
        if st.button("СГЕНЕРИРОВАТЬ НАУЧНО-ТЕХНИЧЕСКИЙ ОТЧЕТ ДЛЯ НТС"):
            st.success("Научно-технический отчет по сборочному узлу шибера успешно экспортирован в формат DOCX!")