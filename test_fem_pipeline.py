# Проверка пайплайна уровня 2: STEP -> gmsh (C3D10) -> CalculiX -> результаты
import os
import subprocess
import sys
import tempfile

import numpy as np

CCX = os.path.expanduser("~/.local/ccx-env/bin/ccx")
STEP_FILE = os.path.join(os.path.dirname(__file__), "65303.stp")


def mesh_step_with_gmsh(step_path, mesh_size_mm=8.0):
    """STEP -> тетраэдры 2-го порядка (C3D10). Возвращает узлы, элементы, грани CAD."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("part")
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(dim=3)
    if len(vols) > 1:
        # сшивка нескольких тел: общие узлы на контактных поверхностях
        gmsh.model.occ.fragment(vols, [])
        gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_mm * 0.25)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)  # квадратичные тетраэдры
    gmsh.model.mesh.optimize("HighOrderElastic")
    gmsh.model.mesh.optimize("HighOrder")

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = dict(zip(node_tags.astype(int), coords.reshape(-1, 3)))

    # объёмные элементы: тип 11 = 10-узловой тетраэдр
    etypes, etags, enodes = gmsh.model.mesh.getElements(dim=3)
    tets = None
    for et, tags, conn in zip(etypes, etags, enodes):
        if et == 11:
            conn = conn.astype(int).reshape(-1, 10)
            # gmsh: рёбра 01,12,02,03,23,13; Abaqus/ccx: 01,12,02,03,13,23 -> своп узлов 9 и 10
            conn = conn[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
            tets = (tags.astype(int), conn)
    assert tets is not None, "нет тетраэдров 2-го порядка"

    # узлы на каждой грани CAD (для BC по граням)
    face_nodes = {}
    for dim, tag in gmsh.model.getEntities(dim=2):
        ntags, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=tag, includeBoundary=True)
        face_nodes[tag] = set(ntags.astype(int))

    # bbox каждой грани, чтобы найти нижнюю
    face_bbox = {}
    for dim, tag in gmsh.model.getEntities(dim=2):
        face_bbox[tag] = gmsh.model.getBoundingBox(dim, tag)

    gmsh.finalize()
    return nodes, tets, face_nodes, face_bbox


def write_inp(path, nodes, tets, fixed_node_ids, density_t_mm3, e_mpa, nu):
    """Статика: гравитация -Z, заделка узлов fixed_node_ids. Единицы: мм, Н, МПа, т/мм3."""
    tet_ids, tet_conn = tets
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for nid, xyz in sorted(nodes.items()):
            f.write(f"{nid}, {xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}\n")
        f.write("*ELEMENT, TYPE=C3D10, ELSET=EALL\n")
        for eid, conn in zip(tet_ids, tet_conn):
            c = ", ".join(map(str, conn))
            f.write(f"{eid}, {c}\n")
        f.write("*NSET, NSET=FIX\n")
        ids = sorted(fixed_node_ids)
        for i in range(0, len(ids), 8):
            f.write(", ".join(map(str, ids[i:i + 8])) + "\n")
        f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n")
        f.write(f"{e_mpa:.1f}, {nu}\n")
        f.write("*DENSITY\n")
        f.write(f"{density_t_mm3:.6e}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
        f.write("*BOUNDARY\nFIX, 1, 3\n")
        f.write("*STEP\n*STATIC\n")
        f.write("*DLOAD\nEALL, GRAV, 9810., 0., 0., -1.\n")  # мм/с2
        f.write("*NODE FILE\nU\n*EL FILE\nS\n")
        f.write("*NODE PRINT, NSET=FIX, TOTALS=ONLY\nRF\n")
        f.write("*END STEP\n")


def parse_frd(frd_path):
    """Читает перемещения и напряжения из .frd. Возвращает dict node->(ux,uy,uz), node->6 компонент S."""
    disp, stress = {}, {}
    with open(frd_path) as f:
        lines = f.readlines()
    i = 0
    block = None
    while i < len(lines):
        line = lines[i]
        if line.startswith(" -4"):
            name = line.split()[1]
            block = name  # DISP / STRESS
        elif line.startswith(" -1") and block in ("DISP", "STRESS"):
            nid = int(line[3:13])
            vals = [float(line[13 + 12 * k:13 + 12 * (k + 1)]) for k in range((len(line.rstrip()) - 13) // 12)]
            if block == "DISP":
                disp[nid] = vals[:3]
            else:
                stress[nid] = vals[:6]
        elif line.startswith(" -3"):
            block = None
        i += 1
    return disp, stress


def von_mises(s):
    sx, sy, sz, sxy, syz, szx = s
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))


def main():
    print("1) Сетка gmsh из STEP...")
    nodes, tets, face_nodes, face_bbox = mesh_step_with_gmsh(STEP_FILE, mesh_size_mm=8.0)
    n_nodes, n_tets = len(nodes), len(tets[0])
    print(f"   узлов: {n_nodes}, C3D10: {n_tets}, граней CAD: {len(face_nodes)}")

    # нижние узлы: все узлы с Z в пределах 1 мм от минимума
    coords = np.array([nodes[k] for k in sorted(nodes)])
    ids_sorted = np.array(sorted(nodes))
    zmin = coords[:, 2].min()
    fixed = set(ids_sorted[coords[:, 2] < zmin + 1.0].tolist())
    print(f"   закреплено узлов (Z min): {len(fixed)}")

    with tempfile.TemporaryDirectory() as tmp:
        job = os.path.join(tmp, "job")
        print("2) Запись .inp и запуск CalculiX...")
        # 08Х18Н10Т: E=196 ГПа=196000 МПа, rho=7900 кг/м3 = 7.9e-9 т/мм3
        write_inp(job + ".inp", nodes, tets, fixed, 7.9e-9, 196000.0, 0.3)
        env = dict(os.environ, OMP_NUM_THREADS="4")
        r = subprocess.run([CCX, "-i", job], capture_output=True, text=True, env=env, cwd=tmp)
        if "Job finished" not in r.stdout:
            print(r.stdout[-3000:])
            print(r.stderr[-2000:])
            sys.exit(1)
        print("   ccx: Job finished")

        print("3) Разбор .frd...")
        disp, stress = parse_frd(job + ".frd")
        umag = {n: np.linalg.norm(v) for n, v in disp.items()}
        vm = {n: von_mises(s) for n, s in stress.items()}
        nmax_u = max(umag, key=umag.get)
        nmax_s = max(vm, key=vm.get)
        print(f"   max |u| = {umag[nmax_u] * 1000:.3f} мкм (узел {nmax_u})")
        print(f"   max von Mises = {vm[nmax_s]:.3f} МПа (узел {nmax_s}, xyz={nodes[nmax_s].round(1)})")

        # проверка реакции: должна равняться весу
        dat = open(job + ".dat").read()
        print("4) Контроль: суммарные реакции из .dat:")
        for ln in dat.splitlines():
            parts = ln.split()
            if len(parts) in (3, 4) and all(("e" in p.lower() or "." in p) for p in parts[-3:]):
                try:
                    fx, fy, fz = map(float, parts[-3:])
                except ValueError:
                    continue
                print(f"   RF = ({fx:.2f}, {fy:.2f}, {fz:.2f}) Н")
        print("   ожидание: RFz ~= +вес детали (~20.6 Н)")


if __name__ == "__main__":
    main()
