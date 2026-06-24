"""
pipeline.py — полный пайплайн: map.osm → hotspot-карта
Прогоны SUMO выполняются параллельно через ProcessPoolExecutor.
"""
import os
import sys
import gzip
import shutil
import subprocess
import tempfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from xml.etree import ElementTree as ET
from typing import Callable, Optional
import torch

from model import load_model

# ── Константы ─────────────────────────────────────────────────────────────────
FREQ      = 60
SIM_END   = 3600
T_BINS    = SIM_END // FREQ
STOPTIME  = 5.0
H_WIN     = 6
TOP_N     = 30
N_RUNS    = 15
MAX_WORKERS = 4   # параллельных прогонов SUMO (оставляем 4 ядра ОС/UI)

FEAT_COLS = [
    "speed", "density", "occupancy", "waitingTime", "timeLoss",
    "traveltime", "entered", "left", "laneChangedFrom", "laneChangedTo"
]
F_DIM = len(FEAT_COLS)

SUMO_HOME    = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
NETCONVERT   = os.path.join(SUMO_HOME, "bin", "netconvert.exe")
SUMO_BIN     = os.path.join(SUMO_HOME, "bin", "sumo.exe")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")


# ── Утилиты ───────────────────────────────────────────────────────────────────
def log(msg: str, callback: Optional[Callable] = None):
    print(msg)
    if callback:
        callback(msg)


def normalize_adj(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    deg = A.sum(axis=1)
    d = 1.0 / np.sqrt(np.clip(deg, 1e-6, None))
    return (d[:, None] * A) * d[None, :]


def lane_to_junction(lane: str) -> str:
    s = lane.lstrip(":")
    parts = s.split("_")
    cut = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            cut = i
        else:
            break
    return "_".join(parts[:cut]) if cut >= 1 else s


# ── Шаг 1: OSM → net.xml ─────────────────────────────────────────────────────
def osm_to_net(osm_path: str, work_dir: str,
               log_cb: Optional[Callable] = None) -> str:
    net_path = os.path.join(work_dir, "map.net.xml")
    cmd = [
        NETCONVERT,
        "--osm-files", osm_path,
        "-o", net_path,
        "--output.street-names", "true",
        "--proj.utm", "true",
        "--roundabouts.guess", "true",
        "--tls.discard-simple", "true",
        "--tls.join", "true",
        "--tls.guess-signals", "true",
        "--ramps.guess", "true",
        "--remove-edges.isolated", "true",
        "--junctions.join", "true",
        "--junctions.corner-detail", "10",
        "--junctions.internal-link-detail", "10",
        "--osm.all-attributes", "true",
        "--no-warnings", "true",
    ]
    log("  Запуск netconvert...", log_cb)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert завершился с ошибкой:\n{result.stderr[-800:]}")
    if not os.path.exists(net_path):
        raise RuntimeError("netconvert не создал map.net.xml")
    log("  map.net.xml создан.", log_cb)
    return net_path


# ── Шаг 2: Парсинг net.xml ───────────────────────────────────────────────────
def parse_net(net_path: str):
    """
    Возвращает:
      juncs         — список cluster junction ID (до TOP_N штук)
      jidx          — {junction_id: index}
      A_norm        — нормализованная матрица смежности (N,N)
      edge_to_junc  — {edge_id: cluster_junction_id}
      xy            — координаты junction (N,2)
      road_segments — [(x0,y0,x1,y1), ...] все рёбра для фона карты
    """
    all_juncs    = {}   # jid -> (x, y)  — только не-internal
    all_junc_xy  = {}   # jid -> (x, y)  — все, включая internal (для рёбер)
    edge_endpoints = {} # eid -> (from, to)

    for event, elem in ET.iterparse(net_path, events=("end",)):
        if elem.tag == "junction":
            jid = elem.get("id", "")
            try:
                x = float(elem.get("x", 0))
                y = float(elem.get("y", 0))
                all_junc_xy[jid] = (x, y)
            except:
                pass
            jtype = elem.get("type", "")
            if not jid.startswith(":") and jtype not in ("internal", "dead_end"):
                try:
                    all_juncs[jid] = (float(elem.get("x", 0)),
                                      float(elem.get("y", 0)))
                except:
                    pass
        elif elem.tag == "edge":
            eid = elem.get("id", "")
            if not eid.startswith(":"):
                fr = elem.get("from", "")
                to = elem.get("to",   "")
                if fr and to:
                    edge_endpoints[eid] = (fr, to)
        elem.clear()

    cluster_juncs = sorted([j for j in all_juncs if j.startswith("cluster")])
    normal_juncs  = {j for j in all_juncs if not j.startswith("cluster")}

    juncs = cluster_juncs[:TOP_N]
    jidx  = {j: i for i, j in enumerate(juncs)}
    N     = len(juncs)

    if N == 0:
        raise RuntimeError(
            "В дорожной сети не найдено ни одного cluster-перекрёстка. "
            "Попробуйте выбрать более крупный район с перекрёстками."
        )

    c_xy = np.array([all_juncs[j] for j in juncs], dtype=np.float32)
    normal_to_cluster = {}
    for nj in normal_juncs:
        if nj not in all_juncs:
            continue
        nx, ny = all_juncs[nj]
        dists  = np.sqrt((c_xy[:, 0] - nx)**2 + (c_xy[:, 1] - ny)**2)
        best   = int(np.argmin(dists))
        if dists[best] <= 150.0:
            normal_to_cluster[nj] = juncs[best]

    edge_to_junc = {}
    A = np.zeros((N, N), dtype=np.float32)
    for eid, (fr_raw, to_raw) in edge_endpoints.items():
        to_c = to_raw if to_raw in jidx else normal_to_cluster.get(to_raw)
        fr_c = fr_raw if fr_raw in jidx else normal_to_cluster.get(fr_raw)
        if to_c:
            edge_to_junc[eid] = to_c
        if fr_c and to_c and fr_c in jidx and to_c in jidx:
            A[jidx[fr_c], jidx[to_c]] = 1.0

    np.fill_diagonal(A, 1.0)
    A_norm = normalize_adj(A)
    xy = np.array([all_juncs[j] for j in juncs], dtype=np.float32)

    road_segments = []
    for eid, (fr, to) in edge_endpoints.items():
        p0 = all_junc_xy.get(fr)
        p1 = all_junc_xy.get(to)
        if p0 and p1:
            road_segments.append((p0[0], p0[1], p1[0], p1[1]))

    return juncs, jidx, A_norm, edge_to_junc, xy, road_segments


# ── Шаг 3: Генерация конфигов SUMO ───────────────────────────────────────────
def make_sumo_configs(net_path: str, work_dir: str,
                      log_cb: Optional[Callable] = None) -> str:
    """
    Все пути в sumocfg относительные от work_dir.
    SUMO запускается с cwd=work_dir.
    """
    log("  Генерация маршрутов...", log_cb)

    python = sys.executable

    vtypes_path = os.path.join(work_dir, "vtypes.add.xml")
    with open(vtypes_path, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <vTypeDistribution id="carDist">
    <vType id="car_norm" vClass="passenger" maxSpeed="13.9" accel="2.5"
           decel="4.5" sigma="0.5" length="4.5" probability="0.7"/>
    <vType id="car_aggr" vClass="passenger" maxSpeed="19.4" accel="3.0"
           decel="5.0" sigma="0.9" length="4.5" probability="0.3"/>
  </vTypeDistribution>
  <vTypeDistribution id="motoDist">
    <vType id="moto_norm" vClass="motorcycle" maxSpeed="16.7" accel="3.5"
           decel="5.0" sigma="0.7" length="2.2" probability="1.0"/>
  </vTypeDistribution>
  <vTypeDistribution id="truckDist">
    <vType id="truck_norm" vClass="truck" maxSpeed="11.1" accel="1.5"
           decel="3.5" sigma="0.4" length="8.0" probability="1.0"/>
  </vTypeDistribution>
</additional>
""")

    outputs_path = os.path.join(work_dir, "outputs.add.xml")
    with open(outputs_path, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <edgeData id="ed" freq="60" file="edgeData.xml"
            excludeEmpty="true" speedThreshold="-1"/>
</additional>
""")

    for prefix, period, attr in [
        ("car",   1.5,  "type='carDist'"),
        ("moto",  4.0,  "type='motoDist'"),
        ("truck", 10.0, "type='truckDist'"),
    ]:
        cmd = [
            python, RANDOM_TRIPS,
            "-n", "map.net.xml",
            "-e", str(SIM_END),
            "-p", str(period),
            "-o", f"trips_{prefix}.xml",
            "-r", f"routes_{prefix}.rou.xml",
            "--seed", "42",
            "--prefix", prefix,
            "--trip-attributes",
            f"{attr} departLane='best' departSpeed='max'",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"randomTrips.py ошибка ({prefix}):\n{result.stderr[-500:]}"
            )

    cfg_path = os.path.join(work_dir, "run.sumocfg")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="map.net.xml"/>
    <route-files value="routes_car.rou.xml,routes_moto.rou.xml,routes_truck.rou.xml"/>
    <additional-files value="vtypes.add.xml,outputs.add.xml"/>
  </input>
  <output>
    <tripinfo-output value="tripinfo.xml"/>
    <collision-output value="collisions.xml.gz"/>
  </output>
  <time>
    <begin value="0"/>
    <end value="{SIM_END}"/>
    <step-length value="0.2"/>
  </time>
  <processing>
    <collision.action value="warn"/>
    <collision.stoptime value="5"/>
    <collision.check-junctions value="true"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <no-warnings value="true"/>
  </report>
</configuration>
""")

    log("  Конфиги SUMO созданы.", log_cb)
    return cfg_path


# ── Шаг 4: Один прогон SUMO (вызывается в отдельном процессе) ────────────────
def _run_single_seed(args: tuple) -> tuple:
    """
    Запускает один прогон SUMO. Выполняется в дочернем процессе —
    НЕ использует log_cb (нельзя передавать через границу процессов).
    Возвращает (seed, ok: bool, returncode: int, stderr_tail: str).
    """
    seed, work_dir, sumo_bin = args
    out_prefix = os.path.join("runs", f"seed{seed}_")
    cmd = [
        sumo_bin,
        "-c", "run.sumocfg",
        "--seed", str(seed),
        "--output-prefix", out_prefix,
        "--quit-on-end",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    edge_path = os.path.join(work_dir, "runs", f"seed{seed}_edgeData.xml")
    ok = os.path.exists(edge_path)
    return seed, ok, proc.returncode, proc.stderr.strip()[-200:]


# ── Шаг 4 (параллельный): Запуск всех прогонов SUMO ─────────────────────────
def run_sumo(cfg_path: str, work_dir: str, n_runs: int = N_RUNS,
             log_cb=None, progress_cb=None) -> list:
    """
    Запускает n_runs прогонов SUMO параллельно (до MAX_WORKERS одновременно).
    По мере завершения каждого прогона обновляет прогресс-бар.
    """
    runs_dir = os.path.join(work_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    seeds = list(range(101, 101 + n_runs))
    workers = min(MAX_WORKERS, n_runs)
    log(f"  Параллельных процессов SUMO: {workers} (из {n_runs} прогонов)", log_cb)

    args_list = [(seed, work_dir, SUMO_BIN) for seed in seeds]

    completed  = 0
    result_files = []

    # ProcessPoolExecutor — каждый прогон в отдельном процессе Python,
    # который запускает subprocess SUMO. GIL не мешает, I/O не блокирует UI.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_seed = {
            executor.submit(_run_single_seed, args): args[0]
            for args in args_list
        }
        for future in as_completed(future_to_seed):
            seed, ok, code, stderr_tail = future.result()
            completed += 1

            if progress_cb:
                progress_cb(completed, n_runs)

            if ok:
                edge_path = os.path.join(runs_dir, f"seed{seed}_edgeData.xml")
                col_path  = os.path.join(runs_dir, f"seed{seed}_collisions.xml.gz")
                result_files.append((edge_path, col_path))
                log(f"  ✓ Прогон seed={seed} завершён ({completed}/{n_runs})", log_cb)
            else:
                log(f"  ✗ Прогон seed={seed} — ошибка (код {code})", log_cb)
                if stderr_tail:
                    log(f"    {stderr_tail}", log_cb)

    log(f"  Успешно: {len(result_files)}/{n_runs} прогонов.", log_cb)

    if len(result_files) == 0:
        raise RuntimeError(
            "Ни один прогон SUMO не завершился успешно. "
            "Проверьте, что SUMO_HOME задан верно и SUMO установлен."
        )

    return result_files


# ── Шаг 5: Парсинг одного edgeData.xml (используется параллельно) ────────────
def _parse_single_edgedata(args: tuple) -> np.ndarray:
    """Вызывается в ThreadPoolExecutor — чистый I/O, GIL не проблема."""
    path, jidx, edge_to_junc = args
    N = len(jidx)
    accum  = np.zeros((T_BINS, N, F_DIM), dtype=np.float64)
    counts = np.zeros((T_BINS, N),        dtype=np.int32)

    tree = ET.parse(path)
    root = tree.getroot()
    for interval in root.findall("interval"):
        begin = float(interval.get("begin", 0))
        tb = min(int(begin // FREQ), T_BINS - 1)
        for edge in interval.findall("edge"):
            ss = float(edge.get("sampledSeconds", 0) or 0)
            if ss <= 0:
                continue
            eid = edge.get("id", "")
            jid = edge_to_junc.get(eid)
            if jid and jid in jidx:
                ni  = jidx[jid]
                row = [float(edge.get(f, 0) or 0) for f in FEAT_COLS]
                accum[tb, ni]  += row
                counts[tb, ni] += 1

    cnt = np.maximum(counts[:, :, None], 1)
    return (accum / cnt).astype(np.float32)


# ── Шаг 6: Инференс ──────────────────────────────────────────────────────────
def run_inference(result_files: list, jidx: dict, edge_to_junc: dict,
                  A_norm: np.ndarray, meta_path: str, pt_path: str,
                  log_cb: Optional[Callable] = None) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    meta = np.load(meta_path, allow_pickle=True)
    mu   = meta["mu"].astype(np.float32)
    std  = meta["std"].astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)

    A_t = torch.tensor(A_norm, dtype=torch.float32)

    log("  Загрузка модели...", log_cb)
    model = load_model(pt_path)

    # Параллельный парсинг edgeData — I/O bound, потоки справляются
    log(f"  Параллельный парсинг {len(result_files)} файлов edgeData...", log_cb)
    edge_paths = [ep for ep, _ in result_files if os.path.exists(ep)]
    args_list  = [(p, jidx, edge_to_junc) for p in edge_paths]

    X_list = []
    parse_workers = min(8, len(args_list))
    with ThreadPoolExecutor(max_workers=parse_workers) as tex:
        for X in tex.map(_parse_single_edgedata, args_list):
            X_list.append(X)

    if not X_list:
        raise RuntimeError("Нет данных для инференса.")

    # Инференс по всем прогонам и окнам
    log(f"  Инференс ({len(X_list)} прогонов × {T_BINS - H_WIN} окон)...", log_cb)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for X in X_list:
            X_norm = (X - mu) / std   # (T, N, F)

            # Собираем все окна одного прогона в батч — быстрее одиночных вызовов
            windows = np.stack(
                [X_norm[t - H_WIN:t] for t in range(H_WIN, T_BINS)]
            )  # (T-H, H, N, F)
            Xb = torch.tensor(windows, dtype=torch.float32)   # (T-H, H, N, F)
            logits = model(Xb, A_t)                            # (T-H, N)
            probs  = torch.sigmoid(logits).numpy()             # (T-H, N)
            all_probs.append(probs)

    probs_all  = np.concatenate(all_probs, axis=0)   # (total_windows, N)
    probs_mean = probs_all.mean(axis=0)               # (N,)

    # Переводим в индекс риска 0–100: 100 = самый опасный перекрёсток
    p_min, p_max = probs_mean.min(), probs_mean.max()
    risk_index = (probs_mean - p_min) / (p_max - p_min + 1e-9) * 100.0

    log(f"  Инференс завершён. Всего окон: {len(probs_all)}", log_cb)
    log(f"  Индекс риска: min={risk_index.min():.1f}, "
        f"max={risk_index.max():.1f}, mean={risk_index.mean():.1f}", log_cb)
    return risk_index


# ── Главная функция пайплайна ─────────────────────────────────────────────────
def run_pipeline(osm_path: str, meta_path: str, pt_path: str,
                 n_runs: int = N_RUNS,
                 log_cb: Optional[Callable] = None,
                 progress_cb: Optional[Callable] = None):
    """
    Полный пайплайн: OSM → hotspot-вероятности.
    Возвращает (juncs, xy, probs, road_segments).
    """
    work_dir = tempfile.mkdtemp(prefix="hotspot_")
    log(f"Рабочая директория: {work_dir}", log_cb)

    try:
        log("[1/5] Конвертация OSM → дорожная сеть...", log_cb)
        net_path = osm_to_net(osm_path, work_dir, log_cb)

        log("[2/5] Построение графа перекрёстков...", log_cb)
        juncs, jidx, A_norm, edge_to_junc, xy, road_segments = parse_net(net_path)
        log(f"  Найдено {len(juncs)} cluster-перекрёстков, "
            f"{len(road_segments)} дорожных сегментов.", log_cb)

        log("[3/5] Подготовка сценария симуляции...", log_cb)
        cfg_path = make_sumo_configs(net_path, work_dir, log_cb)

        log(f"[4/5] Запуск {n_runs} прогонов SUMO (параллельно)...", log_cb)
        result_files = run_sumo(cfg_path, work_dir, n_runs, log_cb, progress_cb)

        log("[5/5] Прогнозирование hotspot'ов...", log_cb)
        probs = run_inference(
            result_files, jidx, edge_to_junc,
            A_norm, meta_path, pt_path, log_cb
        )

        log("Готово!", log_cb)
        return juncs, xy, probs, road_segments

    except Exception:
        raise
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except:
            pass