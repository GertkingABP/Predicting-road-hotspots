"""
main.py — главное окно приложения PyQt5
Система прогнозирования аварийных участков городской УДС
"""
import sys
import os
import numpy as np
from multiprocessing import freeze_support

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QSpinBox, QGroupBox, QSplitter, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from pipeline import run_pipeline, N_RUNS

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "model_meta.npz")
PT_PATH   = os.path.join(BASE_DIR, "model_gru_gcn_multi.pt")


# ── Поток для запуска пайплайна ───────────────────────────────────────────────
class PipelineWorker(QObject):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal     = pyqtSignal(object, object, object, object)  # juncs,xy,probs,roads
    error_signal    = pyqtSignal(str)

    def __init__(self, osm_path, n_runs):
        super().__init__()
        self.osm_path = osm_path
        self.n_runs   = n_runs

    def run(self):
        try:
            juncs, xy, probs, road_segments = run_pipeline(
                osm_path    = self.osm_path,
                meta_path   = META_PATH,
                pt_path     = PT_PATH,
                n_runs      = self.n_runs,
                log_cb      = lambda msg: self.log_signal.emit(msg),
                progress_cb = lambda cur, tot: self.progress_signal.emit(cur, tot),
            )
            self.done_signal.emit(juncs, xy, probs, road_segments)
        except Exception as e:
            self.error_signal.emit(str(e))


# ── Виджет карты (matplotlib) ─────────────────────────────────────────────────
class MapCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self._cbar = None   # храним colorbar, чтобы не плодить новые
        super().__init__(self.fig)
        self.setParent(parent)
        self._draw_placeholder()

    def _draw_placeholder(self):
        # Полностью очищаем фигуру (включая colorbar)
        self.fig.clf()
        self._cbar = None
        self.ax = self.fig.add_subplot(111)

        self.ax.set_facecolor("#F0F0F0")
        self.ax.text(
            0.5, 0.5,
            "Карта hotspot'ов появится здесь\nпосле завершения анализа",
            ha="center", va="center",
            fontsize=13, color="#999999",
            transform=self.ax.transAxes
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.draw()

    def plot_hotspots(self, juncs, xy, probs, road_segments, top_k=10):
        # Полностью пересоздаём фигуру — никаких задвоений colorbar
        self.fig.clf()
        self._cbar = None
        self.ax = self.fig.add_subplot(111)

        ax = self.ax
        ax.set_facecolor("#F7F5F2")   # тёплый фон, как у карт

        # ── 1. Дорожная сеть на фоне ────────────────────────────────────────
        if road_segments:
            segs = [[(x0, y0), (x1, y1)] for x0, y0, x1, y1 in road_segments]
            lc = LineCollection(
                segs,
                linewidths=0.8,
                colors="#BBBBBB",      # светло-серые дороги
                alpha=0.6,
                zorder=1
            )
            ax.add_collection(lc)

        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        # Вписываем координаты в границы дорог
        if road_segments:
            all_x = [x for x0,y0,x1,y1 in road_segments for x in (x0,x1)]
            all_y = [y for x0,y0,x1,y1 in road_segments for y in (y0,y1)]
            margin_x = (max(all_x) - min(all_x)) * 0.03
            margin_y = (max(all_y) - min(all_y)) * 0.03
            ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
            ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)

        # ── 2. Все перекрёстки — цветные точки ──────────────────────────────
        p_norm = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)
        sizes  = 40 + p_norm * 350

        sc = ax.scatter(
            x_coords, y_coords,
            s=sizes,
            c=probs,
            cmap="RdYlGn_r",
            vmin=probs.min(),
            vmax=probs.max(),
            alpha=0.85,
            zorder=3,
            edgecolors="white",
            linewidths=0.6
        )

        # ── 3. Top-K — красная обводка + номер ──────────────────────────────
        top_idx = np.argsort(-probs)[:top_k]
        for rank, idx in enumerate(top_idx):
            ax.scatter(
                x_coords[idx], y_coords[idx],
                s=sizes[idx] + 100,
                facecolors="none",
                edgecolors="#CC0000",
                linewidths=2.2,
                zorder=4
            )
            ax.annotate(
                str(rank + 1),
                (x_coords[idx], y_coords[idx]),
                fontsize=8, fontweight="bold",
                color="#990000",
                ha="center", va="bottom",
                xytext=(0, 9), textcoords="offset points",
                zorder=5
            )

        # ── 4. Единственная colorbar ─────────────────────────────────────────
        self._cbar = self.fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        self._cbar.set_label("Индекс аварийного риска (0–100)", fontsize=9)

        ax.set_title(
            f"Аварийно-опасные участки (Top-{top_k} выделены)",
            fontsize=11, fontweight="bold", pad=10
        )
        ax.set_xlabel("X (м)", fontsize=9)
        ax.set_ylabel("Y (м)", fontsize=9)
        ax.tick_params(labelsize=8)
        self.fig.tight_layout()
        self.draw()


# ── Главное окно ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система прогнозирования УДС")
        self.setMinimumSize(1100, 700)
        self._worker   = None
        self._thread   = None
        self._results  = None
        self._road_seg = None

        self._build_ui()
        self._check_model_files()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ── ЛЕВАЯ ПАНЕЛЬ ────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(340)
        left.setMinimumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)

        grp_file = QGroupBox("Входные данные")
        grp_file.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QVBoxLayout(grp_file)

        self.lbl_osm = QLabel("Файл не выбран")
        self.lbl_osm.setWordWrap(True)
        self.lbl_osm.setStyleSheet(
            "color: #555; background: #F5F5F5; "
            "padding: 6px; border-radius: 4px; font-size: 11px;"
        )
        file_layout.addWidget(self.lbl_osm)

        btn_open = QPushButton("Выбрать map.osm")
        btn_open.setStyleSheet(self._btn_style("#2E74B5"))
        btn_open.clicked.connect(self._open_osm)
        file_layout.addWidget(btn_open)
        left_layout.addWidget(grp_file)

        grp_params = QGroupBox("Параметры симуляции")
        grp_params.setStyleSheet("QGroupBox { font-weight: bold; }")
        params_layout = QVBoxLayout(grp_params)

        params_layout.addWidget(QLabel("Число прогонов SUMO:"))

        self.spin_runs = QSpinBox()
        self.spin_runs.setRange(5, 50)
        self.spin_runs.setValue(N_RUNS)
        self.spin_runs.setSuffix(" прогонов")
        self.spin_runs.setToolTip("Рекомендуется: 10-20")
        params_layout.addWidget(self.spin_runs)
        left_layout.addWidget(grp_params)

        self.btn_run = QPushButton("▶  Запустить анализ")
        self.btn_run.setStyleSheet(self._btn_style("#375623", size=13))
        self.btn_run.setMinimumHeight(44)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._start_pipeline)
        left_layout.addWidget(self.btn_run)

        grp_prog = QGroupBox("Прогресс")
        grp_prog.setStyleSheet("QGroupBox { font-weight: bold; }")
        prog_layout = QVBoxLayout(grp_prog)

        self.lbl_stage = QLabel("Ожидание запуска...")
        self.lbl_stage.setStyleSheet("font-size: 11px; color: #444;")
        prog_layout.addWidget(self.lbl_stage)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet(
            "QProgressBar { border-radius: 4px; background: #EEE; height: 18px; }"
            "QProgressBar::chunk { background: #2E74B5; border-radius: 4px; }"
        )
        prog_layout.addWidget(self.progress)
        left_layout.addWidget(grp_prog)

        self.btn_save = QPushButton("💾  Сохранить карту")
        self.btn_save.setStyleSheet(self._btn_style("#7F3F00"))
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_map)
        left_layout.addWidget(self.btn_save)

        left_layout.addStretch()

        grp_log = QGroupBox("Журнал выполнения")
        grp_log.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(grp_log)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 10px; "
            "background: #1E1E1E; color: #D4D4D4; border-radius: 4px;"
        )
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(grp_log)

        splitter.addWidget(left)

        # ── ПРАВАЯ ПАНЕЛЬ ───────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas  = MapCanvas(self)
        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("font-size: 11px;")

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([320, 780])

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Готов к работе. Выберите файл map.osm для начала анализа.")

    def _btn_style(self, color: str, size: int = 11) -> str:
        return (
            f"QPushButton {{ background-color: {color}; color: white; "
            f"border-radius: 5px; padding: 7px; font-size: {size}px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {color}CC; }}"
            f"QPushButton:disabled {{ background-color: #AAAAAA; }}"
        )

    def _check_model_files(self):
        missing = [f for f in ["model_meta.npz", "model_gru_gcn_multi.pt"]
                   if not os.path.exists(os.path.join(BASE_DIR, f))]
        if missing:
            self._log(f"⚠ Файлы модели не найдены: {', '.join(missing)}")
            self.status.showMessage("Ошибка: файлы модели не найдены!")
        else:
            self._log("✓ Файлы модели загружены успешно.")

    def _open_osm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл карты", "",
            "OpenStreetMap файлы (*.osm *.xml);;Все файлы (*)"
        )
        if path:
            self.osm_path = path
            name = os.path.basename(path)
            size = os.path.getsize(path) // 1024
            self.lbl_osm.setText(f"📂 {name}\n({size} КБ)")
            self.btn_run.setEnabled(True)
            self.status.showMessage(f"Файл выбран: {name}")
            self._log(f"Выбран файл: {path}")

    def _log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _start_pipeline(self):
        if not hasattr(self, "osm_path"):
            return

        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_stage.setText("Запуск пайплайна...")
        self.canvas._draw_placeholder()
        self._log("\n" + "─" * 40)
        self._log("Запуск анализа...")

        self._thread = QThread()
        self._worker = PipelineWorker(self.osm_path, self.spin_runs.value())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_signal.connect(self._on_log)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.error_signal.connect(self._on_error)
        self._worker.done_signal.connect(self._thread.quit)
        self._worker.error_signal.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    def _on_log(self, msg: str):
        self._log(msg)
        self.lbl_stage.setText(msg[:60] + ("..." if len(msg) > 60 else ""))
        self.status.showMessage(msg)

    def _on_progress(self, cur: int, total: int):
        if total > 0:
            self.progress.setValue(int(cur / total * 100))

    def _on_done(self, juncs, xy, probs, road_segments):
        self._results  = (juncs, xy, probs)
        self._road_seg = road_segments
        self.progress.setValue(100)
        self.lbl_stage.setText("Анализ завершён!")
        self.status.showMessage(
            f"Готово! Найдено {len(juncs)} перекрёстков. "
            f"Топ-1 hotspot: индекс риска {probs.max():.1f}/100"
        )
        self._log(f"\n✓ Анализ завершён. Перекрёстков: {len(juncs)}")
        self._log("  Топ-3 по индексу риска:")
        for rank, idx in enumerate(np.argsort(-probs)[:3]):
            self._log(f"    {rank+1}. {juncs[idx][:50]}... риск={probs[idx]:.1f}/100")

        top_k = min(10, len(juncs))
        self.canvas.plot_hotspots(juncs, xy, probs, road_segments, top_k=top_k)
        self.btn_save.setEnabled(True)

    def _on_error(self, msg: str):
        self._log(f"\n✗ ОШИБКА: {msg}")
        self.lbl_stage.setText("Ошибка!")
        self.status.showMessage(f"Ошибка: {msg[:80]}")

    def _on_thread_finished(self):
        self.btn_run.setEnabled(True)

    def _save_map(self):
        if self._results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить карту", "hotspot_map.png",
            "PNG изображение (*.png);;PDF документ (*.pdf)"
        )
        if path:
            self.canvas.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Карта сохранена: {path}")
            self.status.showMessage(f"Карта сохранена: {path}")


# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    freeze_support()
    main()