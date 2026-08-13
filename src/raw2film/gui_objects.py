"""Additional GUI objects used by Raw2Film."""

import queue
import threading
import time

from PyQt6.QtCore import (
    QObject,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import QDialog, QLineEdit, QTextEdit, QVBoxLayout
from spectral_film_lut.gui_objects import AnimatedButton, Slider, SliderLog


class EditableSliderMixin:
    """Adds an editable value field and double-click-to-reset behavior to a slider."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._make_value_field_editable()
        self.slider.mouseDoubleClickEvent = self._reset_to_default

    def _make_value_field_editable(self):
        self.layout.removeWidget(self.text)
        self.text.deleteLater()

        value_field = QLineEdit()
        value_field.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        value_field.setFixedWidth(54)
        value_field.setFrame(False)
        value_field.editingFinished.connect(self._apply_typed_value)
        self.layout.addWidget(value_field)
        self.text = value_field
        self.update_text_display()

    def _apply_typed_value(self):
        try:
            self.setValue(float(self.text.text()))
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
        self.update_text_display()

    def _reset_to_default(self, event):
        self.slider.setValue(self.slider.reference_value)
        event.accept()


class EditableSlider(EditableSliderMixin, Slider):
    """A linear slider with an editable value field and double-click reset."""


class EditableSliderLog(EditableSliderMixin, SliderLog):
    """A logarithmic slider with an editable value field and double-click reset."""


class CpuWorker(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_canceled = False

    @pyqtSlot()
    def cancel(self):
        self._is_canceled = True

    def run_tasks(self, func_execute_cpu, tasks, **kwargs):
        self._is_canceled = False
        total = len(tasks)

        start = time.time()

        for idx, task in enumerate(tasks, 1):
            if self._is_canceled:
                break
            status_msg = func_execute_cpu(
                task, current_idx=idx, total_count=total, **kwargs
            )
            self.progress.emit(status_msg, idx)

        print(f"total {time.time() - start:.2f}s")

        self.finished.emit()


class GpuWorker(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_canceled = False
        self.payload_queue = None

    @pyqtSlot()
    def cancel(self):
        self._is_canceled = True
        if self.payload_queue is not None:
            try:
                self.payload_queue.put((None, None, None), block=False)
            except queue.Full:
                pass

    def run_tasks(self, func_prepare_cpu, func_execute_gpu, tasks, **kwargs):
        self._is_canceled = False
        total = len(tasks)
        self.payload_queue = queue.Queue(maxsize=1)
        start = time.time()

        # Background Producer (CPU raw decoding)
        def cpu_producer():
            for idx, task in enumerate(tasks, 1):
                if self._is_canceled:
                    break
                try:
                    pipeline_payload = func_prepare_cpu(task[0], **kwargs)
                    while not self._is_canceled:
                        try:
                            self.payload_queue.put(
                                (idx, task, pipeline_payload), timeout=0.1
                            )
                            break
                        except queue.Full:
                            continue
                except Exception:
                    self.payload_queue.put((idx, task, None))
            self.payload_queue.put((None, None, None))

        producer_thread = threading.Thread(target=cpu_producer, daemon=True)
        producer_thread.start()

        # Consumer (GPU loop running on the QThread)
        while True:
            if self._is_canceled:
                break

            idx, task, pipeline_payload = self.payload_queue.get()
            if idx is None:
                break
            if pipeline_payload is None:
                self.payload_queue.task_done()
                continue

            status_msg = func_execute_gpu(
                task, pipeline_payload, current_idx=idx, total_count=total, **kwargs
            )
            self.progress.emit(status_msg, idx)
            self.payload_queue.task_done()

        producer_thread.join(timeout=1.0)

        print(f"total {time.time() - start:.2f}s")

        self.finished.emit()


class AutoShortcutsDialog(QDialog):
    """
    A dialog that automatically builds a list of shortcuts from both QActions and
    QShortcuts.
    """

    def __init__(self, shortcuts_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(400, 350)

        layout = QVBoxLayout(self)

        text_area = QTextEdit(self)
        text_area.setReadOnly(True)

        html_content = """
        <h3>Active Keyboard Shortcuts</h3>
        <table border="0" cellpadding="6" cellspacing="0" width="100%">
        """

        for description, shortcut_str in sorted(shortcuts_data.items()):
            html_content += (
                f"<tr><td>{description}</td><td><b>{shortcut_str}</b></td></tr>"
            )

        html_content += "</table>"
        text_area.setHtml(html_content)
        layout.addWidget(text_area)

        close_btn = AnimatedButton("Close", parent=self)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
