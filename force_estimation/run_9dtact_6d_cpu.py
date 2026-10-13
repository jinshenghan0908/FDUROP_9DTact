"""9DTact official DenseNet-169 inference on Windows/CPU without ROS.

Save this file as: <project>/force_estimation/run_9dtact_6d_cpu.py
Run from Anaconda Prompt:
    conda activate 9dtact
    cd /d D:\\9DTact\\force_estimation
    python run_9dtact_6d_cpu.py

The displayed forces are cross-sensor *uncalibrated estimates*, not validated
measurements. The network was trained using the original author's sensor.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

FORCE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FORCE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(FORCE_DIR))

# The official force_estimation/__init__.py imports a ROS-dependent visualizer.
# Import estimator.py directly to avoid requiring rospy on Windows.
from estimator import Estimator  # noqa: E402
from shape_reconstruction import Sensor  # noqa: E402

# Ranges are from original 9DTact/data_collection/wrench_normalization.py.
# Fx, Fy, Fz in N; Mx, My, Mz in N*m.
WRENCH_MIN = np.array([-3.0, -3.0, -12.0, -0.2, -0.2, -0.05], dtype=float)
WRENCH_MAX = np.array([3.0, 3.0, 0.0, 0.2, 0.2, 0.05], dtype=float)
WRENCH_SPAN = WRENCH_MAX - WRENCH_MIN
WEIGHT_PATH = ROOT_DIR / 'saved_models' / 'Densenet-169_09-22_15-17-06' / 'epoch_153.pt'
CSV_PATH = FORCE_DIR / 'force_predictions_unverified.csv'


def draw_panel(normalized: np.ndarray, physical: np.ndarray, device: str) -> np.ndarray:
    panel = np.full((345, 690, 3), 245, dtype=np.uint8)
    cv2.putText(panel, '9DTact | Official model | UNVERIFIED predictions',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (15, 15, 190), 2)
    cv2.putText(panel, f'Device: {device}   Model: DenseNet-169, epoch 153',
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (40, 40, 40), 1)
    labels = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')
    for idx, label in enumerate(labels):
        unit = 'N' if idx < 3 else 'N*m'
        line = f'{label:2s} = {physical[idx]:+10.4f} {unit:<3s}  | norm = {normalized[idx]:.4f}'
        cv2.putText(panel, line, (20, 91 + idx * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.59, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(panel, 'q: quit  |  No-contact: reference when prompted with y',
                (12, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1)
    return panel


def main() -> None:
    # Original configuration uses relative directory paths.
    os.chdir(FORCE_DIR)
    print('Project directory:', ROOT_DIR)
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            f'Official 6D checkpoint is missing:\n  {WEIGHT_PATH}\n'
            'Check your clone or download epoch_153.pt from the official 9DTact GitHub repository.'
        )
    if WEIGHT_PATH.stat().st_size < 1_000_000:
        raise RuntimeError(
            f'The checkpoint is unusually small ({WEIGHT_PATH.stat().st_size} bytes). '
            'Check for an incomplete download or Git LFS pointer file.'
        )

    shape_config_path = ROOT_DIR / 'shape_reconstruction' / 'shape_config.yaml'
    force_config_path = FORCE_DIR / 'force_config.yaml'
    with shape_config_path.open('r', encoding='utf-8') as f:
        shape_config = yaml.safe_load(f)
    with force_config_path.open('r', encoding='utf-8') as f:
        force_config = yaml.safe_load(f)

    # Force the checkpoint configuration to match the publicly released model.
    force_config['model_choice'] = 1
    force_config['weights'][1] = ['_09-22_15-17-06', 153]
    force_config['save_dir'] = '../saved_models'
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('PyTorch:', torch.__version__, '| Device:', device)
    print('Camera channel:', shape_config['camera_setting']['camera_channel'])
    print('Leave gel surface UNTOUCHED, wait for stable image, click camera and press y.')

    sensor = None
    try:
        # Reuse *your own* successful camera calibration and reference image.
        sensor = Sensor(shape_config)
        if not sensor.cap.isOpened():
            raise RuntimeError('Camera cannot be opened; check camera_channel in shape_config.yaml.')

        estimator = Estimator(force_config)  # loads original epoch_153.pt on CPU/GPU
        print('Ready. Press q in any OpenCV window to stop.')
        print('The physical-unit numbers below are UNVERIFIED cross-sensor estimates.')
        fields = ['timestamp', 'Fx_N', 'Fy_N', 'Fz_N', 'Mx_Nm', 'My_Nm', 'Mz_Nm',
                  'Fx_norm', 'Fy_norm', 'Fz_norm', 'Mx_norm', 'My_norm', 'Mz_norm']
        with CSV_PATH.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            last_print = 0.0
            while sensor.cap.isOpened():
                image = sensor.get_rectify_crop_image()
                if image is None or image.size == 0:
                    raise RuntimeError('Empty frame from camera. Check USB connection and ROI calibration.')
                expected_hw = tuple(force_config['img_size'])
                if image.shape[:2] != expected_hw:
                    raise RuntimeError(f'Cropping size {image.shape[:2]} does not match model {expected_hw}. '
                                       'Verify camera calibration and shape_config.yaml crop_size.')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                representation, vis_representation = sensor.raw_image_2_representation(gray)
                prediction_norm = np.asarray(estimator.predict_force(representation), dtype=float)
                prediction_real = WRENCH_MIN + prediction_norm * WRENCH_SPAN
                cv2.imshow('9DTact Camera', image)
                cv2.imshow('9DTact Representation', vis_representation)
                cv2.imshow('9DTact 6D prediction (unverified)',
                           draw_panel(prediction_norm, prediction_real, device))
                now = time.monotonic()
                if now - last_print >= 1.0:
                    row = [datetime.now().isoformat(timespec='milliseconds')]
                    row.extend(prediction_real.tolist())
                    row.extend(prediction_norm.tolist())
                    writer.writerow(row)
                    f.flush()
                    print(f'Fx={prediction_real[0]:+.3f} N | Fy={prediction_real[1]:+.3f} N | '
                          f'Fz={prediction_real[2]:+.3f} N | '
                          f'Mx={prediction_real[3]:+.4f} N*m | '
                          f'My={prediction_real[4]:+.4f} N*m | '
                          f'Mz={prediction_real[5]:+.4f} N*m')
                    last_print = now
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
        print('Saved prediction log to:', CSV_PATH)
    finally:
        if sensor is not None and sensor.cap is not None:
            sensor.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
