#!/usr/bin/env python3
"""
Quantize AdaFace IR101 ONNX model to INT8 (dynamic quantization).

Dynamic quantization pre-quantizes all Conv and MatMul weight tensors to INT8.
At inference time activations are dynamically quantized, letting CPUs use SIMD
INT8 multiply-accumulate instructions (AVX2 VNNI) which run ~2x faster than FP32.

Model: 249 MB FP32 → ~62 MB INT8
Expected speedup: 1.5x–2x on recognition forward pass (CPU only).

Usage:
    cd /path/to/insight-face
    python scripts/quantize_adaface.py

After it completes, add to .env and restart the Celery worker + Gunicorn:
    ADAFACE_MODEL_FILENAME=adaface_ir101_webface12m_int8.onnx
"""

import os
import sys
import time

import numpy as np

FP32_MODEL = 'adaface_ir101_webface12m.onnx'
INT8_MODEL = 'adaface_ir101_webface12m_int8.onnx'
PACK_DIR = os.path.join(os.path.expanduser('~/.insightface'), 'models', 'adaface')
RUNS = 20
WARMUP = 5


def quantize(fp32_path: str, int8_path: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    fp32_mb = os.path.getsize(fp32_path) / 1024 / 1024
    print(f'Source : {fp32_path} ({fp32_mb:.0f} MB)')
    print(f'Output : {int8_path}')
    print('Running dynamic INT8 quantization (Conv + MatMul weights) …')

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'MatMul'],
        per_channel=True,
    )
    elapsed = time.perf_counter() - t0

    int8_mb = os.path.getsize(int8_path) / 1024 / 1024
    reduction = 100 * (1 - int8_mb / fp32_mb)
    print(f'Done in {elapsed:.1f}s  |  {fp32_mb:.0f} MB → {int8_mb:.0f} MB  ({reduction:.0f}% smaller)')


def _make_session(path: str):
    import onnxruntime
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0
    return onnxruntime.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])


def benchmark(fp32_path: str, int8_path: str) -> None:
    print(f'\nBenchmarking ({WARMUP} warm-up + {RUNS} timed runs each, batch=1) …')

    fp32_sess = _make_session(fp32_path)
    int8_sess = _make_session(int8_path)

    # Use the actual input name from the model
    input_name = fp32_sess.get_inputs()[0].name
    input_shape = fp32_sess.get_inputs()[0].shape
    # Replace dynamic dims (None / symbolic) with concrete values
    concrete = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
    dummy = np.random.rand(*concrete).astype(np.float32)

    results = {}
    for label, sess in [('FP32', fp32_sess), ('INT8', int8_sess)]:
        for _ in range(WARMUP):
            sess.run(None, {input_name: dummy})

        t0 = time.perf_counter()
        for _ in range(RUNS):
            sess.run(None, {input_name: dummy})
        ms = (time.perf_counter() - t0) / RUNS * 1000

        results[label] = ms
        print(f'  {label}: {ms:.1f} ms / inference')

    speedup = results['FP32'] / results['INT8']
    print(f'\n  Speedup: {speedup:.2f}x  ({"improvement" if speedup > 1 else "no improvement — likely no AVX2 VNNI support"})')


def main() -> None:
    fp32_path = os.path.join(PACK_DIR, FP32_MODEL)
    int8_path = os.path.join(PACK_DIR, INT8_MODEL)

    if not os.path.exists(fp32_path):
        print(f'ERROR: FP32 model not found at {fp32_path}')
        print('Start the Django server or Celery worker once to trigger the download, then re-run.')
        sys.exit(1)

    quantize(fp32_path, int8_path)
    benchmark(fp32_path, int8_path)

    print('\n── Next steps ──────────────────────────────────────────────────────────')
    print(f'1. Add to .env:')
    print(f'     ADAFACE_MODEL_FILENAME={INT8_MODEL}')
    print('2. Restart Celery worker:')
    print('     supervisorctl restart celery-worker  # or: systemctl restart celery')
    print('3. Restart Gunicorn:')
    print('     supervisorctl restart gunicorn        # or: systemctl restart gunicorn')
    print('────────────────────────────────────────────────────────────────────────')


if __name__ == '__main__':
    main()
