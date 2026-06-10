from __future__ import annotations

import numpy as np
import onnxruntime as ort


def create_onnx_session(onnx_path: str, providers: list[str] | None = None) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session.

    Args:
        onnx_path: Path to the ONNX model file.
        providers: List of execution providers (default: CPUExecutionProvider).

    Returns:
        An ORT InferenceSession ready for inference.
    """
    providers = providers or ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=options, providers=providers)


def run_onnx(session: ort.InferenceSession, batch_np: np.ndarray, input_name: str | None = None):
    """Run inference on an ONNX model.

    Args:
        session: ORT InferenceSession.
        batch_np: Input batch as a numpy array.
        input_name: Name of the input tensor (auto-detected if None).

    Returns:
        Model output (logits array).
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: batch_np.astype(np.float32)})[0]
