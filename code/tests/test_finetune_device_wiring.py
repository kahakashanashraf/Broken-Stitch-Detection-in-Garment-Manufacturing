"""Regression tests: detected CUDA/CPU must reach both CV and refit.

Read module AST rather than invoking 5-fold real-data training in lightweight CI.
"""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "grouped_finetune_cv.py"


def _run_ast():
    tree = ast.parse(SOURCE.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")


def test_auto_detected_device_propagates_to_cv_and_final_refit():
    run = _run_ast()
    calls = [
        c for c in ast.walk(run)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "fit_once"
    ]
    assert len(calls) == 2, "Regression guard: CV and final refit must both call fit_once"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "device" in kwargs, "GPU detection is ineffective unless fit_once receives device"
        assert isinstance(kwargs["device"], ast.Name)
        assert kwargs["device"].id == "device"


def test_device_detection_and_training_tensor_transfer():
    src = SOURCE.read_text()
    run = _run_ast()
    names = {n.id for n in ast.walk(run) if isinstance(n, ast.Name)}
    assert "device" in names
    assert "torch.cuda.is_available()" in src
    assert "FineTuneBinary(model_name,pretrained=True).to(device)" in src
    assert "x=x.to(device);y=y.to(device);w=w.to(device)" in src
    assert "model(x.to(device))" in src
    assert "'compute_device_used':device" in src


def test_research_protocol_is_locked_after_device_fix():
    src = SOURCE.read_text()
    assert "N_SPLITS = 5" in src
    assert "IMAGE_SIZE = 128" in src
    assert "EPOCHS = 2" in src
    assert "MAX_PER_PARENT_CLASS = 40" in src
    assert "EXPECTED_MANIFEST_SHA = 'a873092d19c7b47987c7311a561d35cac3ef254d5aa0110e9a62fd1434a0e2c5'" in src
    assert "StratifiedGroupKFold(n_splits=N_SPLITS" in src
    assert "'train':Path(args.train_zip),'val':Path(args.val_zip)" in src
