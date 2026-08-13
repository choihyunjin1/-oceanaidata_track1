from __future__ import annotations

import importlib
import sys
import unittest

import numpy as np

from p1_qc.ensemble import ConvexProbabilityBlender, convex_blend, simplex_lattice
from p1_qc.models_deep import (
    LossConfig,
    PatchTransformerConfig,
    TCNConfig,
    build_bce_dice_aux_loss,
    build_patch_transformer,
    build_tcn,
    torch_available,
)
from p1_qc.models_tabular import (
    TabularModelConfig,
    catboost_parameters,
    lightgbm_parameters,
    make_tabular_classifier,
    xgboost_parameters,
)


class TabularModelTests(unittest.TestCase):
    def test_import_does_not_import_optional_boosters(self) -> None:
        import p1_qc.models_tabular as module

        importlib.reload(module)
        self.assertNotIn("lightgbm", sys.modules)
        self.assertNotIn("xgboost", sys.modules)
        self.assertNotIn("catboost", sys.modules)

    def test_deterministic_cpu_defaults(self) -> None:
        lightgbm = lightgbm_parameters(TabularModelConfig(backend="lightgbm", seed=7, n_jobs=8))
        xgboost = xgboost_parameters(TabularModelConfig(backend="xgboost", seed=7, n_jobs=8))
        catboost = catboost_parameters(TabularModelConfig(backend="catboost", seed=7, n_jobs=8))
        self.assertTrue(lightgbm["deterministic"])
        self.assertTrue(lightgbm["force_row_wise"])
        self.assertEqual(lightgbm["n_jobs"], 8)
        self.assertEqual(xgboost["device"], "cpu")
        self.assertEqual(xgboost["tree_method"], "hist")
        self.assertEqual(catboost["task_type"], "CPU")
        self.assertEqual(catboost["bootstrap_type"], "No")
        self.assertFalse(catboost["allow_writing_files"])

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name) is not None
            for name in ("lightgbm", "xgboost", "catboost")
        ),
        "optional tree boosters are not installed",
    )
    def test_optional_backends_fit_small_binary_problem(self) -> None:
        rng = np.random.default_rng(5)
        features = rng.normal(size=(80, 5))
        target = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(np.int8)
        backend_parameters = {
            "lightgbm": {"n_estimators": 3, "num_leaves": 7},
            "xgboost": {"n_estimators": 3, "max_depth": 2},
            "catboost": {"iterations": 3, "depth": 2},
        }
        for backend, parameters in backend_parameters.items():
            with self.subTest(backend=backend):
                model = make_tabular_classifier(
                    backend,
                    seed=11,
                    n_jobs=1,
                    parameters=parameters,
                ).fit(features, target)
                probability = model.predict_proba(features)
                self.assertEqual(probability.shape, (80, 2))
                self.assertTrue(np.isfinite(probability).all())
                np.testing.assert_allclose(probability.sum(axis=1), 1.0)


class EnsembleTests(unittest.TestCase):
    def test_convex_blend_is_nonnegative_and_bounded(self) -> None:
        predictions = {"a": [0.0, 1.0], "b": [1.0, 0.0]}
        blended = convex_blend(predictions, [2.0, 1.0])
        np.testing.assert_allclose(blended, [1 / 3, 2 / 3])
        self.assertTrue(((0 <= blended) & (blended <= 1)).all())

    def test_fit_selects_useful_model_and_simplex_weights(self) -> None:
        target = np.array([0, 0, 1, 1])
        predictions = {
            "good": np.array([0.05, 0.15, 0.85, 0.95]),
            "bad": np.array([0.9, 0.8, 0.2, 0.1]),
        }
        blender = ConvexProbabilityBlender(
            weight_resolution=10,
            thresholds=[0.5],
        ).fit(predictions, target)
        self.assertAlmostEqual(float(blender.result_.weights.sum()), 1.0)
        self.assertTrue((blender.result_.weights >= 0).all())
        np.testing.assert_array_equal(blender.predict(predictions), target)
        self.assertEqual(simplex_lattice(3, 4).shape[0], 15)


@unittest.skipUnless(torch_available(), "PyTorch overlay is optional")
class DeepModelSmokeTests(unittest.TestCase):
    def test_tcn_forward_backward(self) -> None:
        import torch

        torch.manual_seed(3)
        model = build_tcn(TCNConfig(input_dim=4, channels=(8, 8), dropout=0.0, aux_classes=5))
        features = torch.randn(2, 32, 4)
        target = torch.randint(0, 2, (2, 32), dtype=torch.float32)
        auxiliary = torch.randint(0, 2, (2, 32, 5), dtype=torch.float32)
        output = model(features)
        self.assertEqual(tuple(output.logits.shape), (2, 32))
        self.assertEqual(tuple(output.aux_logits.shape), (2, 32, 5))
        loss = build_bce_dice_aux_loss(LossConfig())(
            output,
            target,
            auxiliary_target=auxiliary,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_patch_transformer_forward(self) -> None:
        import torch

        model = build_patch_transformer(
            PatchTransformerConfig(
                input_dim=3,
                d_model=8,
                nhead=2,
                num_layers=1,
                dim_feedforward=16,
                dropout=0.0,
                patch_size=4,
                patch_stride=2,
            )
        )
        output = model(torch.randn(2, 24, 3))
        self.assertEqual(tuple(output.logits.shape), (2, 24))
        self.assertEqual(tuple(output.aux_logits.shape), (2, 24, 5))


if __name__ == "__main__":
    unittest.main()
