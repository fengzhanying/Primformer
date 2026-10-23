import contextlib
import glob
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("primformer", ROOT / "Primformer.py")
primformer = importlib.util.module_from_spec(spec)
previous_cwd = os.getcwd()
try:
    spec.loader.exec_module(primformer)
finally:
    os.chdir(previous_cwd)


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = primformer.parse_args([
            "--mode", "validation", "--checkpoint-dir", str(self.root),
            "--val-start", "1000", "--val-stop", "4000",
            "--val-output", str(self.root / "validation.tsv"),
            "--eval-output", str(self.root / "evaluation.tsv"),
        ])
        fake_tf = SimpleNamespace(io=SimpleNamespace(gfile=SimpleNamespace(
            exists=os.path.exists, glob=glob.glob,
            copy=lambda source, destination, overwrite: shutil.copyfile(source, destination),
        )))
        self.enterContext(patch.object(primformer, "tf", fake_tf))
        self.enterContext(patch.object(primformer, "load_runtime"))
        self.dataset = Mock()
        self.dataset.batch.return_value.prefetch.return_value = self.dataset
        self.loader = self.enterContext(patch.object(primformer, "get_dataset", return_value=self.dataset))
        self.score = self.enterContext(patch.object(primformer, "score_checkpoint"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def checkpoint(self, step):
        prefix = self.root / f"MeHSA-{step}"
        for suffix in (".index", ".data-00000-of-00002", ".data-00001-of-00002"):
            Path(str(prefix) + suffix).write_text(f"weights at step {step}")

    def validate(self, scores):
        for step in scores:
            self.checkpoint(step)
        self.score.side_effect = lambda prefix, dataset: (scores[int(prefix.rsplit("-", 1)[1])],) * 6
        primformer.run_validation(self.args)
        return json.loads(Path(primformer.selection_path(self.args)).read_text())

    def test_selects_maximum_and_evaluates_only_saved_model_on_test(self):
        best = self.validate({1000: 0.4, 2000: 0.8, 3000: 0.6})
        self.assertEqual(best["step"], 2000)
        self.loader.assert_called_once_with("Human", "valid")
        self.assertEqual(self.score.call_count, 3)
        self.assertEqual(len((self.root / "validation.tsv").read_text().splitlines()), 4)
        # The final checkpoint remains usable after the source checkpoints disappear.
        for source in self.root.glob("MeHSA-*"):
            source.unlink()
        saved_prefix = str(self.root / "best_model" / "MeHSA-2000")
        self.assertEqual(len(primformer.checkpoint_files(saved_prefix)), 3)
        self.loader.reset_mock()
        self.score.reset_mock(side_effect=True)
        self.score.return_value = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
        primformer.run_eval(self.args)
        self.loader.assert_called_once_with("Human", "test")
        self.score.assert_called_once_with(saved_prefix, self.dataset)
        rows = (self.root / "evaluation.tsv").read_text().splitlines()
        self.assertEqual(rows[0], "Steps\tTest_All\tTest_CAGE\tTest_RNA\tTest_ATAC\tTest_ChIPH\tTest_ChIPT")
        self.assertEqual(rows[1], "2000\t0.7\t0.6\t0.5\t0.4\t0.3\t0.2")

    def test_ties_keep_earliest_and_nan_cannot_win(self):
        best = self.validate({1000: float("nan"), 2000: -0.2, 3000: -0.2, 4000: -0.4})
        self.assertEqual(best["step"], 2000)

    def test_missing_checkpoints_do_not_create_selection(self):
        with self.assertRaisesRegex(ValueError, "No checkpoint"):
            primformer.run_validation(self.args)
        self.score.assert_not_called()
        self.assertFalse(Path(primformer.selection_path(self.args)).exists())

    def test_all_nonfinite_scores_fail_selection(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.validate({1000: float("nan"), 2000: float("inf")})
        self.assertFalse(Path(primformer.selection_path(self.args)).exists())

    def test_evaluation_requires_selection_before_loading_data(self):
        with self.assertRaisesRegex(FileNotFoundError, "validation first"):
            primformer.run_eval(self.args)
        self.loader.assert_not_called()

    def test_evaluation_rejects_different_dataset(self):
        self.validate({1000: 0.5})
        self.loader.reset_mock()
        self.args.human_dataset = "DifferentHuman"
        with self.assertRaisesRegex(ValueError, "must match"):
            primformer.run_eval(self.args)
        self.loader.assert_not_called()

    def test_evaluation_rejects_nonfinite_overall_score(self):
        self.validate({1000: 0.5})
        self.score.side_effect = None
        self.score.return_value = (float("nan"),) * 6
        with self.assertRaisesRegex(ValueError, "Test_All"):
            primformer.run_eval(self.args)
        self.assertFalse((self.root / "evaluation.tsv").exists())

    def test_invalid_validation_ranges_fail_at_cli(self):
        for options in (["--val-interval", "0"], ["--val-start", "-1"], ["--val-stop", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    primformer.parse_args(["--mode", "validation", *options])

    def test_restoration_is_checked_after_full_dataset_scoring(self):
        self.checkpoint(1000)
        model = Mock()
        checkpoint = Mock()
        status = checkpoint.restore.return_value.expect_partial.return_value
        status.assert_existing_objects_matched = Mock()
        evaluator = Mock(return_value=(0.5,) * 6)
        with patch.object(primformer, "score_checkpoint", ORIGINAL_SCORE), \
             patch.object(primformer, "build_model", return_value=model), \
             patch.object(primformer, "evaluate_model1", evaluator), \
             patch.object(primformer.tf, "train", SimpleNamespace(Checkpoint=Mock(return_value=checkpoint)), create=True):
            primformer.score_checkpoint(str(self.root / "MeHSA-1000"), self.dataset)
        evaluator.assert_called_once_with(model, dataset=self.dataset)
        status.assert_existing_objects_matched.assert_called_once_with()


ORIGINAL_SCORE = primformer.score_checkpoint

if __name__ == "__main__":
    unittest.main()
