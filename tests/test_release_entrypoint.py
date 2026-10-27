import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_run', ROOT / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class ReleaseEntryPointTest(unittest.TestCase):
    def test_all_registered_sources_exist(self):
        for relative in list(runner.STAGES.values()) + list(runner.PLOTS.values()) + [runner.TRAIN]:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_dry_run_never_starts_a_process(self):
        with patch.object(runner.subprocess, 'call') as call, redirect_stdout(io.StringIO()) as output:
            code = runner.main(['--dry-run', 'experiment', 'external-controls'])
        self.assertEqual(code, 0)
        call.assert_not_called()
        self.assertIn('--external-strong-controls', output.getvalue())

    def test_unknown_stage_fails_before_execution(self):
        with patch.object(runner.subprocess, 'call') as call, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(['stage', 'not-a-stage'])
        call.assert_not_called()

if __name__ == '__main__':
    unittest.main()
