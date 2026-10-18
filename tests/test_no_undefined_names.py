"""pyflakes over src/, asserting no undefined names.

This exists because of a real failure: --assay_floor was threaded into load_benchmark_datasets' body without
adding it to that function's signature, so every benchmark validation set (ptmuld, s461, ssym) raised
NameError inside its own try block and the run came up with 25 validation loaders instead of 28 - silently,
because the exception was swallowed per dataset. The config parsed, the import succeeded, the unit suite
passed, and the run started. A name that does not exist is a defect no matter which branch reaches it.
"""
import pathlib
import subprocess
import sys
import unittest


class TestNoUndefinedNames(unittest.TestCase):
    def test_pyflakes_reports_no_undefined_names_in_src(self):
        try:
            import pyflakes  # noqa: F401
        except ImportError:
            self.skipTest("pyflakes not installed in this environment")
        src = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'esm_msr'
        files = sorted(str(p) for p in src.glob('*.py'))
        self.assertTrue(files, f"no sources found under {src}")
        out = subprocess.run([sys.executable, '-m', 'pyflakes', *files],
                             capture_output=True, text=True).stdout
        # The exact phrasing matters: pyflakes also prints "unable to detect undefined names" for a
        # wildcard import, which is a different and much weaker finding.
        bad = [l for l in out.splitlines() if "undefined name '" in l]
        self.assertEqual(bad, [], "pyflakes found undefined names:\n" + "\n".join(bad))


if __name__ == '__main__':
    unittest.main()
