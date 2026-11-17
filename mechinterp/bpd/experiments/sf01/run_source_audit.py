"""Standalone stdlib harness for the SF01 math tests, avoiding BPD imports.

Pytest's package-setup phase imports mechinterp.bpd.__init__ (which imports
PyTorch). Running this script validates the same pure mathematical tests in
a lightweight host with only pytest installed for its raises context manager.
"""

from pathlib import Path
import runpy
import unittest


if __name__ == "__main__":
    source = Path(__file__).with_name("test_information.py")
    namespace = runpy.run_path(str(source))
    test_functions = sorted(
        ((name, fn) for name, fn in namespace.items()
         if name.startswith("test_") and callable(fn)),
        key=lambda item: item[0],
    )
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn, description=name)
                               for name, fn in test_functions)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() and len(test_functions) == 10 else 1)
