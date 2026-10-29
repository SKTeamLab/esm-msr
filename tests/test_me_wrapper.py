"""inference_scripts/baselines/mutate_everything_inference.py needs the MutateEverything checkout to import (it chdirs into it and
imports its modules), so these tests read its source. Both defects they pin down were found on the first real run (2026-10-29):
`predict` had lost its @torch.no_grad() (the forward pass kept its autograd graph: 18 GB at L = 66 on a 20 GB card), and `_is_oom`
named torch.cuda.OutOfMemoryError, which ME's torch 1.12 does not have, so every exception of the forward pass became an AttributeError."""
import ast
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'inference_scripts' / 'baselines' / 'mutate_everything_inference.py'


def _func(name):
    tree = ast.parse(SRC.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _is_oom_with(cuda_attrs):
    fn = _func('_is_oom')
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(**cuda_attrs))}
    exec(compile(module, str(SRC), 'exec'), namespace)
    return namespace['_is_oom']


def test_predict_runs_without_autograd():
    assert 'torch.no_grad()' in [ast.unparse(d) for d in _func('predict').decorator_list]


def test_is_oom_on_a_torch_without_the_exception_class():
    is_oom = _is_oom_with({})
    assert is_oom(RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB'))
    assert not is_oom(RuntimeError('shape mismatch'))
    assert not is_oom(ValueError('out of memory'))


def test_is_oom_with_the_exception_class():
    class OutOfMemoryError(RuntimeError):
        pass

    is_oom = _is_oom_with({'OutOfMemoryError': OutOfMemoryError})
    assert is_oom(OutOfMemoryError('anything'))
    assert not is_oom(RuntimeError('shape mismatch'))
