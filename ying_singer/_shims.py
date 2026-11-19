"""Stand-ins for imports the inference path never really uses, so no heavy/compiled packages are needed on the runner.

* dac      - descript-audio-codec: only three tiny layers (WNConv1d, WNConvTranspose1d, Snake1d) are used by the VAE; they are
             reproduced here exactly (same parameter names, so the checkpoint loads).
* laion_clap, natten - only used by training-time conditioners / unused attention variants: replaced by inert dummies.
"""
import sys
import types

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def _install_dac():
    try:
        import dac.nn.layers  # noqa: F401
        return
    except Exception:
        pass

    def WNConv1d(*args, **kwargs):
        return weight_norm(nn.Conv1d(*args, **kwargs))

    def WNConvTranspose1d(*args, **kwargs):
        return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

    def snake(x, alpha):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)

    class Snake1d(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))

        def forward(self, x):
            return snake(x, self.alpha)

    layers = types.ModuleType("dac.nn.layers")
    layers.WNConv1d, layers.WNConvTranspose1d, layers.Snake1d, layers.snake = WNConv1d, WNConvTranspose1d, Snake1d, snake
    nn_mod = types.ModuleType("dac.nn")
    nn_mod.layers = layers
    nn_mod.__path__ = []
    root = types.ModuleType("dac")
    root.nn = nn_mod
    root.__path__ = []
    quant = _dummy_module("dac.nn.quantize")        # residual VQ is only used by discrete bottlenecks, not this VAE
    nn_mod.quantize = quant
    sys.modules.update({"dac": root, "dac.nn": nn_mod, "dac.nn.layers": layers, "dac.nn.quantize": quant})


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Dummy()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()


def _dummy_module(name):
    m = types.ModuleType(name)
    def _ga(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Dummy()
    m.__getattr__ = _ga                           # from x import anything
    m.__path__ = []
    return m


def _install_dummies():
    for name in ("laion_clap", "laion_clap.clap_module", "laion_clap.clap_module.factory", "natten"):
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = _dummy_module(name)


_install_dac()
_install_dummies()
