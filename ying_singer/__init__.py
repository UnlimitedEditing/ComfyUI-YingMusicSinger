import os

from . import _shims  # noqa: F401  (must run before the vendored modules import dac / laion_clap / natten)


def _setup_espeak():
    """Use the espeak-ng bundled in the `espeakng-loader` wheel, so no system package is needed on the runner."""
    try:
        import espeakng_loader
        lib, data = espeakng_loader.get_library_path(), espeakng_loader.get_data_path()
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", lib)
        os.environ.setdefault("ESPEAK_DATA_PATH", data)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(lib)
        if hasattr(EspeakWrapper, "set_data_path"):
            EspeakWrapper.set_data_path(data)
    except Exception as exc:  # noqa: BLE001
        print(f"[YingMusicSinger] bundled espeak not set up ({exc}); falling back to a system espeak-ng if present")


_setup_espeak()
