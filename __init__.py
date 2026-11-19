import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from .nodes import YingMusicSingerLoader, YingMusicSingerSing  # noqa: E402

NODE_CLASS_MAPPINGS = {"YingMusicSingerLoader": YingMusicSingerLoader, "YingMusicSingerSing": YingMusicSingerSing}
NODE_DISPLAY_NAME_MAPPINGS = {"YingMusicSingerLoader": "YingMusic-Singer Loader",
                              "YingMusicSingerSing": "YingMusic-Singer Sing (melody-guided)"}
