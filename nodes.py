"""ComfyUI nodes for YingMusic-Singer-Plus (melody-guided singing voice synthesis).

Inputs: a timbre reference (a singing clip + its lyrics), a melody clip (singing, or any clean monophonic melody) and the
new lyrics. Output: the lyrics sung on the melody, in the reference voice, 44.1 kHz, as long as the melody clip.
All audio inputs must be pure voice/melody (no accompaniment).
"""
import gc
import os
import sys
import tempfile

import numpy as np
import torch

try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "YingMusicSinger")
    TEMP_DIR = folder_paths.get_temp_directory()
except ImportError:  # unit tests outside ComfyUI
    folder_paths = None
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    TEMP_DIR = tempfile.gettempdir()

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)      # the vendored package is imported as `ying_singer` (hydra also resolves classes by that name)
CKPT_NAME = "model.safetensors"
REPO = "ASLP-lab/YingMusic-Singer-Plus"
_CACHE = {}


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _ensure_weights():
    path = os.path.join(MODELS_DIR, CKPT_NAME)
    if os.path.isfile(path):
        return path
    os.makedirs(MODELS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    print(f"[YingMusicSinger] {CKPT_NAME} not staged; downloading from {REPO} (slow: stage it with concept_mapping)")
    return hf_hub_download(REPO, CKPT_NAME, local_dir=MODELS_DIR)


def _load_model():
    from safetensors.torch import load_file
    from ying_singer.infer.YingMusicSinger import YingMusicSinger
    cfg_dir = os.path.join(HERE, "ying_singer", "config")
    model = YingMusicSinger(
        model_cfg_path=os.path.join(cfg_dir, "YingMusic_Singer.yaml"),
        ckpt_path=None,
        vae_config_path=os.path.join(cfg_dir, "stable_audio_2_0_vae_20hz_official.json"),
        vae_ckpt_path=None, midi_teacher_ckpt_path=None, is_distilled=False, use_ema=True)
    state = load_file(_ensure_weights())
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[YingMusicSinger] weights loaded (missing {len(missing)}, unexpected {len(unexpected)})")
    model.eval()
    return model.to(_device())


def _write_wav(audio, path):
    """ComfyUI AUDIO -> 44.1 kHz stereo float wav that the model's torchaudio.load can read."""
    import soundfile as sf
    w = audio["waveform"][0].detach().cpu().float().numpy()          # [C, T]
    sr = int(audio["sample_rate"])
    if w.shape[0] == 1:
        w = np.repeat(w, 2, axis=0)
    elif w.shape[0] > 2:
        w = w[:2]
    if sr != 44100:
        import librosa
        w = np.stack([librosa.resample(ch, orig_sr=sr, target_sr=44100) for ch in w])
        sr = 44100
    sf.write(path, w.T, sr, subtype="FLOAT")
    return w.shape[1] / sr


class YingMusicSingerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"keep_loaded": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("YING_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "YingMusicSinger"

    def load(self, keep_loaded):
        if keep_loaded and "model" in _CACHE:
            return (_CACHE["model"],)
        m = _load_model()
        if keep_loaded:
            _CACHE["model"] = m
        return (m,)


class YingMusicSingerSing:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("YING_MODEL",),
            "ref_audio": ("AUDIO",),
            "ref_text": ("STRING", {"multiline": True, "default": "la la la"}),
            "melody_audio": ("AUDIO",),
            "target_text": ("STRING", {"multiline": True, "default": "neon lights are calling me home|hold on, hold on, feel the beat"}),
            "lrc_align_mode": (["sentence_level", "put_to_front"], {"default": "sentence_level"}),
            "nfe_step": ("INT", {"default": 32, "min": 8, "max": 64, "step": 1}),
            "cfg_strength": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 6.0, "step": 0.1}),
            "t_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 666, "min": 0, "max": 2 ** 31 - 1}),
            "sil_len_to_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "sing"
    CATEGORY = "YingMusicSinger"

    def sing(self, model, ref_audio, ref_text, melody_audio, target_text, lrc_align_mode, nfe_step, cfg_strength,
             t_shift, seed, sil_len_to_end):
        # Graydient slot values stop at the first space, so the reference words may be given with underscores for spaces,
        # or after an @@ separator in the lyrics ("target lyrics @@ reference lyrics").
        if "@@" in target_text:
            target_text, tail = target_text.split("@@", 1)
            if not ref_text.strip() or ref_text.strip() in ("-", "none"):
                ref_text = tail
        ref_text = ref_text.replace("_", " ")
        d = tempfile.mkdtemp(dir=TEMP_DIR)
        rp, mp = os.path.join(d, "ref.wav"), os.path.join(d, "melody.wav")
        rdur = _write_wav(ref_audio, rp)
        mdur = _write_wav(melody_audio, mp)
        print(f"[YingMusicSinger] ref {rdur:.1f}s melody {mdur:.1f}s; ref text {ref_text!r}; target {target_text!r}")
        with torch.no_grad():
            wave, sr = model(ref_audio_path=rp, melody_audio_path=mp, ref_text=ref_text.strip(),
                             target_text=target_text.strip(), lrc_align_mode=lrc_align_mode,
                             sil_len_to_end=sil_len_to_end, t_shift=t_shift, nfe_step=nfe_step,
                             cfg_strength=cfg_strength, seed=seed)
        wave = wave.detach().cpu().float()                              # [C, T]
        info = f"sung {wave.shape[-1] / sr:.2f}s at {sr} Hz for a {mdur:.2f}s melody"
        print(f"[YingMusicSinger] {info}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ({"waveform": wave.unsqueeze(0).contiguous(), "sample_rate": int(sr)}, info)
