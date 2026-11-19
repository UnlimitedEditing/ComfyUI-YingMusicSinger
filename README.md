# ComfyUI-YingMusicSinger

ComfyUI nodes for [YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus) (ASLP@NPU / GiantNetwork):
**melody-guided singing voice synthesis**. Give it a timbre reference (a singing clip and its lyrics), a melody clip and new
lyrics, and it sings the lyrics on that melody in the reference voice (44.1 kHz). No MIDI or phoneme alignment needed.

- **YingMusic-Singer Loader** - loads `models/YingMusicSinger/model.safetensors` (2.9 GB, from
  `ASLP-lab/YingMusic-Singer-Plus`; it contains the sampler, the SOME melody extractor and the Stable Audio 2 VAE).
- **YingMusic-Singer Sing** - `ref_audio` + `ref_text`, `melody_audio`, `target_text` (phrases separated by `|`) -> audio.
  The output is as long as the melody clip, so it can be patched back onto a timeline at the same start time.

All audio inputs must be pure voice / a clean melody (no accompaniment). English lyrics only in this port (the Chinese
pronunciation resources, about 90 MB, are not vendored).

## What is changed from upstream
A minimal inference-only vendored copy (`ying_singer/`): relative data paths; the `espeak-ng` system dependency is replaced by the
bundled library from `espeakng-loader`; `torchaudio.load` is replaced by `soundfile` (torchcodec is unreliable on some runners);
`dac`, `laion_clap` and `natten` are stubbed (the VAE only needs three small layers, reproduced in `ying_singer/_shims.py`);
the Chinese g2p is imported lazily. Everything else is unmodified.

## Licences
Code and model weights: CC BY 4.0 (ASLP@NPU). The VAE weights and `ying_singer/utils/stable_audio_tools` are derived from
Stable Audio Open and are under the Stability AI Community License (`LICENSE-STABILITY`; "Powered by Stability AI").
The g2p code is from Amphion (MIT); the DiT backbone follows F5-TTS (MIT).
