# w-CAT — concatenative audio synthesis

Requires Julia 1.10 or newer. Supports WAV audio, converted to mono for analysis and rendering.

1. From this folder, install dependencies once: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
2. Put your sounds (WAV files) under `audio/` (subfolders are fine).
3. Build the corpus index: `julia --project=. run.jl index`
4. Synthesize using any target WAV: `julia --project=. run.jl synth --target target.wav --output result.wav`

Edit `config.yaml` to choose features, chunking, and synthesis settings. After changing analysis features/chunking/layers, rebuild the index. The `synth` command matches target features against corpus chunks and combines selected audio grains. Audio files may have different sample rates; selected grains are resampled to the target rate.

Optional: open `automation_editor.html` (YAML import uses an online js-yaml library), load `config.yaml`, edit selection/feature curves, export `automation.json`, and rerun synthesis. Without curves, `automation.json` uses defaults from `config.yaml`. Relative paths are resolved from the config's directory. `--audio DIR` overrides the corpus folder for `index`; `--config FILE`, `--corpus FILE`, and `--automation FILE` are also available.

Analysis has four optional descriptors layers (`full`, `harmonic`, `percussive`, `residual`); the simpler full-signal mode is the default. Pitch analysis (`yin_pitch`) and HPSS separation are available but slower.
