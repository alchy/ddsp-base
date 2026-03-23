# Spuštění na Apple Silicon (M-series)

Apple M-series čipy (M1–M5) mají unified memory architekturu — CPU a GPU sdílí stejnou paměť
bez nutnosti kopírovat tensory. PyTorch podporuje tento backend přes **MPS** (Metal Performance Shaders).

---

## Proč je Apple Silicon výhodný pro DDSP trénink

| | Starý Intel/AMD notebook (CPU) | Apple M5 |
|---|---|---|
| Salamander small, 1 epocha | ~13 min | ~1 min (odhad) |
| Přenos dat CPU→GPU | kopírování | není (shared memory) |
| RAM pro model | oddělená CPU RAM | unified (jedna velká banka) |
| Spotřeba | vysoká | nízká |

Unified memory je klíčová výhoda: velké batche a velké modely (medium/large) se vejdou
bez OOM chyb, které trápí notebooky s dedikovanou GPU o malé VRAM.

---

## Instalace

```bash
# PyTorch s MPS (macOS 12.3+, Apple Silicon)
pip install torch torchvision torchaudio

# Ověření
python -c "import torch; print(torch.backends.mps.is_available())"
# -> True
```

Ostatní závislosti (`librosa`, `soundfile`, `gradio`) fungují beze změny.

---

## Spuštění tréninku

Přidej `--device mps` ke všem příkazům:

```bash
# Trénink
python ddsp.py learn --instrument /path/to/salamander --model small --epochs 300 --device mps

# EnvelopeNet
python ddsp.py learn-envelope --instrument /path/to/salamander --device mps

# Generování
python ddsp.py generate --instrument /path/to/salamander --device mps
```

GUI spouštěj stejně jako na Windows:

```bash
python gui.py
```

GUI automaticky předá `--device` pokud bude přidán výběr do záložky Nastroj & Stav
(zatím není implementováno — viz TODO níže).

---

## Autodetekce zařízení v ddsp.py

Pokud chceš, aby ddsp.py vybíralo zařízení automaticky (CUDA → MPS → CPU):

```python
def get_device(requested: str = 'auto') -> torch.device:
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
```

---

## Známá omezení MPS

- `torch.stft` / `torch.istft` — funguje od PyTorch 2.1+, starší verze padají na MPS;
  pokud narazíš, přidej `.cpu()` před STFT a `.to(device)` po ISTFT v `NoiseSynth`
- `float64` tensory — MPS nepodporuje, používáme `float32` všude (OK)
- Gradient clipping (`clip_grad_norm_`) — funguje

---

## TODO

- [ ] Přidat výběr zařízení (auto / cpu / mps / cuda) do GUI záložky Nastroj & Stav
- [ ] Autodetekce v `ddsp.py` místo výchozího `cpu`
- [ ] Otestovat `NoiseSynth` na MPS (STFT path)
