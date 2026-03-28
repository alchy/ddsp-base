"""
gui.py - DDSP Neural Vocoder — Gradio GUI

Launch:
    python gui.py
    python gui.py --port 7860 --share

Opens in browser. Works on Windows and macOS.
"""

import os
import sys
import json
import glob
import subprocess
import threading
import time
import argparse

import gradio as gr

PYTHON = sys.executable
ITHACA_ROOT = os.environ.get('ITHACA_ROOT', r'C:\SoundBanks\IthacaPlayer')


def _run_ddsp(args_list: list, log_queue: list, stop_event: threading.Event):
    """Run `python ddsp.py <args>` in subprocess, feed stdout to log_queue."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ddsp.py')
    cmd    = [PYTHON, '-u', script] + args_list
    proc   = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding='utf-8', errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    def _killer():
        stop_event.wait()       # blokuje dokud neni nastaveno
        if proc.poll() is None:
            proc.terminate()
            time.sleep(1.5)
            if proc.poll() is None:
                proc.kill()

    threading.Thread(target=_killer, daemon=True).start()

    for line in proc.stdout:
        log_queue.append(line.rstrip())
    proc.wait()
    if stop_event.is_set():
        log_queue.append('\n[zastaveno uzivatelem]')
    else:
        log_queue.append(f'\n[exit code {proc.returncode}]')


def _read_status(instrument_dir: str, workspace_dir: str = '') -> str:
    """Read instrument.json and format as status string."""
    if not instrument_dir or not os.path.isdir(instrument_dir):
        return 'Vyberte adresar s nastroji.'
    name     = os.path.basename(instrument_dir.rstrip('/\\'))
    work_dir = workspace_dir.strip() or (instrument_dir.rstrip('/\\') + '-ddsp')
    config_path = os.path.join(work_dir, 'instrument.json')
    if not os.path.exists(config_path):
        return f'Workspace nenalezen: {work_dir}\nSpustte nejprve Extract nebo Learn.'

    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)

    def tick(v): return 'ok' if v else '--'

    npz_count  = len(glob.glob(os.path.join(work_dir, 'extracts', '*.npz')))
    has_ckpt   = os.path.exists(os.path.join(work_dir, 'checkpoints', 'best.pt'))
    has_env    = os.path.exists(os.path.join(work_dir, 'checkpoints', 'envelope.pt'))
    ithaca_dir = os.path.join(ITHACA_ROOT, 'generated', name)
    gen_count  = len(glob.glob(os.path.join(ithaca_dir, '*.wav')))
    ext_info   = cfg.get('extract', {})
    trn_info   = cfg.get('training', {})
    gen_info   = cfg.get('generated', {})

    extracts_dir   = os.path.join(work_dir, 'extracts')
    checkpoints_dir = os.path.join(work_dir, 'checkpoints')
    lines = [
        f'Nastroj:     {cfg.get("instrument", "?")}',
        f'Zdroj:       {instrument_dir}',
        f'Workspace:   {work_dir}',
        f'Extrakce:    {extracts_dir}',
        f'Checkpointy: {checkpoints_dir}',
        f'Vystup:      {ithaca_dir}',
        '',
        f'[{tick(npz_count)}] Extrakce    {npz_count} NPZ souboru' +
            (f'  [{ext_info.get("completed_at","")}]' if ext_info else ''),
        f'[{tick(has_ckpt)}] Model       {cfg.get("model_size","?")}' +
            (f'  ep={trn_info.get("epochs_completed","?")}  '
             f'best_val={trn_info.get("best_val","?")}  '
             f'[{trn_info.get("last_trained","")}]' if trn_info else '  (nenatrenovano)'),
        f'[{tick(has_env)}] EnvelopeNet {"ok" if has_env else "(chybi — spust learn-envelope)"}',
        f'[{tick(gen_count)}] Generovani  {gen_count} WAV souboru  -> {ithaca_dir}' +
            (f'  [{gen_info.get("generated_at","")}]' if gen_info else ''),
    ]
    return '\n'.join(lines)


def build_ui():
    _current = {'log': [], 'stop': threading.Event(), 'thread': None}

    def run_command(args_list, progress_box):
        """Start ddsp command in background, stream output."""
        if _current['thread'] and _current['thread'].is_alive():
            return 'Jiny prikaz jiz bezi. Pockejte nebo stisknete Stop.'
        _current['log'].clear()
        _current['stop'].clear()
        t = threading.Thread(target=_run_ddsp,
                              args=(args_list, _current['log'], _current['stop']),
                              daemon=True)
        _current['thread'] = t
        t.start()
        return 'Spusteno...'

    def stop_command():
        _current['stop'].set()
        return 'Zastaven.'

    def poll_log():
        return '\n'.join(_current['log'][-200:])

    with gr.Blocks(title='DDSP Neural Vocoder') as app:
        gr.Markdown('# DDSP Neural Vocoder\nUceni timbru nastroje a generovani vzorku.')

        with gr.Tabs():

            # -- Tab: Nastaveni & Stav --
            with gr.Tab('Nastroj & Stav'):
                with gr.Row():
                    instrument_in = gr.Textbox(
                        label='Adresar se zdrojovymi WAV soubory',
                        placeholder=r'C:\SoundBanks\ddsp\vintage-vibe',
                        scale=4,
                    )
                    status_btn = gr.Button('Obnovit stav', scale=1)
                workspace_in = gr.Textbox(
                    label='Workspace (prazdne = <instrument>-ddsp/)',
                    placeholder=r'C:\SoundBanks\ddsp\vintage-vibe-ddsp',
                    info='Vyplnte pouze pokud se workspace lisi od vychozi cesty.',
                )
                device_dd = gr.Dropdown(
                    ['auto', 'cpu', 'mps', 'cuda'], value='auto',
                    label='Zarizeni (device)',
                    info='auto = CUDA → MPS (Apple Silicon) → CPU. Vyberte mps pro Apple M-series.',
                )
                status_out = gr.Textbox(label='Stav nastroje', lines=10, interactive=False)
                status_btn.click(fn=_read_status, inputs=[instrument_in, workspace_in], outputs=status_out)
                instrument_in.change(fn=_read_status, inputs=[instrument_in, workspace_in], outputs=status_out)
                workspace_in.change(fn=_read_status, inputs=[instrument_in, workspace_in], outputs=status_out)

            # -- Tab: Extrakce --
            with gr.Tab('Extrakce'):
                gr.Markdown('Extrahuje F0, hlasitost a velocity z WAV souboru do NPZ cache.\n'
                            '**Known-F0 mod** (vychozi): F0 z nazvu souboru mXXX — rychle (<0.1s/soubor).\n'
                            '**pyin mod**: pomalý odhad F0 (~20s/soubor) — pro nastroje bez MIDI v nazvu.')
                ext_status_out = gr.Textbox(label='Hotove extrakce (NPZ cache)',
                                            lines=6, interactive=False, max_lines=6)
                ext_status_timer = gr.Timer(value=3)
                chunk_sec  = gr.Slider(0, 120, value=0, step=10,
                                       label='chunk-sec (0 = bez rozdeleni, doporuceno pro sample banky)',
                                       info='Rozdelit dlouhe WAV na useky. 0 = cely soubor najednou.')
                force_pyin = gr.Checkbox(label='--force-pyin (pouzit pomalý pyin misto known-F0)',
                                         info='Jen pokud nazev souboru neobsahuje MIDI cislo (mXXX).')
                with gr.Row():
                    ext_run  = gr.Button('Spustit extrakci', variant='primary')
                    ext_stop = gr.Button('Stop')
                ext_log    = gr.Textbox(label='Vystup', lines=12, interactive=False)
                ext_timer  = gr.Timer(value=2)

                def read_ext_status(instrument, workspace):
                    if not instrument:
                        return 'Zadejte adresar nastroje.'
                    work_dir  = (workspace or '').strip() or ((instrument or '').rstrip('/\\') + '-ddsp')
                    ext_dir   = os.path.join(work_dir, 'extracts')
                    src_wavs  = len(glob.glob(os.path.join(instrument, '**', '*.wav'),
                                              recursive=True))
                    npz_files = sorted(glob.glob(os.path.join(ext_dir, '*.npz')))
                    n_npz     = len(npz_files)
                    if n_npz == 0:
                        return f'Zadne NPZ soubory v {ext_dir}\nSpustte extrakci.'
                    total_mb  = sum(os.path.getsize(p) for p in npz_files) / 1e6
                    lines     = [f'{n_npz} NPZ  /  {src_wavs} WAV zdroju  '
                                 f'({total_mb:.0f} MB)  ->  {ext_dir}', '']
                    lines    += [os.path.basename(p) for p in npz_files[-20:]]
                    if n_npz > 20:
                        lines.insert(1, f'(zobrazeno poslednich 20 z {n_npz})')
                    return '\n'.join(lines)

                ext_cmd_out = gr.Textbox(label='Sestaveny prikaz', interactive=False, lines=2)

                def run_extract(instrument, workspace, chunk_s, fp, device):
                    if not instrument:
                        return '', 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['extract', '--instrument', instrument, '--chunk-sec', str(int(chunk_s))]
                    if (workspace or '').strip(): args += ['--workspace', (workspace or '').strip()]
                    if fp: args.append('--force-pyin')
                    cmd_str = 'python ddsp.py ' + ' '.join(args)
                    return cmd_str, run_command(args, ext_log)

                ext_status_timer.tick(fn=read_ext_status,
                                      inputs=[instrument_in, workspace_in],
                                      outputs=ext_status_out)
                instrument_in.change(fn=read_ext_status,
                                     inputs=[instrument_in, workspace_in],
                                     outputs=ext_status_out)
                workspace_in.change(fn=read_ext_status,
                                    inputs=[instrument_in, workspace_in],
                                    outputs=ext_status_out)
                ext_run.click(fn=run_extract,   inputs=[instrument_in, workspace_in, chunk_sec, force_pyin, device_dd], outputs=[ext_cmd_out, ext_log])
                ext_stop.click(fn=stop_command, outputs=ext_log)
                ext_timer.tick(fn=poll_log,     outputs=ext_log)

            # -- Tab: EnvelopeNet --
            with gr.Tab('EnvelopeNet'):
                gr.Markdown(
                    '### Trenovani EnvelopeNet\n'
                    'Mala MLP (~30K param) ktera se uci tvar hlasitostni obalky z NPZ dat.\n'
                    '**Vstup:** (midi, velocity)  **Vystup:** tvar obalky + delka tonu\n\n'
                    'Pouziva warped casovou osu (power-law) pro jemne rozliseni attack faze '
                    '(~2 ms v prvnich 500 ms), a vazeny MSE loss pro zdurazneni attacku.\n'
                    '**Spust pred DDSP Model** (coupled mode) nebo samostatne kdykoli.'
                )
                env_status_out   = gr.Textbox(label='Stav EnvelopeNet', lines=2,
                                              interactive=False, max_lines=2)
                env_status_timer = gr.Timer(value=5)

                def read_env_status(instrument, workspace):
                    if not instrument:
                        return 'Zadejte adresar nastroje.'
                    work_dir = (workspace or '').strip() or ((instrument or '').rstrip('/\\') + '-ddsp')
                    env_pt   = os.path.join(work_dir, 'checkpoints', 'envelope.pt')
                    if not os.path.exists(env_pt):
                        return 'Nenatrenovano — envelope.pt neexistuje.'
                    mb = os.path.getsize(env_pt) / 1e6
                    ts = time.strftime('%Y-%m-%d %H:%M',
                                       time.localtime(os.path.getmtime(env_pt)))
                    try:
                        import torch as _torch
                        ckpt   = _torch.load(env_pt, map_location='cpu', weights_only=True)
                        n_env  = ckpt.get('n_env', '?')
                        warp   = ckpt.get('warp', '?')
                        detail = f'n_env={n_env}  warp={warp}'
                    except Exception:
                        detail = '(nelze nacist detail)'
                    return (f'HOTOVO  {detail}  [{ts}]\n'
                            f'checkpoint: {env_pt}  ({mb:.2f} MB)')

                env_status_timer.tick(fn=read_env_status,
                                      inputs=[instrument_in, workspace_in],
                                      outputs=env_status_out)
                instrument_in.change(fn=read_env_status,
                                     inputs=[instrument_in, workspace_in],
                                     outputs=env_status_out)
                workspace_in.change(fn=read_env_status,
                                    inputs=[instrument_in, workspace_in],
                                    outputs=env_status_out)

                with gr.Row():
                    env_epochs_sl = gr.Slider(100, 5000, value=1000, step=100,
                                              label='Pocet epoch',
                                              info='1000 epoch staci pro vetsinu nastroju; '
                                                   'zvys pro komplexnejsi obalky')
                    env_lr_sl     = gr.Slider(1e-5, 1e-2, value=1e-3, step=1e-5,
                                              label='Learning rate',
                                              info='Doporuceno: 1e-3')
                with gr.Row():
                    env_warp_sl   = gr.Slider(1.0, 8.0, value=4.0, step=0.5,
                                              label='Envelope Warp',
                                              info='Stupen koncentrace bodu na zacatku obalky. '
                                                   '4.0 = prvni polovina bodu pokryva prvnich 6% '
                                                   'tonu (attack). Vys = jemnejsi attack rozliseni.')
                    env_nenv_sl   = gr.Slider(128, 1024, value=512, step=64,
                                              label='N ENV (pocet ridicicich bodu)',
                                              info='Pocet bodu na warped ose. 512 = ~30K param.')
                    env_atk_w_sl  = gr.Slider(1.0, 20.0, value=5.0, step=0.5,
                                              label='Attack weight (MSE)',
                                              info='Nasobitel chyby v attack oblasti (prvni ~4% '
                                                   'bodu). 5.0 = attack prispiva 5x vice do loss.')
                env_cmd_out = gr.Textbox(label='Sestaveny prikaz', interactive=False, lines=2)
                with gr.Row():
                    env_run  = gr.Button('Spustit trenovani EnvelopeNet', variant='primary')
                    env_stop = gr.Button('Stop')
                env_log   = gr.Textbox(label='Vystup', lines=15, interactive=False)
                env_timer = gr.Timer(value=2)

                def run_learn_envelope(instrument, workspace, epochs, lr, warp, n_env, atk_w, device):
                    if not instrument:
                        return '', 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['learn-envelope', '--instrument', instrument,
                            '--epochs', str(int(epochs)),
                            '--lr', f'{lr:.2e}',
                            '--envelope-warp', f'{warp:.2f}',
                            '--n-env', str(int(n_env)),
                            '--attack-weight', f'{atk_w:.2f}',
                            '--device', device]
                    if (workspace or '').strip(): args += ['--workspace', (workspace or '').strip()]
                    cmd_str = 'python ddsp.py ' + ' '.join(args)
                    return cmd_str, run_command(args, env_log)

                env_run.click(fn=run_learn_envelope,
                               inputs=[instrument_in, workspace_in, env_epochs_sl,
                                       env_lr_sl, env_warp_sl, env_nenv_sl, env_atk_w_sl,
                                       device_dd],
                               outputs=[env_cmd_out, env_log])
                env_stop.click(fn=stop_command, outputs=env_log)
                env_timer.tick(fn=poll_log, outputs=env_log)

            # -- Tab: DDSP Model --
            with gr.Tab('DDSP Model'):
                gr.Markdown(
                    '### Trenovani DDSP modelu\n'
                    '**Decoupled timbre architektura**: sit se uci POUZE timbre (overtone balance, '
                    'barvu zvuku) z F0 a velocity — bez hlasitosti. Hlasitostní obalka se aplikuje '
                    'az po synteze jako linearni multiplikator (dB → linear scale).\n\n'
                    'Tato architektura oddeluje "jak nastroj zní" od "jak hlasity je" — model se '
                    'naucí cistou barvu zvuku nezavislou na dynamice. '
                    'Extrakce probehne automaticky pokud chybi.'
                )
                ddsp_status_out   = gr.Textbox(label='Stav modelu', lines=2,
                                               interactive=False, max_lines=2)
                ddsp_status_timer = gr.Timer(value=5)

                def read_ddsp_status(instrument, workspace):
                    if not instrument:
                        return 'Zadejte adresar nastroje.'
                    work_dir = (workspace or '').strip() or ((instrument or '').rstrip('/\\') + '-ddsp')
                    best_pt  = os.path.join(work_dir, 'checkpoints', 'best.pt')
                    cfg_path = os.path.join(work_dir, 'instrument.json')
                    if not os.path.exists(best_pt):
                        return 'Nenatrenovano — checkpoint neexistuje.'
                    trn = {}
                    if os.path.exists(cfg_path):
                        with open(cfg_path, encoding='utf-8') as f:
                            trn = json.load(f).get('training', {})
                    size   = trn.get('model_size') or json.load(open(cfg_path, encoding='utf-8')).get('model_size', '?') if os.path.exists(cfg_path) else '?'
                    ep     = trn.get('epochs_completed', '?')
                    bv     = trn.get('best_val', '?')
                    ts     = trn.get('last_trained', '')
                    mb     = os.path.getsize(best_pt) / 1e6
                    return (f'HOTOVO  model={size}  ep={ep}  best_val={bv}  '
                            f'[{ts}]\n'
                            f'checkpoint: {best_pt}  ({mb:.1f} MB)')

                ddsp_status_timer.tick(fn=read_ddsp_status,
                                       inputs=[instrument_in, workspace_in],
                                       outputs=ddsp_status_out)
                instrument_in.change(fn=read_ddsp_status,
                                     inputs=[instrument_in, workspace_in],
                                     outputs=ddsp_status_out)
                workspace_in.change(fn=read_ddsp_status,
                                    inputs=[instrument_in, workspace_in],
                                    outputs=ddsp_status_out)

                with gr.Row():
                    model_size = gr.Dropdown(
                        ['small', 'medium', 'large'], value='small',
                        label='Velikost modelu',
                        info='small ~238K param (rychle CPU), medium ~696K, large ~3.4M'
                    )
                    epochs_sl  = gr.Slider(10, 500, value=100, step=10,
                                           label='Pocet epoch',
                                           info='1 epocha ≈ 6-18 min CPU, dle velikosti modelu')
                    lr_sl      = gr.Slider(1e-5, 1e-3, value=3e-4, step=1e-5,
                                           label='Learning rate',
                                           info='Doporuceno: 3e-4; sniz pokud loss osciluje')
                resume_chk = gr.Checkbox(
                    label='Pokracovat od posledniho checkpointu (--resume)',
                    info='Pouzij po preruseni tréninku — zachova natrenovane vahy'
                )
                lrn_cmd_out = gr.Textbox(label='Sestaveny prikaz', interactive=False, lines=2)
                with gr.Row():
                    lrn_run  = gr.Button('Spustit uceni', variant='primary')
                    lrn_stop = gr.Button('Stop')
                lrn_log   = gr.Textbox(label='Vystup', lines=12, interactive=False)
                lrn_timer = gr.Timer(value=2)

                gr.Markdown('#### train.log (posledni epochy)')
                trainlog_out   = gr.Textbox(label='train.log', lines=12,
                                            interactive=False, max_lines=12)
                trainlog_timer = gr.Timer(value=3)

                def run_learn(instrument, workspace, size, epochs, lr, resume, device):
                    if not instrument:
                        return '', 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['learn', '--instrument', instrument,
                            '--model', size, '--epochs', str(int(epochs)),
                            '--lr', f'{lr:.2e}',
                            '--device', device]
                    if (workspace or '').strip(): args += ['--workspace', (workspace or '').strip()]
                    if resume:  args.append('--resume')
                    cmd_str = 'python ddsp.py ' + ' '.join(args)
                    return cmd_str, run_command(args, lrn_log)

                def read_train_log(instrument, workspace):
                    work_dir = (workspace or '').strip() or ((instrument or '').rstrip('/\\') + '-ddsp')
                    log_path = os.path.join(work_dir, 'train.log')
                    if not os.path.exists(log_path):
                        return '(train.log nenalezen)'
                    with open(log_path, encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                    return ''.join(lines[-60:])

                lrn_run.click(fn=run_learn,
                               inputs=[instrument_in, workspace_in, model_size,
                                       epochs_sl, lr_sl, resume_chk, device_dd],
                               outputs=[lrn_cmd_out, lrn_log])
                lrn_stop.click(fn=stop_command, outputs=lrn_log)
                lrn_timer.tick(fn=poll_log, outputs=lrn_log)
                trainlog_timer.tick(fn=read_train_log,
                                    inputs=[instrument_in, workspace_in],
                                    outputs=trainlog_out)
                instrument_in.change(fn=read_train_log,
                                     inputs=[instrument_in, workspace_in],
                                     outputs=trainlog_out)
                workspace_in.change(fn=read_train_log,
                                    inputs=[instrument_in, workspace_in],
                                    outputs=trainlog_out)

            # -- Tab: Generovani --
            with gr.Tab('Generovani'):
                gr.Markdown(
                    'Generuje WAV vzorky pomoci natrenovaneho modelu.\n'
                    'Vystup: `C:\\SoundBanks\\IthacaPlayer\\generated\\<nastroj>\\`'
                )
                gen_status_out   = gr.Textbox(label='Stav generovani', lines=3,
                                              interactive=False, max_lines=3)
                gen_status_timer = gr.Timer(value=5)

                def read_gen_status(instrument, workspace, output):
                    if not instrument:
                        return 'Zadejte adresar nastroje.'
                    name      = os.path.basename(instrument.rstrip('/\\'))
                    work_dir  = (workspace or '').strip() or (instrument.rstrip('/\\') + '-ddsp')
                    cfg_path  = os.path.join(work_dir, 'instrument.json')
                    model_size = '?'
                    if os.path.exists(cfg_path):
                        with open(cfg_path, encoding='utf-8') as f:
                            model_size = json.load(f).get('model_size', '?')
                    out_dir   = (output or '').strip() or os.path.join(ITHACA_ROOT, 'generated', name)
                    wav_files = glob.glob(os.path.join(out_dir, '*.wav'))
                    n_wav     = len(wav_files)
                    if n_wav == 0:
                        return (f'model: {model_size}\n'
                                f'Zadne WAV soubory v {out_dir}')
                    total_mb  = sum(os.path.getsize(p) for p in wav_files) / 1e6
                    ts = time.strftime('%Y-%m-%d %H:%M',
                                       time.localtime(max(os.path.getmtime(p) for p in wav_files)))
                    return (f'model: {model_size}\n'
                            f'HOTOVO  {n_wav} WAV souboru  ({total_mb:.0f} MB)  '
                            f'posledni zmena [{ts}]\n'
                            f'vystup: {out_dir}')

                with gr.Row():
                    full_range_chk = gr.Checkbox(
                        label='Full-range mod (kompletni chromaticka banka)',
                        info='Bez zdrojovych WAV — syntetizuje vsechny noty od MIDI lo do hi. '
                             'Vyzaduje natrenovany EnvelopeNet nebo NPZ cache.'
                    )
                    no_skip = gr.Checkbox(
                        label='--no-skip (prepsat existujici soubory)',
                        info='Vychozi: existujici WAV preskoci. Zaskrtni pro vynuceni prepisu.'
                    )
                with gr.Row():
                    midi_lo_sl  = gr.Slider(0, 127, value=21, step=1,
                                             label='MIDI lo (jen full-range)',
                                             info='21 = A0 (nejnizsi nota klaviru)')
                    midi_hi_sl  = gr.Slider(0, 127, value=108, step=1,
                                             label='MIDI hi (jen full-range)',
                                             info='108 = C8 (nejvyssi nota klaviru)')
                    vel_layers_sl = gr.Slider(1, 8, value=8, step=1,
                                              label='Velocity vrstvy',
                                              info='Pocet velocity vrstev pro kazdou notu: '
                                                   '1 = pouze vel 0, 8 = vel 0-7')
                with gr.Row():
                    env_source_radio = gr.Radio(
                        choices=['auto', 'envelopenet', 'npz'],
                        value='auto',
                        label='Zdroj obalek',
                        info='auto: EnvelopeNet pokud existuje, jinak NPZ sablony. '
                             'envelopenet: vzdy NN (vyzaduje envelope.pt). '
                             'npz: vzdy nejblizsi sablona z extrakci (hrubsi, bez interpolace).'
                    )
                    attack_ramp_sl = gr.Slider(0, 50, value=10, step=1,
                                               label='Attack ramp (ms)',
                                               info='Raised-cosine nabehu na zacatku tonu. '
                                                    '0 = vypnuto. 10 ms = prirozeny uder kladivka.')
                with gr.Row():
                    wet_sl = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                       label='Wet (jen note-list mod)',
                                       info='1.0 = cisty model. < 1.0 = mix s originalem '
                                            '(vyzaduje zdrojove WAV; bez nich preskoci danou notu).')
                    inh_scale_sl = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                             label='Inharmonicity scale',
                                             info='0.0 = ciste harmonicke, 1.0 = naucene B, '
                                                  '2.0 = zesilenа inharmonicita.')
                    decay_scale_sl = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                               label='Decay scale',
                                               info='0.0 = zadny fyzikalni decay (plochy sustain), '
                                                    '1.0 = naucene b1/b3 (fyzikalni), '
                                                    '2.0 = rychlejsi doznivani.')
                    notes_in = gr.Textbox(label='Noty (note-list mod, prazdne = chyba)',
                                           placeholder='C3 A3 C4 A4 C5',
                                           info='Seznam not k vygenerovani. Pouziva se kdyz '
                                                'full-range neni zaskrtnuto.')
                output_in = gr.Textbox(
                    label='Vystupni adresar (prazdne = IthacaPlayer/generated/<nastroj>/)',
                    placeholder='',
                    info=f'Vychozi: {ITHACA_ROOT}\\generated\\<nastroj>\\  '
                         r'Muze byt relativni i absolutni cesta.'
                )
                gen_cmd_out = gr.Textbox(label='Sestaveny prikaz', interactive=False, lines=2)
                with gr.Row():
                    gen_run  = gr.Button('Generovat', variant='primary')
                    gen_stop = gr.Button('Stop')
                gen_log   = gr.Textbox(label='Vystup', lines=20, interactive=False)
                gen_timer = gr.Timer(value=2)

                def run_generate(instrument, workspace, full_range, midi_lo, midi_hi,
                                 vel_layers, env_src, atk_ramp, wet, inh_scale, decay_scale,
                                 notes, output, no_skip_val, device):
                    if not instrument:
                        return '', 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['generate', '--instrument', instrument,
                            '--envelope-source', env_src,
                            '--attack-ramp-ms', str(int(atk_ramp)),
                            '--inharmonicity-scale', f'{inh_scale:.2f}',
                            '--decay-scale', f'{decay_scale:.2f}',
                            '--vel-layers', str(int(vel_layers)),
                            '--device', device]
                    if (workspace or '').strip(): args += ['--workspace', (workspace or '').strip()]
                    if full_range:
                        args += ['--full-range',
                                 '--midi-lo', str(int(midi_lo)),
                                 '--midi-hi', str(int(midi_hi))]
                    else:
                        args += ['--wet', f'{wet:.2f}']
                        if notes.strip(): args += ['--notes'] + notes.split()
                    if (output or '').strip(): args += ['--output', output.strip()]
                    if no_skip_val:      args.append('--no-skip')
                    cmd_str = 'python ddsp.py ' + ' '.join(args)
                    return cmd_str, run_command(args, gen_log)

                gen_run.click(fn=run_generate,
                               inputs=[instrument_in, workspace_in, full_range_chk,
                                       midi_lo_sl, midi_hi_sl, vel_layers_sl,
                                       env_source_radio, attack_ramp_sl,
                                       wet_sl, inh_scale_sl, decay_scale_sl, notes_in,
                                       output_in, no_skip, device_dd],
                               outputs=[gen_cmd_out, gen_log])
                gen_stop.click(fn=stop_command, outputs=gen_log)
                gen_timer.tick(fn=poll_log, outputs=gen_log)
                # gen_status wiring — output_in must be defined first
                gen_status_timer.tick(fn=read_gen_status,
                                      inputs=[instrument_in, workspace_in, output_in],
                                      outputs=gen_status_out)
                instrument_in.change(fn=read_gen_status,
                                     inputs=[instrument_in, workspace_in, output_in],
                                     outputs=gen_status_out)
                workspace_in.change(fn=read_gen_status,
                                    inputs=[instrument_in, workspace_in, output_in],
                                    outputs=gen_status_out)
                output_in.change(fn=read_gen_status,
                                 inputs=[instrument_in, workspace_in, output_in],
                                 outputs=gen_status_out)

    return app


def main():
    ap = argparse.ArgumentParser(description='DDSP Neural Vocoder GUI')
    ap.add_argument('--port',  type=int, default=7860)
    ap.add_argument('--share', action='store_true', help='Create public Gradio link')
    ap.add_argument('--host',  default='127.0.0.1')
    args = ap.parse_args()

    app = build_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               inbrowser=True, theme=gr.themes.Soft())


if __name__ == '__main__':
    main()
