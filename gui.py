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
    for line in proc.stdout:
        log_queue.append(line.rstrip())
        if stop_event.is_set():
            proc.terminate()
            break
    proc.wait()
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
    ithaca_dir = os.path.join(ITHACA_ROOT, name)
    gen_count  = len(glob.glob(os.path.join(ithaca_dir, '*.wav')))
    ext_info   = cfg.get('extract', {})
    trn_info   = cfg.get('training', {})
    gen_info   = cfg.get('generated', {})

    lines = [
        f'Nastroj:   {cfg.get("instrument", "?")}',
        f'Zdroj:     {instrument_dir}',
        f'Workspace: {work_dir}',
        f'Vystup:    {ithaca_dir}',
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

    with gr.Blocks(title='DDSP Neural Vocoder', theme=gr.themes.Soft()) as app:
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
                status_out = gr.Textbox(label='Stav nastroje', lines=10, interactive=False)
                status_btn.click(fn=_read_status, inputs=[instrument_in, workspace_in], outputs=status_out)
                instrument_in.change(fn=_read_status, inputs=[instrument_in, workspace_in], outputs=status_out)
                workspace_in.change(fn=_read_status, inputs=[instrument_in, workspace_in], outputs=status_out)

            # -- Tab: Extrakce --
            with gr.Tab('Extrakce'):
                gr.Markdown('Extrahuje F0, hlasitost a velocity z WAV souboru do NPZ cache.\n'
                            '**Known-F0 mod** (vychozi): F0 z nazvu souboru mXXX — rychle (<0.1s/soubor).\n'
                            '**pyin mod**: pomalý odhad F0 (~20s/soubor) — pro nastroje bez MIDI v nazvu.')
                chunk_sec  = gr.Slider(0, 120, value=0, step=10,
                                       label='chunk-sec (0 = bez rozdeleni, doporuceno pro sample banky)')
                force_pyin = gr.Checkbox(label='--force-pyin (pouzit pomalý pyin misto known-F0)')
                ext_run    = gr.Button('Spustit extrakci', variant='primary')
                ext_stop   = gr.Button('Stop')
                ext_log    = gr.Textbox(label='Vystup', lines=15, interactive=False)
                ext_timer  = gr.Timer(value=2)

                def run_extract(instrument, workspace, chunk_s, fp):
                    if not instrument:
                        return 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['extract', '--instrument', instrument, '--chunk-sec', str(int(chunk_s))]
                    if workspace.strip(): args += ['--workspace', workspace.strip()]
                    if fp: args.append('--force-pyin')
                    return run_command(args, ext_log)

                ext_run.click(fn=run_extract,   inputs=[instrument_in, workspace_in, chunk_sec, force_pyin], outputs=ext_log)
                ext_stop.click(fn=stop_command, outputs=ext_log)
                ext_timer.tick(fn=poll_log,     outputs=ext_log)

            # -- Tab: Uceni --
            with gr.Tab('Uceni'):
                gr.Markdown('Trenuje DDSP model na extrahovanych datech. '
                            'Extrakce probehne automaticky, pokud chybi. '
                            'Po dokonceni se automaticky trénuje i **EnvelopeNet** (~sekund).')
                with gr.Row():
                    model_size = gr.Dropdown(['small', 'medium', 'large'], value='small',
                                              label='Velikost modelu')
                    epochs_sl  = gr.Slider(10, 500, value=100, step=10, label='Pocet epoch')
                    lr_sl      = gr.Slider(1e-5, 1e-3, value=3e-4, step=1e-5,
                                           label='Learning rate', info='Doporuceno: 3e-4')
                resume_chk  = gr.Checkbox(label='Pokracovat od posledniho checkpointu (--resume)')
                with gr.Row():
                    lrn_run     = gr.Button('Spustit uceni', variant='primary')
                    env_run     = gr.Button('Pouze EnvelopeNet', variant='secondary')
                    lrn_stop    = gr.Button('Stop')
                lrn_log    = gr.Textbox(label='Vystup', lines=20, interactive=False)
                lrn_timer  = gr.Timer(value=2)

                def run_learn(instrument, workspace, size, epochs, lr, resume):
                    if not instrument:
                        return 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['learn', '--instrument', instrument,
                            '--model', size, '--epochs', str(int(epochs)),
                            '--lr', f'{lr:.2e}']
                    if workspace.strip(): args += ['--workspace', workspace.strip()]
                    if resume: args.append('--resume')
                    return run_command(args, lrn_log)

                def run_learn_envelope(instrument, workspace):
                    if not instrument:
                        return 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['learn-envelope', '--instrument', instrument]
                    if workspace.strip(): args += ['--workspace', workspace.strip()]
                    return run_command(args, lrn_log)

                lrn_run.click(fn=run_learn,
                               inputs=[instrument_in, workspace_in, model_size, epochs_sl, lr_sl, resume_chk],
                               outputs=lrn_log)
                env_run.click(fn=run_learn_envelope,
                               inputs=[instrument_in, workspace_in],
                               outputs=lrn_log)
                lrn_stop.click(fn=stop_command, outputs=lrn_log)
                lrn_timer.tick(fn=poll_log,     outputs=lrn_log)

            # -- Tab: Generovani --
            with gr.Tab('Generovani'):
                gr.Markdown('Generuje WAV vzorky pomoci natrenovaneho modelu.\n'
                            'Vystup: `C:\\SoundBanks\\IthacaPlayer\\<nastroj>\\`')
                with gr.Row():
                    full_range_chk = gr.Checkbox(label='--full-range (kompletni chromaticka banka bez zdrojovych WAV)')
                    no_skip        = gr.Checkbox(label='--no-skip (prepsat existujici soubory)')
                with gr.Row():
                    midi_lo_sl  = gr.Slider(0, 127, value=21, step=1,
                                             label='MIDI lo (--midi-lo, jen full-range)',
                                             info='21 = A0')
                    midi_hi_sl  = gr.Slider(0, 127, value=108, step=1,
                                             label='MIDI hi (--midi-hi, jen full-range)',
                                             info='108 = C8')
                    vel_layers_sl = gr.Slider(1, 8, value=8, step=1,
                                              label='Velocity vrstvy (--vel-layers, jen full-range)')
                with gr.Row():
                    wet_sl   = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                          label='Wet (1.0 = plny DDSP, 0.0 = original, jen standardni mod)')
                    notes_in = gr.Textbox(label='Noty (prazdne = vse, jen standardni mod)',
                                           placeholder='C4 A3 G3')
                    vel_in   = gr.Textbox(label='Velocity (prazdne = vse, jen standardni mod)',
                                           placeholder='5 7')
                output_in = gr.Textbox(
                    label='Vystupni adresar (prazdne = C:\\SoundBanks\\IthacaPlayer\\<nastroj>\\)',
                    placeholder='',
                )
                gen_run   = gr.Button('Generovat', variant='primary')
                gen_stop  = gr.Button('Stop')
                gen_log   = gr.Textbox(label='Vystup', lines=20, interactive=False)
                gen_timer = gr.Timer(value=2)

                def run_generate(instrument, workspace, full_range, midi_lo, midi_hi,
                                 vel_layers, wet, notes, vel, output, no_skip_val):
                    if not instrument:
                        return 'Zadejte adresar nastroje na zalozce "Nastroj & Stav".'
                    args = ['generate', '--instrument', instrument]
                    if workspace.strip(): args += ['--workspace', workspace.strip()]
                    if full_range:
                        args += ['--full-range',
                                 '--midi-lo', str(int(midi_lo)),
                                 '--midi-hi', str(int(midi_hi)),
                                 '--vel-layers', str(int(vel_layers))]
                    else:
                        args += ['--wet', f'{wet:.2f}']
                        if notes.strip(): args += ['--notes'] + notes.split()
                        if vel.strip():   args += ['--vel']   + vel.split()
                    if output.strip():   args += ['--output', output.strip()]
                    if no_skip_val:      args.append('--no-skip')
                    return run_command(args, gen_log)

                gen_run.click(fn=run_generate,
                               inputs=[instrument_in, workspace_in, full_range_chk,
                                       midi_lo_sl, midi_hi_sl, vel_layers_sl,
                                       wet_sl, notes_in, vel_in, output_in, no_skip],
                               outputs=gen_log)
                gen_stop.click(fn=stop_command, outputs=gen_log)
                gen_timer.tick(fn=poll_log,     outputs=gen_log)

    return app


def main():
    ap = argparse.ArgumentParser(description='DDSP Neural Vocoder GUI')
    ap.add_argument('--port',  type=int, default=7860)
    ap.add_argument('--share', action='store_true', help='Create public Gradio link')
    ap.add_argument('--host',  default='127.0.0.1')
    args = ap.parse_args()

    app = build_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               inbrowser=True)


if __name__ == '__main__':
    main()
