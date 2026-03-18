import cmd
import os
import time
import random
import numpy as np
import serial
import time
from psychopy import core, gui
from psychopy.data import ExperimentHandler
from psychopy.hardware import keyboard
from PIL import Image
import tempfile
import shutil
from pathlib import Path
import threedipa.utils as utils
from threedipa.renderer.haploscopeRender import HaplscopeRender2D
from threedipa.renderer.haploscopeConfig import monitor_settings, physical_calibration
from threedipa.procedure import OneIntervalDraw
from threedipa.stimuli.stimulus2D import Stimulus2DImage
# from johnston_renderer import generate_stimulus, FINAL_W_PX, FINAL_H_PX

# ============================================================
# EXPERIMENT DESIGN — JOHNSTON (1991) FAITHFUL REPLICATION
# ============================================================
#
# PHASE 1 — STAIRCASE (pilot, ~20-40 trials per half-height)
#   Interleaved 1-up 1-down staircases, one per half-height.
#   Stops after N_REVERSALS. PSE = mean of last reversals.
#   Purpose: locate each participant's PSE so we can centre
#   the method-of-constants range around it.
#
# PHASE 2 — METHOD OF CONSTANTS (main experiment, 200 trials)
#   5 depth factors centred on the staircase PSE per half-height.
#   Each presented N_REPS times in randomised order.
#   Johnston used 40 reps × 5 depths = 200 per half-height.
#   Here we use 20 reps × 5 depths × 2 half-heights = 200 total.
#
# DEPTH FACTOR SPACING (Johnston used 5 b values per run):
#   Spaced ±2 and ±1 steps around PSE, where STEP = 0.15
#   e.g. PSE=0.40 → [0.10, 0.25, 0.40, 0.55, 0.70]
# ============================================================

HALF_HEIGHTS = [0.04]   # meters — Johnston used 2.5–7.5 cm; we use 2 sizes
TEXTURES = [0, 1, 2]
DEPTH_FACTORS = [0.5, 0.75, 1, 1.25, 1.5]
SHAPE_DICT = {0.5:'-2', 0.75:'-1', 1:'0', 1.25:'1', 1.5:'2'}
VIBRATION_LEVELS = [0, 2, 3]
# --- Staircase parameters ---
SC_START_DF    = 1.25   # start well above expected PSE
SC_STEP        = 0.5   # step size
SC_N_REVERSALS = 8      # stop after 8 reversals; PSE = mean of last 6
SC_MIN_DF      = 0.5
SC_MAX_DF      = 2.00

# --- Method of constants parameters ---
MOC_N_STEPS  = 5       # number of depth factor levels (Johnston used 5)
MOC_SPACING  = 0.25    # spacing between levels (in df units)
MOC_N_REPS   = [10, 10, 5]      # repetitions per condition → 20×5×2 = 200 trials
MOC_N_Tex    = 3

TEXTURE_DICT = {0:"noise", 1:"lines", 2:"circle"}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_empty_stimulus():
    img_arr = np.zeros((1024, 1024), dtype=np.uint8)
    
    return img_arr
    
def load_stimulus(texture, df, data_path):
    img_path = Path(os.path.join(data_path, f"{TEXTURE_DICT[texture]}_{df}.png"))
    img = Image.open(img_path)
    img_arr = np.array(img)
    alpha = img_arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    left = xs.min()
    right = xs.max()+1
    top = ys.min()
    bottom=ys.max()
    img.crop((left, top, right, bottom))
    img_arr = np.array(img)

    return img_arr

    
    
def exit_experiment(phaseTracker):
    phaseTracker.set_stimulus_phase(utils.StimulusPhase.NONE)
    phaseTracker.set_experiment_phase(utils.ExperimentPhase.END)


def trialPhase_timing(phaseTracker, trial_time, config):
    if phaseTracker.get_experiment_phase() == utils.ExperimentPhase.TRIAL:
        if trial_time.getTime() <= config['FixationDuration']:
            phaseTracker.set_stimulus_phase(utils.StimulusPhase.FIXATION)
        elif trial_time.getTime() <= config['StimulusDuration'] + config['FixationDuration']:
            phaseTracker.set_stimulus_phase(utils.StimulusPhase.STIMULUS)
        elif trial_time.getTime() > config['StimulusDuration'] + config['FixationDuration']:
            phaseTracker.set_stimulus_phase(utils.StimulusPhase.NONE)
            phaseTracker.set_response_phase(utils.ResponsePhase.WAIT_FOR_RESPONSE)


def reset_phase_tracker(phaseTracker):
    phaseTracker.set_experiment_phase(utils.ExperimentPhase.PRE_TRIAL)
    phaseTracker.set_stimulus_phase(utils.StimulusPhase.FIXATION)
    phaseTracker.set_response_phase(utils.ResponsePhase.NO_RESPONSE)


def render_to_stimulus(l_arr, r_arr, tmp_dir):
    l_path = os.path.join(tmp_dir, "current_L.png")
    r_path = os.path.join(tmp_dir, "current_R.png")

    Image.fromarray(l_arr).save(l_path)
    Image.fromarray(r_arr).save(r_path)

    return Stimulus2DImage(
        left_image_path=l_path,
        right_image_path=r_path
    )

def run_single_trial(renderer, kb, phaseTracker, trial_time, parameters,
                     stimulus, stimulusVisualAngle):
    """
    Run one trial: fixation → stimulus → response.
    Returns (response_key_str, rt) or raises SystemExit on escape.
    """
    stimulus.visual_size_degrees = (stimulusVisualAngle, stimulusVisualAngle)
    reset_phase_tracker(phaseTracker)
    phaseTracker.set_experiment_phase(utils.ExperimentPhase.TRIAL)
    trial_time.reset()
    rt = None
    response_key = None

    while phaseTracker.get_experiment_phase() == utils.ExperimentPhase.TRIAL:
        trialPhase_timing(phaseTracker, trial_time, parameters['parameters'])
        phase = phaseTracker.get_stimulus_phase()
        OneIntervalDraw(renderer, stimulus, phase)

        if phaseTracker.get_response_phase() == utils.ResponsePhase.WAIT_FOR_RESPONSE:
            kb.clock.reset()
            kb.clearEvents()
            response_key = kb.waitKeys(
                keyList    = ['num_3', '3', '6', 'num_6', 'escape'],
                waitRelease= False,
            )
            if response_key:
                rt = kb.clock.getTime()
                phaseTracker.set_response_phase(utils.ResponsePhase.RESPONSE_RECEIVED)
                break

    rname = response_key[0].name
    if rname == 'escape':
        exit_experiment(phaseTracker)
        raise SystemExit('Escape pressed')

    phaseTracker.set_experiment_phase(utils.ExperimentPhase.POST_TRIAL)
    if rname.startswith('num_'):
        rname = rname[4:]
    return rname, rt


def df_range_from_pse(pse, n_steps, spacing, min_df, max_df):
    """
    Build n_steps depth factors centred on pse, evenly spaced by spacing.
    Clamps to [min_df, max_df].

    e.g. pse=0.40, n=5, spacing=0.15
         → [0.10, 0.25, 0.40, 0.55, 0.70]
    """
    half = (n_steps - 1) / 2
    factors = [pse + (i - half) * spacing for i in range(n_steps)]
    factors = [max(min_df, min(max_df, f)) for f in factors]
    return sorted(set(round(f, 4) for f in factors))


# ============================================================
# STAIRCASE CLASS
# ============================================================

class Staircase:
    """
    1-up 1-down staircase homing in on PSE (50% point).

    3 = stretched → df decreases (was too deep)
    6 = squashed  → df increases (was too shallow)

    PSE = mean of reversal values, excluding first 2 (noisy).
    """

    def __init__(self, half_height):
        self.half_height    = half_height
        self.df             = SC_START_DF
        self.direction      = None
        self.reversal_count = 0
        self.reversal_dfs   = []
        self.history        = []
        self.trial_count    = 0
        self.finished       = False
        self.texture        = 2

    def current_df(self):
        return self.df

    def is_finished(self):
        return self.finished

    def update(self, response_key):
        """response_key: '3' (stretched/deep) or '6' (squashed/shallow)"""
        perceived     = 'deep' if response_key == '3' else 'shallow'
        new_direction = 'down' if perceived == 'deep' else 'up'

        self.history.append((self.df, perceived))
        self.trial_count += 1

        if self.direction is not None and new_direction != self.direction:
            self.reversal_count += 1
            self.reversal_dfs.append(self.df)
            if self.reversal_count >= SC_N_REVERSALS:
                self.finished = True
                return

        self.direction = new_direction
        if perceived == 'deep':
            self.df = max(SC_MIN_DF, self.df - SC_STEP)
        else:
            self.df = min(SC_MAX_DF, self.df + SC_STEP)

    def pse(self):
        """Exclude first 2 reversals (acquisition noise), average the rest."""
        vals = self.reversal_dfs[2:] if len(self.reversal_dfs) > 2 else self.reversal_dfs
        return float(np.mean(vals)) if vals else SC_START_DF / 2


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Participant dialog
    # ----------------------------------------------------------
    info = {'Participant ID': 'test', 'IOD (mm)': '56', 'Session': '1'}
    dlg  = gui.DlgFromDict(info, title='Johnston Stereopsis')
    escaped = False
    if not dlg.OK:
        core.quit()
    
    iod_m = float(info['IOD (mm)']) / 1000.0

    # ----------------------------------------------------------
    # 2. Paths and parameters
    # ----------------------------------------------------------
    exp_dir = os.path.join(".", "templates", "group3Template")
    data_v = os.path.join(exp_dir, "data", f"group3_{info['Participant ID']}_{info['Session']}_visual_only")
    data_h = os.path.join(exp_dir, "data", f"group3_{info['Participant ID']}_{info['Session']}_haptic_only")
    data_vh = os.path.join(exp_dir, "data", f"group3_{info['Participant ID']}_{info['Session']}_visual_haptic")
    
    parameters = utils.parse_parameters_file(os.path.join(exp_dir, 'parameters.txt'))
    debug_mode = parameters['parameters']['Debug']
    fixationDistance = parameters['parameters']['FixationDistance']
    stimulusVisualAngle = parameters['parameters']['VisualAngle']

    os.makedirs(os.path.join(exp_dir, 'data'), exist_ok=True)
    tmp_dir = os.path.join(exp_dir, "stimuli")
    os.makedirs(tmp_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 3. Hardware
    # ----------------------------------------------------------
    phaseTracker = utils.PhaseTracker()
    renderer = HaplscopeRender2D(
        fixation_distance    = fixationDistance,
        iod                  = float(info['IOD (mm)']),
        physical_calibration = physical_calibration,
        screen_config        = monitor_settings,
        debug_mode           = debug_mode,
    )
    kb         = keyboard.Keyboard(clock=core.Clock())
    trial_time = core.Clock()
    # change this to your Arduino port
    port = '/dev/cu.usbmodem101'   # Mac example
    # port = 'COM3'                # Windows example
    # port = '/dev/ttyACM0'       # Linux example

    arduino = serial.Serial(port, 9600)

    time.sleep(2)  # wait for Arduino reset

    print("Connected to Arduino")

       

   
    # ----------------------------------------------------------
    # 4. Calibration
    # ----------------------------------------------------------
    renderer.draw_physical_calibration()
    renderer.render_screen()
    kb.waitKeys(keyList=['return'], waitRelease=True)

    # ----------------------------------------------------------
    # PHASE 1 — VISUAL_ONLY
    # ----------------------------------------------------------
    trial_list = []
    for d in DEPTH_FACTORS:
        for rep in range(MOC_N_REPS[0]):
            trial_list.append({'half_height': HALF_HEIGHTS[0], 'depth_factor': d, 'texture': 2, 'repetition':rep})
    random.shuffle(trial_list)
    n_trials = len(trial_list)
    
    exp_v = ExperimentHandler(
        name='visual-only', version='1.0', extraInfo=info,
        runtimeInfo=None, dataFileName=data_v
    )
    
    for t_idx, trial in enumerate(trial_list):
        a = trial['half_height']
        df = trial['depth_factor']
        texture = trial['texture']
        rep = trial['repetition']
        seed = abs(hash((a, df, rep, info['Participant ID'], 'moc'))) % (2**32)
        t0 = time.time()
        img_arr = load_stimulus(texture, df, tmp_dir)
        l_arr = img_arr
        r_arr = img_arr
        t_render = time.time() - t0

        stimulus = render_to_stimulus(l_arr, r_arr, tmp_dir)

        try:
            rname, rt = run_single_trial(
                renderer, kb, phaseTracker, trial_time, parameters,
                stimulus, stimulusVisualAngle
            )
        except SystemExit:
            escaped = True
            break

        perceived = 'stretched' if rname == '3' else 'squashed'

        exp_v.addData('phase', 'visual-only')
        exp_v.addData('half_height', a)
        exp_v.addData('depth_factor', df)
        exp_v.addData('repetition', rep)
        exp_v.addData('response_key', rname)
        exp_v.addData('perceived', perceived)
        exp_v.addData('rt_s', rt)
        exp_v.addData('render_time_ms', round(t_render * 1000))
        exp_v.nextEntry()

        print(f"[MOC] Trial {t_idx+1:>3}/{n_trials} | "
              f"a={a*1000:.0f}mm  df={df:.2f}  resp={perceived}")

        # Rest break every 50 trials
        if (t_idx + 1) % 50 == 0 and (t_idx + 1) < n_trials:
            renderer.draw_text(
                f"Trial {t_idx+1} of {n_trials} complete.\n\n"
                "Take a short rest if needed.\n"
                "Press Enter to continue.",
                pos=(0, 0)
            )
            renderer.render_screen()
            kb.waitKeys(keyList=['return'], waitRelease=True)

    exp_v.saveAsWideText(data_v + '.csv')

    # ----------------------------------------------------------
    # PHASE 2 — HAPTIC_ONLY
    # ----------------------------------------------------------
    trial_list = []
    for d in DEPTH_FACTORS:
        for rep in range(MOC_N_REPS[0]):
            trial_list.append({'half_height': HALF_HEIGHTS[0], 'depth_factor': d,'repetition':rep})
    random.shuffle(trial_list)
    n_trilas = len(trial_list)
    

    exp_h = ExperimentHandler(
        name='haptic-only', version='1.0', extraInfo=info,
        runtimeInfo=None, dataFileName=data_h
    )
    

    
    for t_idx, trial in enumerate(trial_list):
        a = trial['half_height']
        df = trial['depth_factor']
        rep = trial['repetition']
        seed = abs(hash((a, df, rep, info['Participant ID'], 'moc'))) % (2**32)
        t0 = time.time()
        img_arr = load_empty_stimulus()
        # img_arr = load_stimulus(texture, df, tmp_dir)
        l_arr = img_arr
        r_arr = img_arr
        # l_arr, r_arr = generate_stimulus(a=a, df=df, iod=iod_m, seed=seed)
        t_render = time.time() - t0
        stimulus = render_to_stimulus(l_arr, r_arr, tmp_dir)
        renderer.draw_text_single_window(
            f"Next Shape: {SHAPE_DICT[df]}",
            pos=(0, 0)
        )
        renderer.render_screen()
        kb.waitKeys(keyList=['return'], waitRelease=True)
        try:
            rname, rt = run_single_trial(
                renderer, kb, phaseTracker, trial_time, parameters,
                stimulus, stimulusVisualAngle
            )
        except SystemExit:
            escaped = True
            break

        perceived = 'stretched' if rname == '3' else 'squashed'

        exp_h.addData('phase', 'haptic-only')
        exp_h.addData('half_height', a)
        exp_h.addData('depth_factor', df)
        exp_h.addData('repetition', rep)
        exp_h.addData('response_key', rname)
        exp_h.addData('perceived', perceived)
        exp_h.addData('rt_s', rt)
        exp_h.addData('render_time_ms', round(t_render * 1000))
        exp_h.nextEntry()

        print(f"[MOC] Trial {t_idx+1:>3}/{n_trials} | "
              f"a={a*1000:.0f}mm  df={df:.2f}  resp={perceived}")

        # Rest break every 50 trials
        if (t_idx + 1) % 50 == 0 and (t_idx + 1) < n_trials:
            renderer.draw_text(
                f"Trial {t_idx+1} of {n_trials} complete.\n\n"
                "Take a short rest if needed.\n"
                "Press Enter to continue.",
                pos=(0, 0)
            )
            renderer.render_screen()
            kb.waitKeys(keyList=['return'], waitRelease=True)
    exp_h.saveAsWideText(data_h + '.csv')
 
            
    # ----------------------------------------------------------
    # PHASE 3 — Visual + Haptic
    # ----------------------------------------------------------
    # TODO : add shape and vibration randomization
    trial_list = []
    for a in HALF_HEIGHTS:
        # for df in moc_depth_factors[a]:
        for df in DEPTH_FACTORS:
            for texture in TEXTURES:
                for rep in range(MOC_N_REPS[2]):
                    for vibration_level in VIBRATION_LEVELS:
                        trial_list.append({'half_height': a, 'depth_factor': df, 'texture': texture, 'repetition': rep, 'vibration_level': vibration_level})
    random.shuffle(trial_list)
    n_trials = len(trial_list)

    exp_vh = ExperimentHandler(
        name='visual-haptic', version='1.0', extraInfo=info,
        runtimeInfo=None, dataFileName=data_vh
    )
    for t_idx, trial in enumerate(trial_list):
        a   = trial['half_height']
        df  = trial['depth_factor']
        texture = trial['texture']
        rep = trial['repetition']
        vibration_level = trial['vibration_level']
        seed = abs(hash((a, df, rep, info['Participant ID'], 'moc'))) % (2**32)

        t0 = time.time()
        img_arr = load_stimulus(texture, df, tmp_dir)
        l_arr = img_arr
        r_arr = img_arr
        # l_arr, r_arr = generate_stimulus(a=a, df=df, iod=iod_m, seed=seed)
        t_render = time.time() - t0
        arduino.write(vibration_level)

        stimulus = render_to_stimulus(l_arr, r_arr, tmp_dir)

        try:
            rname, rt = run_single_trial(
                renderer, kb, phaseTracker, trial_time, parameters,
                stimulus, stimulusVisualAngle
            )
        except SystemExit:
            escaped = True
            break

        perceived = 'stretched' if rname == '3' else 'squashed'

        exp_vh.addData('phase', 'visual-haptic')
        exp_vh.addData('half_height', a)
        exp_vh.addData('depth_factor', df)
        exp_vh.addData('repetition', rep)
        exp_vh.addData('response_key', rname)
        exp_vh.addData('perceived', perceived)
        exp_vh.addData('rt_s', rt)
        exp_vh.addData('render_time_ms', round(t_render * 1000))
        exp_vh.nextEntry()

        print(f"[MOC] Trial {t_idx+1:>3}/{n_trials} | "
              f"a={a*1000:.0f}mm  df={df:.2f}  resp={perceived}")

        # Rest break every 50 trials
        if (t_idx + 1) % 50 == 0 and (t_idx + 1) < n_trials:
            renderer.draw_text(
                f"Trial {t_idx+1} of {n_trials} complete.\n\n"
                "Take a short rest if needed.\n"
                "Press Enter to continue.",
                pos=(0, 0)
            )
            renderer.render_screen()
            kb.waitKeys(keyList=['return'], waitRelease=True)
    exp_vh.saveAsWideText(data_vh + '.csv')

    # ----------------------------------------------------------
    # End
    # ----------------------------------------------------------
    if not escaped:
        renderer.draw_text("Experiment complete. Thank you!\n\nPress Enter to exit.", pos=(0, 0))
        renderer.render_screen()
        kb.waitKeys(keyList=['return'], waitRelease=True)

    renderer.close_windows()
    # shutil.rmtree(tmp_dir, ignore_errors=True)
    arduino.close()
    core.quit()


if __name__ == "__main__":
    main()
