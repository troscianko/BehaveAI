#!/usr/bin/env python3
import cv2
import numpy as np
import csv
import os
import time
import threading
import configparser
import sys
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import platform
import subprocess


# -------------------------
# Typical resolutions list
# -------------------------
RES_LIST = [
	(640, 480),
	(800, 600),
	(1024, 768),
	(1280, 720),
	(1280, 960),
	(1280, 1024),
	(1600, 1200),
	(1920, 1080),
	(2560, 1440),
	(3840, 2160)]


# --- NCNN helper utilities -----------------------

def ncnn_dir_for_weights(weights_path):
	base, ext = os.path.splitext(weights_path)
	return base + "_ncnn_model"

def ncnn_files_exist(ncnn_dir):
	if not os.path.isdir(ncnn_dir):
		return False
	has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
	has_bin   = any(f.endswith(".bin")   for f in os.listdir(ncnn_dir))
	return has_param and has_bin

def ensure_ncnn_export(weights_path, task, timeout=300):
	ncnn_dir = ncnn_dir_for_weights(weights_path)
	if ncnn_files_exist(ncnn_dir):
		return ncnn_dir
	try:
		print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
		model = YOLO(weights_path, task=task)
		model.export(format="ncnn")
		start = time.time()
		while time.time() - start < timeout:
			if ncnn_files_exist(ncnn_dir):
				print(f"NCNN export complete: {ncnn_dir}")
				return ncnn_dir
			time.sleep(0.5)
		print(f"NCNN export timeout for {weights_path}")
		return None
	except Exception as e:
		print(f"Warning: NCNN export failed for {weights_path}: {e}")
		return None

def load_model_with_ncnn_preference(weights_path, task):
	if not weights_path.endswith(".pt"):
		try:
			return YOLO(weights_path, task=task)
		except Exception as e:
			print(f"Error loading model {weights_path}: {e}")
			raise
	ncnn_dir = ncnn_dir_for_weights(weights_path)
	if ncnn_files_exist(ncnn_dir):
		try:
			print(f"Loading NCNN model from {ncnn_dir}")
			return YOLO(ncnn_dir, task=task)
		except Exception as e:
			print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")
	exported = ensure_ncnn_export(weights_path, task)
	if exported:
		try:
			return YOLO(exported, task=task)
		except Exception as e:
			print(f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)")
	print(f"Using original weights (PyTorch) at {weights_path}")
	return YOLO(weights_path, task=task)


# -------------------------
# Config loading
# -------------------------
def pick_ini_via_dialog():
	root = tk.Tk(); root.withdraw()
	p = filedialog.askopenfilename(title="Select BehaveAI settings INI", filetypes=[("INI files","*.ini"),("All files","*.*")])
	root.destroy()
	return p

if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	config_path = os.path.join(arg, "BehaveAI_settings.ini") if os.path.isdir(arg) else arg
else:
	config_path = pick_ini_via_dialog()
	if not config_path:
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
	tk.messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
	sys.exit(1)

project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
print(f"Working directory set to project dir: {project_dir}")
print(f"Using settings file: {config_path}")

config = configparser.ConfigParser()
config.optionxform = str
config.read(config_path)

def resolve_project_path(value, fallback):
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	return os.path.normpath(value) if os.path.isabs(value) else os.path.normpath(os.path.join(project_dir, value))

clips_dir_ini   = config['DEFAULT'].get('clips_dir',   'clips')
input_dir_ini   = config['DEFAULT'].get('input_dir',   'input')
output_dir_ini  = config['DEFAULT'].get('output_dir',  'output')
clips_dir       = resolve_project_path(clips_dir_ini,  'clips')
input_folder    = resolve_project_path(input_dir_ini,  'input')
output_folder   = resolve_project_path(output_dir_ini, 'output')

# ---------- Raspberry Pi detection & picamera2 handling ----------
def is_raspberry_pi():
	try:
		if platform.system() != "Linux":
			return False
		model_path = "/proc/device-tree/model"
		if os.path.exists(model_path):
			with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
				model = f.read().lower()
				if "raspberry pi" in model:
					return True
		if os.path.exists("/etc/os-release"):
			with open("/etc/os-release", "r", encoding="utf-8", errors="ignore") as f:
				txt = f.read().lower()
				if any(tok in txt for tok in ("raspbian", "raspberry", "raspberry pi os")):
					return True
	except Exception:
		pass
	return False

IS_RPI = is_raspberry_pi()
PICAMERA2_AVAILABLE = False
try:
	if IS_RPI:
		try:
			from picamera2 import Picamera2  # noqa: F401
			PICAMERA2_AVAILABLE = True
		except Exception as e:
			print("picamera2 import failed:", e)
			print("Attempting pip install of picamera2 (best-effort)...")
			try:
				subprocess.check_call([sys.executable, "-m", "pip", "install", "picamera2"])
				from picamera2 import Picamera2  # noqa: F401
				PICAMERA2_AVAILABLE = True
				print("picamera2 installed via pip and imported successfully.")
			except Exception as e2:
				print("Automatic picamera2 install failed:", e2)
				PICAMERA2_AVAILABLE = False
except Exception:
	PICAMERA2_AVAILABLE = False


# -------------------------
# Read parameters
# -------------------------
try:
	primary_motion_classes = [name.strip() for name in config['DEFAULT']['primary_motion_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('primary_motion_colors','').split(';') if c.strip()]
	primary_motion_colors = [tuple(map(int,c.split(',')))[::-1] for c in cols]
	primary_motion_hotkeys = [k.strip() for k in config['DEFAULT']['primary_motion_hotkeys'].split(',')]

	secondary_motion_classes = [name.strip() for name in config['DEFAULT']['secondary_motion_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('secondary_motion_colors','').split(';') if c.strip()]
	secondary_motion_colors = [tuple(map(int,c.split(',')))[::-1] for c in cols]
	secondary_motion_hotkeys = [k.strip() for k in config['DEFAULT']['secondary_motion_hotkeys'].split(',')]

	primary_static_classes = [name.strip() for name in config['DEFAULT']['primary_static_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('primary_static_colors','').split(';') if c.strip()]
	primary_static_colors = [tuple(map(int,c.split(',')))[::-1] for c in cols]
	primary_static_hotkeys = [k.strip() for k in config['DEFAULT']['primary_static_hotkeys'].split(',')]

	secondary_static_classes = [name.strip() for name in config['DEFAULT']['secondary_static_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('secondary_static_colors','').split(';') if c.strip()]
	secondary_static_colors = [tuple(map(int,c.split(',')))[::-1] for c in cols]
	secondary_static_hotkeys = [k.strip() for k in config['DEFAULT']['secondary_static_hotkeys'].split(',')]

	if len(secondary_motion_classes) >= 2 or len(secondary_static_classes) >= 2:
		hierarchical_mode = True
		motion_cropped_base_dir = 'annot_motion_crop'
		static_cropped_base_dir = 'annot_static_crop'
		if len(secondary_motion_classes) == 1:
			secondary_motion_classes = []; secondary_motion_colors = []; secondary_motion_hotkeys = []
		if len(secondary_static_classes) == 1:
			secondary_static_classes = []; secondary_static_colors = []; secondary_static_hotkeys = []
	else:
		hierarchical_mode = False

	primary_classes = primary_static_classes + primary_motion_classes
	primary_colors  = primary_static_colors  + primary_motion_colors
	primary_hotkeys = primary_static_hotkeys + primary_motion_hotkeys

	secondary_classes = secondary_static_classes + secondary_motion_classes
	secondary_colors  = secondary_static_colors  + secondary_motion_colors
	secondary_hotkeys = secondary_static_hotkeys + secondary_motion_hotkeys

	primary_static_model_path = os.path.join('model_primary_static', "train", "weights", "best.pt")
	primary_motion_model_path = os.path.join('model_primary_motion', "train", "weights", "best.pt")

	ignore_secondary  = [name.strip() for name in config['DEFAULT']['ignore_secondary'].split(',')]
	dominant_source   = config['DEFAULT']['dominant_source'].lower()

	scale_factor       = float(config['DEFAULT'].get('scale_factor','1.0'))
	use_ncnn           = config['DEFAULT']['use_ncnn'].lower()
	expA               = float(config['DEFAULT'].get('expA','0.5'))
	expB               = float(config['DEFAULT'].get('expB','0.8'))
	lum_weight         = float(config['DEFAULT'].get('lum_weight','0.7'))
	strategy           = config['DEFAULT'].get('strategy','exponential')
	chromatic_tail_only= config['DEFAULT']['chromatic_tail_only'].lower()
	rgb_multipliers    = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	primary_conf_thresh   = float(config['DEFAULT'].get('primary_conf_thresh','0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh','0.5'))
	match_distance_thresh = float(config['DEFAULT'].get('match_distance_thresh','200'))
	delete_after_missed   = float(config['DEFAULT'].get('delete_after_missed','5'))
	centroid_merge_thresh = float(config['DEFAULT'].get('centroid_merge_thresh','50'))
	iou_thresh            = float(config['DEFAULT'].get('iou_thresh','0.95'))
	line_thickness        = int(config['DEFAULT'].get('line_thickness','1'))
	font_size             = float(config['DEFAULT'].get('font_size','0.5'))
	frame_skip            = int(config['DEFAULT'].get('frame_skip','0'))

	process_noise_pos  = float(config['kalman'].get('process_noise_pos','0.01'))
	process_noise_vel  = float(config['kalman'].get('process_noise_vel','0.1'))
	measurement_noise  = float(config['kalman'].get('measurement_noise','0.1'))
	motion_threshold   = -1 * int(config['DEFAULT'].get('motion_threshold','0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")

if dominant_source not in ('motion','static','confidence'):
	raise ValueError("dominant_source must be motion, static, or confidence")

os.makedirs(clips_dir,     exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Live settings helpers
# -------------------------
LIVE_DEFAULTS = {
	'camera_index':            '0',
	'resolution':              '640x480',
	'throttle_fps':            '0.0',
	'display_mode':            'static',
	'classifier_enabled':      'true',
	'manual_recording':        'false',
	'detect_recording':        'false',
	'show_detections_in_rec':  'true',
	'schedule_enabled':        'false',
	'schedule_start_hh':       '00',
	'schedule_start_mm':       '00',
	'schedule_end_hh':         '23',
	'schedule_end_mm':         '59',
	'framesaver_enabled':      'false',
	'framesaver_interval_min': '1.0',
	'roi':                     '',
}

def load_live_settings():
	sec = config['live'] if 'live' in config else {}
	return {k: sec.get(k, v) for k, v in LIVE_DEFAULTS.items()}

def save_live_settings(settings_dict):
	if 'live' not in config:
		config.add_section('live')
	for k, v in settings_dict.items():
		config.set('live', k, str(v))
	try:
		with open(config_path, 'w') as f:
			config.write(f)
		print(f"Live settings saved to {config_path}")
	except Exception as e:
		print(f"Warning: could not save live settings: {e}")

# -------------------------
# Utilities
# -------------------------
def iou(box1, box2):
	xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
	xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
	inter = max(0, xb-xa) * max(0, yb-ya)
	area1 = (box1[2]-box1[0])*(box1[3]-box1[1]) if box1[2]>box1[0] and box1[3]>box1[1] else 0
	area2 = (box2[2]-box2[0])*(box2[3]-box2[1]) if box2[2]>box2[0] and box2[3]>box2[1] else 0
	prop1 = inter/area1 if area1>0 else 0
	prop2 = inter/area2 if area2>0 else 0
	return max(prop1, prop2)


# -------------------------
# Tracker
# -------------------------
class KalmanTracker:
	def __init__(self, dist_thresh, max_missed):
		self.next_id = 1
		self.tracks = {}
		self.prev_positions = {}
		self.dist_thresh = dist_thresh
		self.max_missed   = max_missed

	def _create_kf(self, initial_pt):
		kf = cv2.KalmanFilter(4,2)
		kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
		kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
		kf.processNoiseCov   = np.diag([process_noise_pos, process_noise_pos,
		                                process_noise_vel, process_noise_vel]).astype(np.float32)
		kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
		kf.statePre  = np.array([[initial_pt[0]],[initial_pt[1]],[0.],[0.]], dtype=np.float32)
		kf.statePost = kf.statePre.copy()
		return kf

	def predict_all(self):
		preds = []
		for tid, tr in self.tracks.items():
			pred = tr['kf'].predict()
			preds.append((tid, (float(pred[0,0]), float(pred[1,0]))))
		return preds

	def update(self, detections):
		preds      = self.predict_all()
		track_ids  = [t[0] for t in preds]
		pred_pts   = [t[1] for t in preds]

		if pred_pts and detections:
			cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
			for i, p in enumerate(pred_pts):
				for j, d in enumerate(detections):
					cost[i,j] = np.hypot(p[0]-d[0], p[1]-d[1])
			row_idx, col_idx = linear_sum_assignment(cost)
		else:
			row_idx = np.array([], dtype=int)
			col_idx = np.array([], dtype=int)
			cost    = np.array([])

		assigned_detects = {}; matched_tracks = set(); matched_dets = set()

		for r,c in zip(row_idx, col_idx):
			if cost[r,c] < self.dist_thresh:
				tid = track_ids[r]
				matched_tracks.add(tid); matched_dets.add(c); assigned_detects[c] = tid
				dpt  = detections[c]
				meas = np.array([[np.float32(dpt[0])],[np.float32(dpt[1])]])
				self.tracks[tid]['kf'].correct(meas)
				self.tracks[tid]['missed'] = 0
				self.prev_positions[tid]   = (dpt[0], dpt[1])

		for i, dpt in enumerate(detections):
			if i in matched_dets: continue
			best_tid, best_dist = None, float('inf')
			for tid,(px,py) in preds:
				d = np.hypot(dpt[0]-px, dpt[1]-py)
				if d < best_dist:
					best_dist, best_tid = d, tid
			if best_dist < self.dist_thresh:
				assigned_detects[i] = best_tid
				self.tracks[best_tid]['missed'] = 0
				meas = np.array([[np.float32(dpt[0])],[np.float32(dpt[1])]])
				self.tracks[best_tid]['kf'].correct(meas)
				self.prev_positions[best_tid] = (dpt[0], dpt[1])
				matched_tracks.add(best_tid)
			else:
				tid  = self.next_id
				kf   = self._create_kf(dpt)
				self.tracks[tid] = {'kf': kf, 'missed': 0}
				assigned_detects[i] = tid
				self.prev_positions[tid] = (dpt[0], dpt[1])
				matched_tracks.add(tid)
				self.next_id += 1

		for tid in list(self.tracks.keys()):
			if tid not in matched_tracks:
				self.tracks[tid]['missed'] += 1
				noise_scale = min(2.0, 1.0 + self.tracks[tid]['missed'] * 0.2)
				kf = self.tracks[tid]['kf']
				new_noise = kf.processNoiseCov.copy(); new_noise *= noise_scale
				kf.processNoiseCov = new_noise
				if self.tracks[tid]['missed'] > self.max_missed:
					del self.tracks[tid]
					if tid in self.prev_positions: del self.prev_positions[tid]

		return assigned_detects


# =========================================================
# CameraProcessor
# =========================================================
class CameraProcessor:
	def __init__(self):
		self.cap = None
		self.camera_index = 0
		self.thread   = None
		self.stop_event = threading.Event()
		self.running  = False
		self.classifier_enabled = True
		self.manual_recording   = False
		self.manual_writer      = None
		self.display_mode       = 'static'
		self.show_window_name   = "BehaveAI Live"
		self.frame_lock         = threading.Lock()
		self.frame_timestamps   = deque()
		self.latest_fps         = 0.0
		self.desired_fps        = 0.0
		self.desired_resolution = (640, 480)
		self.detection_recording_enabled = False
		self.detection_writer   = None
		self.detection_active   = False
		self.detection_last_seen= 0.0
		self.detection_stop_seconds = 3.0
		self.show_detections_in_recording = True

		self.display_scale = 0.5

		# ------- Scheduling -------
		self.schedule_enabled  = False
		self.schedule_start_s  = None
		self.schedule_end_s    = None

		# ------- Camera status (polled by GUI) -------
		self.camera_status = "stopped"

		# ------- Periodic frame saver (timelapse) -------
		self.frame_saver_enabled  = False
		self.frame_saver_interval = 1.0
		self._last_saved_frame_t  = 0.0
		self._timelapse_session_dir = None

		# Thread pool: 1 worker for async timelapse disk I/O
		self._save_executor = ThreadPoolExecutor(max_workers=1)

		# ------- ROI for YOLO -------
		self.roi = None

		# ROI selection mode state
		self._roi_mode   = False
		self._roi_done   = False
		self._roi_result = None
		self._roi_lock   = threading.Lock()

		# Tracker & models
		self.tracker = KalmanTracker(match_distance_thresh, delete_after_missed)
		self.model_static = None
		self.model_motion = None
		self.secondary_static_models = {}
		self.secondary_motion_models = {}

		self.prev_frames = None

		os.makedirs(clips_dir,     exist_ok=True)
		os.makedirs(output_folder, exist_ok=True)
		self._maybe_load_models()

	# ------------------------------------------------------------------
	# Setters (called from GUI thread)
	# ------------------------------------------------------------------

	def set_schedule(self, enabled: bool,
	                 start_h=0, start_m=0, start_s=0,
	                 end_h=0,   end_m=0,   end_s=0):
		with self.frame_lock:
			self.schedule_enabled  = bool(enabled)
			self.schedule_start_s  = int(start_h)*3600 + int(start_m)*60 + int(start_s)
			self.schedule_end_s    = int(end_h)*3600   + int(end_m)*60   + int(end_s)
		if enabled:
			if self.schedule_start_s > self.schedule_end_s:
				print(f"Schedule set (overnight): start {start_h:02d}:{start_m:02d}:{start_s:02d} "
				      f"end {end_h:02d}:{end_m:02d}:{end_s:02d} (next day)")
			else:
				print(f"Schedule set: start {start_h:02d}:{start_m:02d}:{start_s:02d} "
				      f"end {end_h:02d}:{end_m:02d}:{end_s:02d}")

	def set_frame_saver(self, enabled: bool, interval_minutes: float):
		with self.frame_lock:
			self.frame_saver_enabled  = bool(enabled)
			self.frame_saver_interval = max(0.0, float(interval_minutes))
			self._last_saved_frame_t  = 0.0
		if enabled:
			timelapse_dir = os.path.join(output_folder, 'timelapse')
			os.makedirs(timelapse_dir, exist_ok=True)
			print(f"Frame saver enabled: every {interval_minutes} min → {timelapse_dir}")

	def set_roi(self, roi):
		with self._roi_lock:
			self.roi = roi
		if roi:
			print(f"ROI set: {roi[0]},{roi[1]} → {roi[2]},{roi[3]}  "
			      f"({roi[2]-roi[0]}×{roi[3]-roi[1]} px)")
		else:
			print("ROI cleared — using full frame for YOLO")

	def begin_roi_selection(self):
		if not self.running:
			return False
		with self._roi_lock:
			self._roi_mode   = True
			self._roi_done   = False
			self._roi_result = None
		return True

	def roi_selection_done(self):
		with self._roi_lock:
			return self._roi_done

	def get_roi_result(self):
		with self._roi_lock:
			return self._roi_result

	def _run_roi_selection_in_loop(self, frozen_frame):
		WIN   = self.show_window_name
		H, W  = frozen_frame.shape[:2]
		CW, CH = 640, 480

		sel = {'drawing': False, 'pt1': None, 'pt2': None, 'roi': None, 'mode': None}

		hint1 = "ROI SELECT: drag=rect  click=640x480 crop"
		hint2 = "ENTER confirm  ESC cancel  R reset"

		def clamp(v, lo, hi):
			return max(lo, min(hi, v))

		def redraw():
			vis = frozen_frame.copy()
			overlay = vis.copy()
			cv2.rectangle(overlay, (0, 0), (W, 44), (0, 0, 0), -1)
			cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
			cv2.putText(vis, hint1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			cv2.putText(vis, hint2, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
			if sel['drawing'] and sel['pt1'] is not None and sel['pt2'] is not None:
				cv2.rectangle(vis, sel['pt1'], sel['pt2'], (0, 200, 255), 1)
			if sel['roi'] is not None:
				rx1, ry1, rx2, ry2 = sel['roi']
				cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
				lbl = (f"ROI {rx2-rx1}x{ry2-ry1}px "
				       f"({'centre crop' if sel['mode']=='center' else 'rect'})"
				       f"  — press ENTER")
				cv2.putText(vis, lbl, (rx1, max(ry1 - 6, 50)),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
			cv2.imshow(WIN, vis)

		def mouse_cb(event, x, y, flags, param):
			x = clamp(x, 0, W - 1); y = clamp(y, 0, H - 1)
			if event == cv2.EVENT_LBUTTONDOWN:
				sel['drawing'] = True; sel['pt1'] = (x, y); sel['pt2'] = (x, y)
				sel['roi'] = None; sel['mode'] = None; redraw()
			elif event == cv2.EVENT_MOUSEMOVE:
				if sel['drawing']: sel['pt2'] = (x, y); redraw()
			elif event == cv2.EVENT_LBUTTONUP:
				sel['drawing'] = False
				if sel['pt1'] is None: return
				x1r = min(sel['pt1'][0], x); x2r = max(sel['pt1'][0], x)
				y1r = min(sel['pt1'][1], y); y2r = max(sel['pt1'][1], y)
				if (x2r - x1r) < 8 and (y2r - y1r) < 8:
					cx, cy  = x, y
					x1n = clamp(cx - CW // 2, 0, W - CW) if W >= CW else 0
					y1n = clamp(cy - CH // 2, 0, H - CH) if H >= CH else 0
					x2n = min(x1n + CW, W); y2n = min(y1n + CH, H)
					sel['roi'] = (x1n, y1n, x2n, y2n); sel['mode'] = 'center'
				else:
					sel['roi'] = (x1r, y1r, x2r, y2r); sel['mode'] = 'rect'
				redraw()

		cv2.setMouseCallback(WIN, mouse_cb)
		redraw()

		result = None
		while True:
			k = cv2.waitKey(30) & 0xFF
			if k in (13, 10):   result = sel['roi']; break
			elif k == 27:       result = None;       break
			elif k in (ord('r'), ord('R')):
				sel['roi'] = None; sel['pt1'] = None; sel['pt2'] = None; sel['mode'] = None; redraw()

		cv2.setMouseCallback(WIN, lambda *a: None)

		with self._roi_lock:
			self._roi_result = result
			if result is not None:
				self.roi = result
			self._roi_mode = False
			self._roi_done = True

		if result:
			print(f"ROI confirmed: {result[0]},{result[1]} → {result[2]},{result[3]}  "
			      f"({result[2]-result[0]}×{result[3]-result[1]} px)")
		else:
			print("ROI selection cancelled")

	# ------------------------------------------------------------------
	# Existing setters
	# ------------------------------------------------------------------

	def set_display_mode(self, mode: str):
		if mode not in ('static', 'motion', 'disabled'):
			print("set_display_mode: invalid mode:", mode); return
		with self.frame_lock:
			self.display_mode = mode
		print(f"Display mode set to: {mode}")

	def set_show_detections_in_recording(self, flag: bool):
		with self.frame_lock:
			self.show_detections_in_recording = bool(flag)
		print(f"Show detections in recordings set to: {self.show_detections_in_recording}")

	def set_camera(self, index):
		with self.frame_lock:
			if self.running:
				print("set_camera: processor already running; index change ignored."); return
			self.camera_index = index if isinstance(index, str) else str(index)

	def set_resolution(self, wh_tuple):
		if isinstance(wh_tuple, str):
			try: w,h = map(int, wh_tuple.split('x'))
			except Exception: return
		else:
			w,h = wh_tuple
		self.desired_resolution = (int(w), int(h))
		if getattr(self,'cap',None) is not None and self.cap.isOpened():
			try:
				self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
				self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
				time.sleep(0.05)
				print(f"set_resolution: applied {w}x{h} to open camera")
			except Exception: pass
		else:
			print(f"set_resolution: desired set to {w}x{h}; will apply when camera opens")

	def toggle_classifier(self, enable: bool):
		self.classifier_enabled = bool(enable)
		if self.classifier_enabled and (self.model_static is None and self.model_motion is None):
			self._maybe_load_models()

	def toggle_manual_recording(self, enable: bool):
		if enable and not self.manual_recording:
			timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
			# Use the session's clips directory if already set, else fall back to
			# a date-stamped subfolder based on the current date.
			clips_target = getattr(self, '_session_clips_dir', None) \
			               or os.path.join(clips_dir, time.strftime("%Y-%m-%d"))
			os.makedirs(clips_target, exist_ok=True)
			path = os.path.join(clips_target, f"{timestamp}_manual.mp4")
			self.manual_writer_path = path
			self.manual_recording   = True
			print("Manual recording started:", path)
		elif not enable and self.manual_recording:
			self.manual_recording = False
			if self.manual_writer:
				self.manual_writer.release()
				self.manual_writer = None
			print("Manual recording stopped")

	def toggle_detection_recording(self, enable: bool):
		self.detection_recording_enabled = bool(enable)
		if not enable and self.detection_writer:
			self.detection_writer.release()
			self.detection_writer = None
			self.detection_active = False

	def start(self):
		if getattr(self,'thread',None) is not None and self.thread.is_alive():
			print("CameraProcessor.start: thread already running; ignored."); return
		self.stop_event.clear()
		self.thread  = threading.Thread(target=self._run_loop, daemon=True)
		self.running = True
		self.thread.start()

	def stop(self):
		self.stop_event.set()
		if getattr(self,'thread',None) is not None:
			try:
				if self.thread.is_alive():
					self.thread.join(timeout=5.0)
			except Exception: pass
		self.running = False
		for attr in ('manual_writer', 'detection_writer'):
			w = getattr(self, attr, None)
			if w:
				try: w.release()
				except: pass
				setattr(self, attr, None)
		if getattr(self,'cap',None):
			try: self.cap.release()
			except: pass
			self.cap = None
		if getattr(self, 'frame_saver_enabled', False) and self._timelapse_session_dir:
			sd = self._timelapse_session_dir
			self._timelapse_session_dir = None
			t = threading.Thread(
				target=self._compile_timelapse_video, args=(sd,), daemon=True)
			t.start()
			t.join(timeout=30)
		try: self._save_executor.shutdown(wait=False)
		except: pass
		self.camera_status = "stopped"
		self.thread = None

	# ------------------------------------------------------------------
	# Schedule helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _now_seconds():
		t = time.localtime()
		return t.tm_hour*3600 + t.tm_min*60 + t.tm_sec

	def _is_in_schedule_window(self):
		if not self.schedule_enabled:
			return True
		now_s = self._now_seconds()
		start = self.schedule_start_s
		end   = self.schedule_end_s
		if start > end:
			return now_s >= start or now_s < end
		return start <= now_s < end

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _maybe_load_models(self):
		try:
			if len(primary_static_classes) > 0 and os.path.exists(primary_static_model_path):
				if use_ncnn == 'true':
					self.model_static = load_model_with_ncnn_preference(primary_static_model_path, "detect")
					print("Loaded primary static NCNN model")
				else:
					self.model_static = YOLO(primary_static_model_path)
					print("Loaded primary static YOLO model")
			if len(primary_motion_classes) > 0 and os.path.exists(primary_motion_model_path):
				if use_ncnn == 'true':
					self.model_motion = load_model_with_ncnn_preference(primary_motion_model_path, "detect")
					print("Loaded primary motion NCNN model")
				else:
					self.model_motion = YOLO(primary_motion_model_path)
					print("Loaded primary motion YOLO model")
		except Exception as e:
			print("Primary model load failed:", e)
			self.model_static = None; self.model_motion = None

		if hierarchical_mode:
			for p in primary_classes:
				if p in ignore_secondary: continue
				try:
					model_dir  = f"model_secondary_static_{p}"
					model_path = os.path.join(model_dir, "train", "weights", "best.pt")
					if os.path.exists(model_path):
						if use_ncnn == 'true':
							self.secondary_static_models[p] = load_model_with_ncnn_preference(model_path, "classify")
							print(f"Loaded secondary static NCNN model for '{p}'")
						else:
							self.secondary_static_models[p] = YOLO(model_path)
							print(f"Loaded secondary static YOLO model for '{p}'")
				except Exception as e:
					print(f"Secondary static model load failed for {p}: {e}")
				try:
					model_dir  = f"model_secondary_motion_{p}"
					model_path = os.path.join(model_dir, "train", "weights", "best.pt")
					if os.path.exists(model_path):
						if use_ncnn == 'true':
							self.secondary_motion_models[p] = load_model_with_ncnn_preference(model_path, "classify")
							print(f"Loaded secondary motion NCNN model for '{p}'")
						else:
							self.secondary_static_models[p] = YOLO(model_path)
							print(f"Loaded secondary motion YOLO model for '{p}'")
				except Exception as e:
					print(f"Secondary motion model load failed for {p}: {e}")

	def _open_camera(self, timeout=5.0, retry_interval=0.2):
		deadline = time.time() + float(timeout)
		while not self.stop_event.is_set() and time.time() < deadline:
			try:
				if getattr(self, "cap", None):
					try: self.cap.release()
					except: pass
					self.cap = None

				if str(self.camera_index) == "picamera" and IS_RPI and PICAMERA2_AVAILABLE:
					try:
						res     = self.desired_resolution if getattr(self, "desired_resolution", None) else (640, 480)
						fps_req = self.desired_fps if getattr(self, "desired_fps", 0.0) else 30.0
						self.cap = Picamera2Wrapper(resolution=tuple(map(int, res)),
						                           fps=int(round(fps_req)), retries=4, retry_delay=0.5)
						self.cap.start()
						if self.cap.isOpened():
							print(f"_open_camera: picamera opened ({res[0]}x{res[1]} @ {fps_req}fps)")
							return True
					except Exception as e:
						print("_open_camera: picamera2 error:", e)
						try:
							if getattr(self, "cap", None):
								try: self.cap.release()
								except: pass
						except Exception: pass
						self.cap = None
				else:
					try: idx = int(self.camera_index)
					except Exception: idx = 0
					self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
					time.sleep(0.06)
					if self.cap is not None and self.cap.isOpened():
						w,h = self.desired_resolution
						try:
							self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
							self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
							time.sleep(0.05)
							print(f"_open_camera: camera opened ({w}x{h})")
						except Exception: pass
						return True
					else:
						try:
							if self.cap: self.cap.release()
						except: pass
						self.cap = None
			except Exception: pass
			time.sleep(retry_interval)
		print(f"_open_camera: failed to open camera {self.camera_index} within timeout")
		return False

	def _start_manual_writer(self, w, h, fps):
		if not hasattr(self,'manual_writer_path'): return
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.manual_writer = cv2.VideoWriter(self.manual_writer_path, fourcc, fps, (w,h))
		delattr(self, 'manual_writer_path')

	def _start_detection_writer(self, w, h, fps):
		timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
		# Use the session's output directory (set at session start) so that
		# detection clips triggered after midnight stay in the start-date folder.
		output_target = getattr(self, '_session_output_dir', output_folder)
		path = os.path.join(output_target, f"{timestamp}_detected.mp4")
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.detection_writer = cv2.VideoWriter(path, fourcc, fps, (w,h))
		self.detection_active = True
		print("Detection recording started:", path)

	# ------------------------------------------------------------------
	# Timelapse helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _write_timelapse_frame(path, frame):
		try:
			cv2.imwrite(path, frame)
		except Exception as e:
			print(f"Frame saver write error ({path}): {e}")

	def _compile_timelapse_video(self, session_dir):
		try:
			frames = sorted(
				f for f in os.listdir(session_dir) if f.lower().endswith('.jpg'))
			if not frames:
				print(f"Timelapse compile: no frames found in {session_dir}")
				return

			first = cv2.imread(os.path.join(session_dir, frames[0]))
			if first is None:
				print(f"Timelapse compile: could not read first frame in {session_dir}")
				return

			h, w     = first.shape[:2]
			out_path = os.path.join(session_dir, 'timelapse.mp4')
			fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
			writer   = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))

			written = 0
			for fname in frames:
				img = cv2.imread(os.path.join(session_dir, fname))
				if img is not None:
					writer.write(img)
					written += 1

			writer.release()
			print(f"Timelapse video saved: {out_path}  "
			      f"({written}/{len(frames)} frames @ 10 fps playback)")
		except Exception as e:
			print(f"Timelapse compile error for {session_dir}: {e}")

	# ------------------------------------------------------------------
	# Session close helper
	# ------------------------------------------------------------------

	def _close_session(self):
		if self.manual_writer:
			self.manual_writer.release()
			self.manual_writer = None
			print("Session: manual clip closed.")
		if self.detection_writer:
			self.detection_writer.release()
			self.detection_writer = None
			self.detection_active = False
			print("Session: detection clip closed.")
		if self.cap:
			try: self.cap.release()
			except: pass
			self.cap = None
			print("Session: camera released.")
		try:
			cv2.destroyWindow(self.show_window_name)
		except: pass

		if self.frame_saver_enabled and self._timelapse_session_dir:
			sd = self._timelapse_session_dir
			self._timelapse_session_dir = None
			self._save_executor.submit(self._compile_timelapse_video, sd)
			print(f"Session: queued timelapse compilation — {sd}")

		# Clear the per-session directory references
		self._session_output_dir = None
		self._session_clips_dir  = None

	# ------------------------------------------------------------------
	# Outer scheduling loop
	# ------------------------------------------------------------------

	def _run_loop(self):
		while not self.stop_event.is_set():

			if self.schedule_enabled and not self._is_in_schedule_window():
				self.camera_status = "waiting"
				now_s = self._now_seconds()
				start = self.schedule_start_s
				end   = self.schedule_end_s
				if start > end:
					wait_s = start - now_s
				else:
					wait_s = (start - now_s) if now_s < start else (86400 - now_s + start)
				wh, wm = divmod(abs(int(wait_s)), 3600)
				wm, ws = divmod(wm, 60)
				print(f"Scheduler: outside window — camera off.  "
				      f"Next window in ~{wh:02d}:{wm:02d}:{ws:02d}.")

				while not self.stop_event.is_set():
					time.sleep(1.0)
					if self._is_in_schedule_window():
						break
				if self.stop_event.is_set():
					break
				print("Scheduler: window opened — starting camera.")

			self.camera_status = "starting"
			self.tracker = KalmanTracker(match_distance_thresh, delete_after_missed)

			# ----------------------------------------------------------
			# Capture the session start date ONCE here.  All output files
			# (CSV, detection clips, manual clips, timelapse) for this
			# session will be stored under this date-stamped folder even
			# if the session runs past midnight.
			# ----------------------------------------------------------
			session_date_str = time.strftime("%Y-%m-%d")
			self._session_output_dir = os.path.join(output_folder, session_date_str)
			self._session_clips_dir  = os.path.join(clips_dir,     session_date_str)
			os.makedirs(self._session_output_dir, exist_ok=True)
			os.makedirs(self._session_clips_dir,  exist_ok=True)

			if self.frame_saver_enabled:
				timelapse_base = os.path.join(output_folder, 'timelapse')
				session_dir    = os.path.join(timelapse_base, session_date_str)
				os.makedirs(session_dir, exist_ok=True)
				self._timelapse_session_dir = session_dir
				self._last_saved_frame_t    = 0.0
				print(f"Timelapse session folder: {session_dir}")

			if not self._open_camera(timeout=15.0, retry_interval=0.5):
				print("Scheduler: failed to open camera; will retry.")
				self.camera_status = "waiting"
				time.sleep(10.0)
				if not self.schedule_enabled:
					break
				continue

			self.camera_status = "active"
			try:
				self._run_capture_loop()
			except Exception as e:
				print(f"CameraProcessor: unhandled exception in capture loop: {e}")
				import traceback; traceback.print_exc()

			self.camera_status = "stopping"
			self._close_session()

			if not self.schedule_enabled:
				break

			print("Scheduler: session complete — waiting for next window.")

		self.running       = False
		self.camera_status = "stopped"
		print("CameraProcessor: loop exited.")

	# ------------------------------------------------------------------
	# Inner per-frame capture loop
	# ------------------------------------------------------------------

	def _run_capture_loop(self):
		cap = self.cap
		w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  * scale_factor)
		h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
		fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
		if fps <= 0: fps = 20.0

		timestamp_session = time.strftime("%Y-%m-%d_%H-%M-%S")

		# Use the session output directory established at session start so that
		# the CSV (and any other outputs) land in the start-date folder even
		# when the session crosses midnight.
		session_output_dir = getattr(self, '_session_output_dir', output_folder)

		csv_path  = os.path.join(session_output_dir, f"{timestamp_session}_tracking.csv")
		csv_file  = open(csv_path, 'w', newline='')
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow([
			"frame","date","time","id","x","y",
			"primary_static_class","primary_static_conf",
			"primary_motion_class","primary_motion_conf",
			"secondary_static_class","secondary_static_conf",
			"secondary_motion_class","secondary_motion_conf"
		])

		prev_frames = None
		frame_idx   = 0
		frame_count = 0

		if self.schedule_enabled and self.schedule_start_s is not None:
			print(f"Session active: "
			      f"{self.schedule_start_s//3600:02d}:{(self.schedule_start_s%3600)//60:02d}:{self.schedule_start_s%60:02d}"
			      f" → "
			      f"{self.schedule_end_s//3600:02d}:{(self.schedule_end_s%3600)//60:02d}:{self.schedule_end_s%60:02d}")

		cv2.namedWindow(self.show_window_name,
		                cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)

		while not self.stop_event.is_set():

			if self.schedule_enabled and not self._is_in_schedule_window():
				print("Scheduler: window closed — ending session.")
				break

			frame_start_time = time.time()

			ret, raw_frame = cap.read()
			if not ret:
				time.sleep(0.01); continue
			frame_idx += 1

			if scale_factor != 1.0:
				raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)

			# ---- ROI selection mode ----
			with self._roi_lock:
				roi_mode_active = self._roi_mode
			if roi_mode_active:
				self._run_roi_selection_in_loop(raw_frame)
				continue

			# ---- Periodic frame saver (timelapse) ----
			if self.frame_saver_enabled and self.frame_saver_interval > 0:
				now_t      = time.time()
				interval_s = self.frame_saver_interval * 60.0
				if (now_t - self._last_saved_frame_t) >= interval_s:
					self._last_saved_frame_t = now_t
					if self._timelapse_session_dir is None:
						session_date_str = time.strftime("%Y-%m-%d")
						timelapse_base = os.path.join(output_folder, 'timelapse')
						session_dir    = os.path.join(timelapse_base, session_date_str)
						os.makedirs(session_dir, exist_ok=True)
						self._timelapse_session_dir = session_dir
					fname    = os.path.join(self._timelapse_session_dir,
					                        f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
					snapshot = raw_frame.copy()
					self._save_executor.submit(self._write_timelapse_frame, fname, snapshot)

			gray  = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
			frame = raw_frame.copy()

			# ---- ROI: extract sub-region for YOLO ----
			roi = self.roi
			if roi is not None:
				rx1, ry1, rx2, ry2 = roi
				fH, fW = raw_frame.shape[:2]
				rx1 = max(0, min(fW-1, rx1)); rx2 = max(0, min(fW,   rx2))
				ry1 = max(0, min(fH-1, ry1)); ry2 = max(0, min(fH,   ry2))
				if rx2 <= rx1 or ry2 <= ry1:
					roi = None

			if roi is not None:
				rx1, ry1, rx2, ry2 = roi
				yolo_frame = raw_frame[ry1:ry2, rx1:rx2]
				yolo_gray  = gray[ry1:ry2, rx1:rx2]
				roi_offset = (rx1, ry1)
			else:
				yolo_frame = raw_frame
				yolo_gray  = gray
				roi_offset = (0, 0)

			if prev_frames is not None and prev_frames[0].shape != yolo_gray.shape:
				prev_frames = None

			if prev_frames is None:
				prev_frames   = [yolo_gray.copy() for _ in range(3)]
				motion_image  = yolo_frame.copy()
				tnow = time.time(); self.frame_timestamps.append(tnow)
				while self.frame_timestamps and (tnow - self.frame_timestamps[0]) > 1.0:
					self.frame_timestamps.popleft()
				self.latest_fps = float(len(self.frame_timestamps))
				display_frame = frame.copy()
				if roi:
					cv2.rectangle(display_frame, (rx1,ry1), (rx2,ry2), (0,255,255), 1)
				if self.manual_recording or self.detection_active:
					self._draw_record_icon(display_frame, self.manual_recording, self.detection_active)
				cv2.imshow(self.show_window_name, display_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					self.stop_event.set()
				if self.desired_fps > 0.0:
					target  = 1.0 / self.desired_fps
					elapsed = time.time() - frame_start_time
					if elapsed < target: time.sleep(target - elapsed)
				continue

			# ---- Motion processing ----
			if primary_motion_classes[0] != '0' or secondary_motion_classes[0] != '0':
				diffs = [cv2.absdiff(prev_frames[j], yolo_gray) for j in range(3)]
				if strategy == 'exponential':
					prev_frames[0] = yolo_gray
					prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, yolo_gray, 1-expA, 0)
					prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, yolo_gray, 1-expB, 0)
				else:
					prev_frames[2] = prev_frames[1]; prev_frames[1] = prev_frames[0]
					prev_frames[0] = yolo_gray

				if chromatic_tail_only == 'true':
					tb = cv2.subtract(diffs[0], diffs[1])
					tr = cv2.subtract(diffs[2], diffs[1])
					tg = cv2.subtract(diffs[1], diffs[0])
					blue  = cv2.addWeighted(yolo_gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(yolo_gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
					red   = cv2.addWeighted(yolo_gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
				else:
					blue  = cv2.addWeighted(yolo_gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(yolo_gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
					red   = cv2.addWeighted(yolo_gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)

				motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
			else:
				motion_image = yolo_frame.copy()

			# ---- Primary detections ----
			all_detections = []
			if self.classifier_enabled:
				if primary_static_classes[0] != '0' and self.model_static is not None:
					try:
						results_static = self.model_static.predict(yolo_frame, conf=primary_conf_thresh, verbose=False)
						for box in results_static[0].boxes:
							x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
							cls_idx = int(box.cls[0]); conf = float(box.conf[0])
							class_name = primary_static_classes[cls_idx]
							all_detections.append({'coords':(x1,y1,x2,y2),'primary_class':class_name,
							                       'primary_conf':conf,'source':'static'})
					except Exception as e:
						print("Static model inference error:", e)

				if primary_motion_classes[0] != '0' and self.model_motion is not None:
					try:
						results_motion = self.model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
						for box in results_motion[0].boxes:
							x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
							cls_idx = int(box.cls[0]); conf = float(box.conf[0])
							class_name = primary_motion_classes[cls_idx]
							all_detections.append({'coords':(x1,y1,x2,y2),'primary_class':class_name,
							                       'primary_conf':conf,'source':'motion'})
					except Exception as e:
						print("Motion model inference error:", e)

			# ---- Merge detections ----
			merged_detections = []
			for det in all_detections:
				x1,y1,x2,y2 = det['coords']; cx,cy = (x1+x2)//2, (y1+y2)//2
				matched = False
				for md in merged_detections:
					md_cx, md_cy = md['centroid']
					dist    = np.hypot(cx-md_cx, cy-md_cy)
					md_x1,md_y1,md_x2,md_y2 = md['coords']
					overlap = iou((x1,y1,x2,y2),(md_x1,md_y1,md_x2,md_y2))
					ms_source = md.get('source','')
					if dist < centroid_merge_thresh or overlap > iou_thresh:
						if det['source'] == ms_source or dominant_source == 'confidence':
							if det['primary_conf'] > md.get('primary_conf',0):
								md['primary_class_combined'] = md.get('primary_class','')
								md['primary_conf_combined']  = md.get('primary_conf',0)
								md['primary_class'] = det['primary_class']
								md['primary_conf']  = det['primary_conf']
								md['coords'] = det['coords']; md['centroid'] = (cx,cy); md['source'] = det['source']
						elif det['source'] == 'static' and dominant_source == 'static':
							md['primary_class_combined'] = md.get('primary_class','')
							md['primary_conf_combined']  = md.get('primary_conf',0)
							md['primary_class'] = det['primary_class']
							md['primary_conf']  = det['primary_conf']
							md['coords'] = det['coords']; md['centroid'] = (cx,cy); md['source'] = det['source']
						elif det['source'] == 'motion' and dominant_source == 'motion':
							md['primary_class_combined'] = md.get('primary_class','')
							md['primary_conf_combined']  = md.get('primary_conf',0)
							md['primary_class'] = det['primary_class']
							md['primary_conf']  = det['primary_conf']
							md['coords'] = det['coords']; md['centroid'] = (cx,cy); md['source'] = det['source']
						matched = True; break
				if not matched:
					new_det = {'coords':det['coords'], 'centroid':(cx,cy), 'source':det['source'],
					           'primary_class':det['primary_class'], 'primary_conf':det['primary_conf'],
					           'primary_class_combined':'', 'primary_conf_combined':0.0}
					merged_detections.append(new_det)

			# ---- Secondary classification ----
			processed_detections = []
			for det in merged_detections:
				coords  = det['coords']; source = det['source']
				primary_class = det['primary_class']; primary_conf = det['primary_conf']
				if source == 'static':
					det['primary_static_class'] = primary_class; det['primary_static_conf'] = primary_conf
					det['primary_motion_class'] = det.get('primary_class_combined','')
					det['primary_motion_conf']  = det.get('primary_conf_combined',0.0)
				else:
					det['primary_motion_class'] = primary_class; det['primary_motion_conf'] = primary_conf
					det['primary_static_class'] = det.get('primary_class_combined','')
					det['primary_static_conf']  = det.get('primary_conf_combined',0.0)

				if hierarchical_mode:
					x1,y1,x2,y2 = coords
					sec_model = None; crop_img = None
					if source == 'static':
						if primary_class in self.secondary_static_models:
							sec_model = self.secondary_static_models.get(primary_class)
							crop_img  = yolo_frame
						elif primary_class in self.secondary_motion_models:
							sec_model = self.secondary_motion_models.get(primary_class)
							crop_img  = motion_image if primary_motion_classes[0] != '0' else yolo_frame
					else:
						if primary_class in self.secondary_motion_models:
							sec_model = self.secondary_motion_models.get(primary_class)
							crop_img  = motion_image
						elif primary_class in self.secondary_static_models:
							sec_model = self.secondary_static_models.get(primary_class)
							crop_img  = yolo_frame

					h_img, w_img = (crop_img.shape[:2] if crop_img is not None else (0,0))
					x1c = max(0, min(w_img-1, int(x1))); x2c = max(0, min(w_img-1, int(x2)))
					y1c = max(0, min(h_img-1, int(y1))); y2c = max(0, min(h_img-1, int(y2)))
					crop = None
					if crop_img is not None and x2c > x1c and y2c > y1c:
						crop = crop_img[y1c:y2c, x1c:x2c]

					secondary_class = primary_class; secondary_conf = 1.0
					if sec_model and crop is not None and crop.size > 0:
						try:
							sec_results = sec_model.predict(crop, verbose=False)
							if hasattr(sec_results[0], 'probs') and sec_results[0].probs is not None:
								idx2           = sec_results[0].probs.top1
								secondary_conf = sec_results[0].probs.top1conf.item()
								secondary_class= sec_model.names[idx2]
						except Exception: pass

					if source == 'static':
						det['secondary_static_class'] = secondary_class
						det['secondary_static_conf']  = secondary_conf
						det['secondary_motion_class'] = ''
						det['secondary_motion_conf']  = 0.0
					else:
						det['secondary_motion_class'] = secondary_class
						det['secondary_motion_conf']  = secondary_conf
						det['secondary_static_class'] = ''
						det['secondary_static_conf']  = 0.0

				processed_detections.append(det)

			# ---- Tracking ----
			ox, oy = roi_offset
			full_cents = [(d['centroid'][0]+ox, d['centroid'][1]+oy) for d in processed_detections]
			assignment = self.tracker.update(full_cents)

			# ---- Motion display canvas (full-frame, so annotations share coordinates) ----
			display_mode = self.display_mode
			motion_display = None
			if display_mode == 'motion':
				if roi is not None:
					motion_display = np.zeros_like(raw_frame)
					motion_display[ry1:ry2, rx1:rx2] = motion_image
				else:
					motion_display = motion_image.copy()
			draw_targets = [frame] if motion_display is None else [frame, motion_display]

			# ---- Draw annotations ----
			for idx_d, det in enumerate(processed_detections):
				tid = assignment.get(idx_d, None)
				if tid is None or tid not in self.tracker.tracks: continue

				x1r,y1r,x2r,y2r = det['coords']
				x1 = x1r+ox; y1 = y1r+oy; x2 = x2r+ox; y2 = y2r+oy
				cx,cy = det['centroid'][0]+ox, det['centroid'][1]+oy

				ps_class = det.get('primary_static_class','');  ps_conf = det.get('primary_static_conf',0)
				pm_class = det.get('primary_motion_class','');  pm_conf = det.get('primary_motion_conf',0)
				ss_class = det.get('secondary_static_class','');ss_conf = det.get('secondary_static_conf',0)
				sm_class = det.get('secondary_motion_class','');sm_conf = det.get('secondary_motion_conf',0)
				p_source = det.get('source','')

				primary_cls = ps_class if p_source=='static' else pm_class
				primary_col = primary_colors[primary_classes.index(primary_cls)] \
				              if primary_cls in primary_classes else (0,255,0)
				secondary_col = (255,255,255)

				for img in draw_targets:
					if hierarchical_mode:
						secondary_cls = ''
						if sm_class and sm_class != primary_cls:
							secondary_cls = sm_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
						if ss_class and ss_class != primary_cls:
							secondary_cls = ss_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
						if primary_cls in ignore_secondary:
							label = f"{tid} {primary_cls.upper()}"
							ls,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
							lw,lh = ls
							cv2.rectangle(img, (x1-line_thickness, y1-lh-line_thickness*4),
							              (x1+lw+line_thickness*2, y1), (0,0,0), -1)
							cv2.rectangle(img, (x1,y1), (x2,y2), primary_col, line_thickness)
							cv2.putText(img, label, (x1, y1-line_thickness*2),
							            cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_col, line_thickness, cv2.LINE_AA)
						else:
							outer_t = line_thickness+2
							cv2.rectangle(img, (x1-outer_t,y1-outer_t), (x2+outer_t,y2+outer_t),
							              primary_col, outer_t)
							label = f"{tid} {primary_cls.upper()} {secondary_cls}"
							ls,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
							lw,lh = ls
							cv2.rectangle(img, (x1-line_thickness, y1-lh-line_thickness*4),
							              (x1+lw+line_thickness*2, y1), (0,0,0), -1)
							cv2.rectangle(img, (x1,y1), (x2,y2), secondary_col, line_thickness)
							cv2.putText(img, label, (x1, y1-line_thickness*2),
							            cv2.FONT_HERSHEY_SIMPLEX, font_size, secondary_col, line_thickness, cv2.LINE_AA)
					else:
						label = f"{tid} {primary_cls}"
						ls,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
						lw,lh = ls
						cv2.rectangle(img, (x1-line_thickness, y1-lh-line_thickness*4),
						              (x1+lw+line_thickness*2, y1), (0,0,0), -1)
						cv2.rectangle(img, (x1,y1), (x2,y2), primary_col, line_thickness)
						cv2.putText(img, label, (x1, y1-line_thickness*3),
						            cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_col, line_thickness, cv2.LINE_AA)

					if tid in self.tracker.tracks:
						sp = self.tracker.tracks[tid]['kf'].statePost
						sx,sy = int(sp[0,0]), int(sp[1,0])
						vx,vy = float(sp[2,0]), float(sp[3,0])
						nx,ny = int(sx+vx), int(sy+vy)
						light_color = tuple(int(0.8*ch+0.2*255) for ch in primary_col)
						cv2.line(img, (sx,sy), (nx,ny), primary_col, line_thickness)
						cv2.circle(img, (nx,ny), 3, light_color, -line_thickness)
						cv2.circle(img, (int(cx),int(cy)), 3, primary_col, -line_thickness)

				if self.detection_recording_enabled:
					now = time.localtime()
					date_str = time.strftime("%Y-%m-%d", now)
					time_str = time.strftime("%H:%M:%S", now)
					csv_writer.writerow([
						frame_idx, date_str, time_str, tid, cx, cy,
						ps_class, f"{ps_conf:.3f}",
						pm_class, f"{pm_conf:.3f}",
						ss_class, f"{ss_conf:.3f}",
						sm_class, f"{sm_conf:.3f}"
					])

			# ROI boundary
			if roi is not None:
				rx1,ry1,rx2,ry2 = roi
				for img in draw_targets:
					cv2.rectangle(img, (rx1,ry1), (rx2,ry2), (0,255,255), 1)
					cv2.putText(img, "ROI", (rx1+2, ry1+14),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

			# Frame counter
			text_color = (255,255,255)
			label = str(frame_idx)
			ls,_  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			lw,lh = ls
			for img in draw_targets:
				cv2.rectangle(img, (0,0), (lw+line_thickness*4, lh+line_thickness*4), (0,0,0), -1)
				cv2.putText(img, label, (line_thickness*2, lh+line_thickness*2),
				            cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, line_thickness)

			# ---- Display frame ----
			if display_mode == 'static':
				display_frame = frame
			elif display_mode == 'motion':
				display_frame = motion_display
			else:
				display_frame = np.zeros_like(frame)

			record_frame = frame if self.show_detections_in_recording else raw_frame

			# ---- Writers ----
			if self.manual_recording and self.manual_writer is None:
				self._start_manual_writer(w, h, fps)
			if self.manual_writer is not None:
				try: self.manual_writer.write(record_frame)
				except Exception as e: print("Manual writer error:", e)

			has_detections = len(processed_detections) > 0
			nowt = time.time()
			if has_detections:
				self.detection_last_seen = nowt
				if self.detection_recording_enabled and not self.detection_active:
					self._start_detection_writer(w, h, fps)
			else:
				if self.detection_active and (nowt - self.detection_last_seen) > self.detection_stop_seconds:
					if self.detection_writer:
						self.detection_writer.release(); self.detection_writer = None
					self.detection_active = False

			if self.detection_writer is not None:
				try: self.detection_writer.write(record_frame)
				except Exception as e: print("Detection writer error:", e)

			if self.manual_recording or self.detection_active:
				self._draw_record_icon(display_frame, self.manual_recording, self.detection_active)

			cv2.imshow(self.show_window_name, display_frame)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('q'):
				self.stop_event.set()
				break

			tnow = time.time(); self.frame_timestamps.append(tnow)
			while self.frame_timestamps and (tnow - self.frame_timestamps[0]) > 1.0:
				self.frame_timestamps.popleft()
			self.latest_fps = float(len(self.frame_timestamps))

			if self.desired_fps and self.desired_fps > 0.0:
				target_period = 1.0 / float(self.desired_fps)
				elapsed = time.time() - frame_start_time
				if elapsed < target_period: time.sleep(target_period - elapsed)

			frame_count += 1
			if frame_count > frame_skip: frame_count = 0

		csv_file.close()
		print(f"Session CSV saved: {csv_path}")

	def _draw_record_icon(self, img, manual_active: bool, detection_active: bool):
		h_i,w_i = img.shape[:2]; margin = 8; dot_radius = 8; label = "REC"
		if   manual_active and detection_active: label = "REC M+D"
		elif manual_active:                      label = "REC M"
		elif detection_active:                   label = "REC D"
		font  = cv2.FONT_HERSHEY_SIMPLEX; fscale = 0.5; fth = 1
		(text_w,text_h),_ = cv2.getTextSize(label, font, fscale, fth)
		rect_w = dot_radius*2+6+text_w+6; rect_h = max(text_h+6, dot_radius*2+6)
		x1 = w_i-margin-rect_w; y1 = margin; x2 = w_i-margin; y2 = y1+rect_h
		overlay = img.copy(); cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,0,0),-1)
		alpha = 0.45; cv2.addWeighted(overlay,alpha,img,1-alpha,0,img)
		dot_cx = x1+6+dot_radius; dot_cy = y1+rect_h//2
		cv2.circle(img,(dot_cx,dot_cy),dot_radius,(0,0,255),-1)
		cv2.circle(img,(dot_cx+1,dot_cy-1),max(1,dot_radius//3),(255,255,255),-1)
		text_x = dot_cx+dot_radius+6; text_y = dot_cy+text_h//2-2
		cv2.putText(img,label,(text_x,text_y),font,fscale,(200,200,200),fth,cv2.LINE_AA)


# =========================================================
# Picamera2Wrapper
# =========================================================
class Picamera2Wrapper:
	def __init__(self, resolution=(640,480), fps=30, retries=3, retry_delay=0.5):
		from picamera2 import Picamera2
		self.Picamera2   = Picamera2
		self.resolution  = tuple(map(int, resolution))
		self.target_fps  = float(fps) if fps and fps > 0 else 30.0
		self.retries     = int(retries)
		self.retry_delay = float(retry_delay)
		self.picam       = None
		self._started    = False
		self._width      = float(self.resolution[0])
		self._height     = float(self.resolution[1])
		self._fps        = float(self.target_fps)
		try:
			self.picam = self.Picamera2()
		except Exception as e:
			raise RuntimeError(f"Picamera2 construction failed: {e}") from e

	def configure_and_start(self):
		last_exc = None
		for attempt in range(1, self.retries+1):
			try:
				cfg = self.picam.create_video_configuration(
					main={"size":(int(self._width),int(self._height)),"format":"BGR888"},
					controls={"FrameRate":int(round(self._fps))}
				)
				self.picam.configure(cfg); self.picam.start(); self._started = True
				import time; time.sleep(0.05)
				for _ in range(2):
					try: _ = self.picam.capture_array()
					except Exception: time.sleep(0.02)
				return True
			except Exception as e:
				last_exc = e
				try:
					if self._started:
						try: self.picam.stop()
						except Exception: pass
				except Exception: pass
				try: self.picam = self.Picamera2()
				except Exception:
					last_exc = RuntimeError(f"Failed to recreate Picamera2 after attempt {attempt}: {e}")
					break
				import time; time.sleep(self.retry_delay)
		raise RuntimeError(f"Picamera2 configure/start failed after {self.retries} attempts: {last_exc}") from last_exc

	def start(self):
		if self._started: return True
		return self.configure_and_start()

	def isOpened(self): return bool(self._started)

	def read(self):
		if not self._started: return False, None
		try:
			arr   = self.picam.capture_array()
			frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
			return True, frame
		except Exception: return False, None

	def release(self):
		try:
			if self._started:
				try: self.picam.stop()
				except Exception: pass
			try:
				if hasattr(self.picam,"close"): self.picam.close()
			except Exception: pass
		except Exception: pass
		self._started = False

	def get(self, prop):
		if prop == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._width)
		if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._height)
		if prop == cv2.CAP_PROP_FPS:          return float(self._fps)
		return 0.0

	def set(self, prop, value):
		try:
			if   prop == cv2.CAP_PROP_FRAME_WIDTH:  self._width  = float(value)
			elif prop == cv2.CAP_PROP_FRAME_HEIGHT: self._height = float(value)
			elif prop == cv2.CAP_PROP_FPS:          self._fps    = float(value)
			else: return False
			if self._started:
				try:
					try: self.picam.stop()
					except Exception: pass
					cfg = self.picam.create_video_configuration(
						main={"size":(int(self._width),int(self._height)),"format":"RGB888"},
						controls={"FrameRate":int(round(self._fps))}
					)
					self.picam.configure(cfg)
					import time; time.sleep(0.05)
					self.picam.start()
					try: _ = self.picam.capture_array()
					except Exception: time.sleep(0.02)
				except Exception:
					self._started = False; return False
			return True
		except Exception: return False


# =========================================================
# Camera enumeration
# =========================================================
def scan_cameras(max_search=6):
	available = []
	for idx in range(max_search):
		try:
			cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
			if cap is None or not cap.isOpened():
				try: cap.release()
				except: pass
				continue
			ret, _ = cap.read()
			try: cap.release()
			except: pass
			if ret: available.append(str(idx))
		except Exception: continue
	if IS_RPI and PICAMERA2_AVAILABLE:
		available.append("picamera")
	return available if available else ["0"]


# =========================================================
# GUI
# =========================================================
class ControlGUI:
	_STATUS_DISPLAY = {
		"stopped":  ("Camera stopped",               "gray"),
		"waiting":  ("⏳  Waiting for schedule window…", "#c87000"),
		"starting": ("Starting camera…",             "blue"),
		"active":   ("🟢  Camera active",             "green"),
		"stopping": ("Stopping…",                    "gray"),
	}

	def __init__(self, root, processor: CameraProcessor):
		self.root      = root
		self.processor = processor
		root.title("BehaveAI - Live Camera Controls")

		frm = ttk.Frame(root, padding=8)
		frm.grid(sticky="nsew")

		# Load saved live settings (falls back to defaults if section absent)
		ls = load_live_settings()

		# ---- Row 0: camera & resolution ----
		ttk.Label(frm, text="Camera:").grid(row=0, column=0, sticky="w")
		cams = scan_cameras(8)
		self.cam_combo = ttk.Combobox(frm, values=cams, state="readonly", width=8)
		saved_cam = ls.get('camera_index', '0')
		self.cam_combo.set(saved_cam if saved_cam in cams else cams[0])
		self.cam_combo.grid(row=0, column=1, sticky="w")

		ttk.Label(frm, text="Resolution:").grid(row=0, column=2, sticky="w", padx=(12,0))
		res_strs = [f"{w}x{h}" for (w,h) in RES_LIST]
		if "640x480" in res_strs: res_strs.remove("640x480"); res_strs.insert(0,"640x480")
		self.res_combo = ttk.Combobox(frm, values=res_strs, state="readonly", width=12)
		saved_res = ls.get('resolution', '640x480')
		self.res_combo.set(saved_res if saved_res in res_strs else "640x480")
		self.res_combo.grid(row=0, column=3, sticky="w")

		self.start_btn = ttk.Button(frm, text="Start Camera", command=self._start_camera)
		self.start_btn.grid(row=0, column=4, padx=8)

		# ---- Row 1: manual recording & classifier ----
		self.manual_rec_var = tk.BooleanVar(value=ls.get('manual_recording','false').lower()=='true')
		self.manual_btn = ttk.Checkbutton(frm, text="Manual recording (clips)",
		                                  variable=self.manual_rec_var, command=self._toggle_manual)
		self.manual_btn.grid(row=1, column=0, columnspan=2, sticky="w")

		self.classifier_var = tk.BooleanVar(value=ls.get('classifier_enabled','true').lower()=='true')
		self.classifier_chk = ttk.Checkbutton(frm, text="Enable classifier",
		                                      variable=self.classifier_var,
		                                      command=self._toggle_classifier)
		self.classifier_chk.grid(row=1, column=2, sticky="w")

		# ---- Row 2: detection recording & show-detections toggle ----
		self.detect_rec_var = tk.BooleanVar(value=ls.get('detect_recording','false').lower()=='true')
		self.detect_chk = ttk.Checkbutton(frm, text="Record on detection (output)",
		                                  variable=self.detect_rec_var,
		                                  command=self._toggle_detect_recording)
		self.detect_chk.grid(row=2, column=0, columnspan=2, sticky="w")

		self.show_det_var = tk.BooleanVar(value=ls.get('show_detections_in_rec','true').lower()=='true')
		self.show_det_chk = ttk.Checkbutton(frm, text="Show detections in recordings",
		                                    variable=self.show_det_var,
		                                    command=self._toggle_show_detections)
		self.show_det_chk.grid(row=2, column=2, columnspan=2, sticky="w")

		# ---- Row 3: display stream ----
		ttk.Label(frm, text="Display stream:").grid(row=3, column=0, sticky="w")
		self.display_var = tk.StringVar(value=ls.get('display_mode','static'))
		ttk.Radiobutton(frm, text="Static",   variable=self.display_var, value='static',
		                command=self._set_display).grid(row=3, column=1, sticky="w")
		ttk.Radiobutton(frm, text="Motion",   variable=self.display_var, value='motion',
		                command=self._set_display).grid(row=3, column=2, sticky="w")
		ttk.Radiobutton(frm, text="Disabled", variable=self.display_var, value='disabled',
		                command=self._set_display).grid(row=3, column=3, sticky="w")

		# ---- Row 4-5: FPS display & throttle ----
		ttk.Label(frm, text="FPS:").grid(row=4, column=0, sticky="w")
		self.fps_label = ttk.Label(frm, text="—")
		self.fps_label.grid(row=4, column=1, sticky="w")

		ttk.Label(frm, text="Throttle FPS (0=off):").grid(row=5, column=0, sticky="w")
		try:   _saved_fps = float(ls.get('throttle_fps', '0.0'))
		except: _saved_fps = 0.0
		self.throttle_var  = tk.DoubleVar(value=_saved_fps)
		self.throttle_spin = tk.Spinbox(frm, from_=0, to=120, increment=1,
		                                textvariable=self.throttle_var, width=6,
		                                command=self._on_throttle_change)
		self.throttle_spin.grid(row=5, column=1, sticky="w")
		self.throttle_spin.bind("<Return>", lambda e: self._on_throttle_change())

		# ============================================================
		# Scheduling panel
		# ============================================================
		ttk.Separator(frm, orient='horizontal').grid(
			row=6, column=0, columnspan=6, sticky="ew", pady=(6,2))

		self.schedule_var = tk.BooleanVar(value=ls.get('schedule_enabled','false').lower()=='true')
		ttk.Checkbutton(frm, text="Enable schedule  (camera ON only during window)",
		                variable=self.schedule_var,
		                command=self._on_schedule_toggle).grid(row=7, column=0, columnspan=2, sticky="w")

		ttk.Label(frm, text="Start HH:").grid(row=7, column=2, sticky="e")
		self.sched_sh = tk.StringVar(value=ls.get('schedule_start_hh','00'))
		ttk.Entry(frm, textvariable=self.sched_sh, width=3).grid(row=7, column=3, sticky="w", padx=(0,2))
		ttk.Label(frm, text="MM:").grid(row=7, column=3, sticky="e")
		self.sched_sm = tk.StringVar(value=ls.get('schedule_start_mm','00'))
		ttk.Entry(frm, textvariable=self.sched_sm, width=3).grid(row=7, column=4, sticky="w", padx=(0,2))

		ttk.Label(frm, text="End   HH:").grid(row=8, column=2, sticky="e")
		self.sched_eh = tk.StringVar(value=ls.get('schedule_end_hh','23'))
		ttk.Entry(frm, textvariable=self.sched_eh, width=3).grid(row=8, column=3, sticky="w", padx=(0,2))
		ttk.Label(frm, text="MM:").grid(row=8, column=3, sticky="e")
		self.sched_em = tk.StringVar(value=ls.get('schedule_end_mm','59'))
		ttk.Entry(frm, textvariable=self.sched_em, width=3).grid(row=8, column=4, sticky="w", padx=(0,2))

		ttk.Label(frm, text="(24-hour clock — overnight windows supported)",
		          foreground="gray").grid(row=9, column=0, columnspan=5, sticky="w")

		# ============================================================
		# Frame saver panel
		# ============================================================
		ttk.Separator(frm, orient='horizontal').grid(
			row=10, column=0, columnspan=6, sticky="ew", pady=(6,2))

		self.framesaver_var = tk.BooleanVar(value=ls.get('framesaver_enabled','false').lower()=='true')
		ttk.Checkbutton(frm, text="Save frame every N minutes (timelapse)",
		                variable=self.framesaver_var,
		                command=self._on_framesaver_toggle).grid(row=11, column=0, columnspan=2, sticky="w")

		ttk.Label(frm, text="Interval (min):").grid(row=11, column=2, sticky="e", padx=(8,2))
		self.framesaver_interval_var = tk.StringVar(value=ls.get('framesaver_interval_min','1.0'))
		ttk.Entry(frm, textvariable=self.framesaver_interval_var, width=6).grid(row=11, column=3, sticky="w")
		ttk.Label(frm, text="Full-res • no annotations • MP4 compiled at session end",
		          foreground="gray").grid(row=12, column=0, columnspan=5, sticky="w")

		# ============================================================
		# ROI panel
		# ============================================================
		ttk.Separator(frm, orient='horizontal').grid(
			row=13, column=0, columnspan=6, sticky="ew", pady=(6,2))

		ttk.Label(frm, text="YOLO Region of Interest:").grid(row=14, column=0, columnspan=2, sticky="w")
		ttk.Button(frm, text="Set ROI from camera view",
		           command=self._open_roi_selector).grid(row=14, column=2, columnspan=2,
		                                                 sticky="w", padx=(4,0))
		ttk.Button(frm, text="Clear ROI (full frame)",
		           command=self._clear_roi).grid(row=14, column=4, sticky="w", padx=(4,0))

		self.roi_label = ttk.Label(frm, text="ROI: full frame", foreground="gray")
		self.roi_label.grid(row=15, column=0, columnspan=5, sticky="w")
		ttk.Label(frm, text="Drag = arbitrary rect  |  Click = 640×480 centred crop",
		          foreground="gray").grid(row=16, column=0, columnspan=5, sticky="w")

		# Restore saved ROI into processor immediately (before camera opens)
		_saved_roi_str = ls.get('roi', '')
		if _saved_roi_str:
			try:
				_rx1, _ry1, _rx2, _ry2 = map(int, _saved_roi_str.split(','))
				self.processor.set_roi((_rx1, _ry1, _rx2, _ry2))
				self.roi_label.config(
					text=(f"ROI: ({_rx1},{_ry1}) → ({_rx2},{_ry2})  "
					      f"{_rx2-_rx1}×{_ry2-_ry1} px"),
					foreground="green")
			except Exception:
				pass

		# ============================================================
		# Status bar
		# ============================================================
		ttk.Separator(frm, orient='horizontal').grid(
			row=17, column=0, columnspan=6, sticky="ew", pady=(6,2))

		self.status_label = ttk.Label(frm, text="Camera stopped", foreground="gray",
		                              font=("TkDefaultFont", 9, "bold"))
		self.status_label.grid(row=18, column=0, columnspan=5, sticky="w")

		# Save settings button persists current GUI state to the [live] INI section
		self.save_settings_btn = ttk.Button(frm, text="💾 Save settings",
		                                    command=self._save_settings)
		self.save_settings_btn.grid(row=18, column=3, sticky="e", padx=(0,4), pady=(4,0))

		self.quit_btn = ttk.Button(frm, text="Quit", command=self._quit)
		self.quit_btn.grid(row=18, column=4, sticky="e", pady=(4,0))

		# ============================================================
		# Apply all saved settings to the processor NOW, so that
		# settings restored from the INI actually take effect without
		# requiring the user to manually click each control.
		# ============================================================
		self._apply_saved_settings(ls)

		self._update_gui()

	# ------------------------------------------------------------------
	# Apply saved settings to the processor on startup
	# ------------------------------------------------------------------
	def _apply_saved_settings(self, ls):
		"""Push every saved live setting into the processor immediately.

		The tk variables (BooleanVar, StringVar, etc.) are already
		pre-populated from `ls` above, but the processor attributes are
		only updated when a callback fires.  This method calls each
		callback-equivalent directly so the processor state matches the
		GUI state from the moment the window opens.
		"""
		# Classifier
		self.processor.toggle_classifier(self.classifier_var.get())

		# Detection recording
		self.processor.toggle_detection_recording(self.detect_rec_var.get())

		# Show detections in recordings
		self.processor.set_show_detections_in_recording(self.show_det_var.get())

		# Display mode (static / motion / disabled)
		self.processor.set_display_mode(self.display_var.get())

		# FPS throttle
		try:
			fps_val = float(self.throttle_var.get())
		except Exception:
			fps_val = 0.0
		self.processor.desired_fps = max(0.0, fps_val)

		# Schedule — reuse the existing validated callback
		self._on_schedule_toggle()

		# Frame saver — reuse the existing validated callback
		self._on_framesaver_toggle()

		# Manual recording: only restore if it was active when last saved.
		# We intentionally do NOT auto-start a new recording file here
		# (the camera isn't open yet), but we do reset the var to False
		# so the checkbox reflects a clean "not recording" state on launch,
		# which is the safe default.  A user who wants manual recording must
		# start the camera first and then tick the box.
		if self.manual_rec_var.get():
			self.manual_rec_var.set(False)

	# ------------------------------------------------------------------
	# Camera start
	# ------------------------------------------------------------------
	def _start_camera(self):
		cam_val = self.cam_combo.get()
		try:
			cam_index = cam_val if cam_val == "picamera" else int(cam_val)
		except Exception:
			cam_index = cam_val
		self.processor.set_camera(cam_index)
		try:
			w,h = map(int, self.res_combo.get().split('x'))
			self.processor.set_resolution((w,h))
		except Exception:
			pass
		self._on_schedule_toggle()
		self._on_framesaver_toggle()
		self.processor.start()

		if self.processor.schedule_enabled:
			self.start_btn.config(text="Scheduler running", state="disabled")
		else:
			self.start_btn.config(text="Camera started", state="disabled")
		self.cam_combo.config(state="disabled")
		self.res_combo.config(state="disabled")

	# ------------------------------------------------------------------
	# Existing callbacks
	# ------------------------------------------------------------------
	def _toggle_manual(self):           self.processor.toggle_manual_recording(self.manual_rec_var.get())
	def _toggle_classifier(self):       self.processor.toggle_classifier(self.classifier_var.get())
	def _toggle_detect_recording(self): self.processor.toggle_detection_recording(self.detect_rec_var.get())
	def _toggle_show_detections(self):  self.processor.set_show_detections_in_recording(self.show_det_var.get())
	def _set_display(self):             self.processor.set_display_mode(self.display_var.get())

	def _on_throttle_change(self):
		try:   fps_val = float(self.throttle_var.get())
		except: fps_val = 0.0
		if fps_val < 0: fps_val = 0.0
		self.processor.desired_fps = fps_val

	# ------------------------------------------------------------------
	# Schedule callbacks
	# ------------------------------------------------------------------
	def _on_schedule_toggle(self):
		enabled = self.schedule_var.get()
		try:
			sh = int(self.sched_sh.get()); sm = int(self.sched_sm.get())
			eh = int(self.sched_eh.get()); em = int(self.sched_em.get())
		except ValueError:
			messagebox.showerror("Schedule", "Please enter valid integer values for all time fields.")
			self.schedule_var.set(False); return
		for name,(v,lo,hi) in [("Start hour",(sh,0,23)),("Start min",(sm,0,59)),
		                        ("End hour",  (eh,0,23)),("End min",  (em,0,59))]:
			if not (lo <= v <= hi):
				messagebox.showerror("Schedule", f"{name} must be {lo}–{hi}.")
				self.schedule_var.set(False); return
		self.processor.set_schedule(enabled, sh, sm, 0, eh, em, 0)

	# ------------------------------------------------------------------
	# Frame saver callbacks
	# ------------------------------------------------------------------
	def _on_framesaver_toggle(self):
		enabled = self.framesaver_var.get()
		try:
			interval = float(self.framesaver_interval_var.get())
			if interval <= 0: raise ValueError
		except ValueError:
			messagebox.showerror("Frame Saver", "Interval must be a positive number (minutes).")
			self.framesaver_var.set(False); return
		self.processor.set_frame_saver(enabled, interval)

	# ------------------------------------------------------------------
	# ROI callbacks
	# ------------------------------------------------------------------
	def _open_roi_selector(self):
		if not self.processor.running or self.processor.camera_status != "active":
			messagebox.showwarning("ROI Selector",
			                       "The camera must be actively running to set a ROI.\n"
			                       "(If scheduled, wait until the window opens.)")
			return
		started = self.processor.begin_roi_selection()
		if not started:
			messagebox.showwarning("ROI Selector",
			                       "Could not start ROI selection — is the camera running?")
			return
		self.root.after(100, self._poll_roi_done)

	def _poll_roi_done(self):
		if not self.processor.roi_selection_done():
			self.root.after(100, self._poll_roi_done)
			return
		roi = self.processor.get_roi_result()
		if roi:
			self.roi_label.config(
				text=(f"ROI: ({roi[0]},{roi[1]}) → ({roi[2]},{roi[3]})  "
				      f"{roi[2]-roi[0]}×{roi[3]-roi[1]} px"),
				foreground="green")
		else:
			self.roi_label.config(text="ROI: full frame (selection cancelled)", foreground="gray")

	def _clear_roi(self):
		self.processor.set_roi(None)
		self.roi_label.config(text="ROI: full frame", foreground="gray")

	# ------------------------------------------------------------------
	# Save live settings to INI
	# ------------------------------------------------------------------
	def _save_settings(self):
		"""Snapshot all current GUI values into the [live] section of the INI."""
		roi = self.processor.roi
		roi_str = ','.join(map(str, roi)) if roi else ''
		settings = {
			'camera_index':            str(self.cam_combo.get()),
			'resolution':              self.res_combo.get(),
			'throttle_fps':            str(self.throttle_var.get()),
			'display_mode':            self.display_var.get(),
			'classifier_enabled':      str(self.classifier_var.get()).lower(),
			'manual_recording':        str(self.manual_rec_var.get()).lower(),
			'detect_recording':        str(self.detect_rec_var.get()).lower(),
			'show_detections_in_rec':  str(self.show_det_var.get()).lower(),
			'schedule_enabled':        str(self.schedule_var.get()).lower(),
			'schedule_start_hh':       self.sched_sh.get(),
			'schedule_start_mm':       self.sched_sm.get(),
			'schedule_end_hh':         self.sched_eh.get(),
			'schedule_end_mm':         self.sched_em.get(),
			'framesaver_enabled':      str(self.framesaver_var.get()).lower(),
			'framesaver_interval_min': self.framesaver_interval_var.get(),
			'roi':                     roi_str,
		}
		save_live_settings(settings)
		# Brief visual feedback on the button
		self.save_settings_btn.config(text="✓ Saved")
		self.root.after(1500, lambda: self.save_settings_btn.config(text="💾 Save settings"))

	# ------------------------------------------------------------------
	# GUI update loop
	# ------------------------------------------------------------------
	def _update_gui(self):
		status = self.processor.camera_status
		text, colour = self._STATUS_DISPLAY.get(status, ("", "gray"))
		self.status_label.config(text=text, foreground=colour)

		if status == "active":
			self.fps_label.config(text=f"{self.processor.latest_fps:.1f}")
		else:
			self.fps_label.config(text="—")

		self.root.after(250, self._update_gui)

	def _quit(self):
		if messagebox.askokcancel("Quit", "Stop and exit?"):
			self.processor.stop()
			self.root.quit()


# =========================================================
# Main
# =========================================================
def main():
	processor = CameraProcessor()
	root      = tk.Tk()
	gui       = ControlGUI(root, processor)
	cams = scan_cameras(8)
	if cams:
		gui.cam_combo.set(cams[0])
		processor.set_camera(cams[0])
	processor.set_resolution((640, 480))
	root.protocol("WM_DELETE_WINDOW", gui._quit)
	root.mainloop()

if __name__ == '__main__':
	main()
