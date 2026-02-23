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
import platform
import subprocess




# -------------------------
# Typical resolutions list
# add your own if necessary - opencv negotiates the
# nearest resolution supported by your camera
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
	"""Return the expected NCNN export directory for a given .pt path."""
	base, ext = os.path.splitext(weights_path)
	# Ultralytics export typically creates a folder named like "<base>_ncnn_model"
	return base + "_ncnn_model"

def ncnn_files_exist(ncnn_dir):
	"""Return True if NCNN param+bin appear to exist in the export dir."""
	if not os.path.isdir(ncnn_dir):
		return False
	# Look for .param and .bin files (ncnn export creates *.param and *.bin)
	has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
	has_bin = any(f.endswith(".bin") for f in os.listdir(ncnn_dir))
	return has_param and has_bin

def ensure_ncnn_export(weights_path, task, timeout=300):
	"""
	Ensure an NCNN conversion exists for weights_path.
	Returns the ncnn_dir on success, None on failure (falls back to .pt).
	This will skip conversion if the ncnn folder already exists.
	"""
	ncnn_dir = ncnn_dir_for_weights(weights_path)
	if ncnn_files_exist(ncnn_dir):
		return ncnn_dir

	try:
		print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
		model = YOLO(weights_path, task=task)
		# Use Ultralytics export API. This creates the folder "<base>_ncnn_model".
		# Some installs can be slow; we try and catch errors below.
		model.export(format="ncnn")
		# Wait a short time for files to appear (export is synchronous in most versions).
		start = time.time()
		while time.time() - start < timeout:
			if ncnn_files_exist(ncnn_dir):
				print(f"NCNN export complete: {ncnn_dir}")
				return ncnn_dir
			time.sleep(0.5)
		# timed out
		print(f"NCNN export timeout for {weights_path}")
		return None
	except Exception as e:
		# Don't crash — export can fail on some systems; print useful debugging info and return None
		print(f"Warning: NCNN export failed for {weights_path}: {e}")
		return None

def load_model_with_ncnn_preference(weights_path, task):
	"""
	Attempt to use NCNN if available (or convert it). If conversion or loading fails,
	fall back to the original PyTorch .pt path.
	Returns a YOLO model instance (which may wrap NCNN or .pt).
	"""
	# If a .pt was not provided (maybe already a folder), just try loading directly
	if not weights_path.endswith(".pt"):
		try:
			return YOLO(weights_path, task=task)
		except Exception as e:
			print(f"Error loading model {weights_path}: {e}")
			raise

	ncnn_dir = ncnn_dir_for_weights(weights_path)
	# prefer existing NCNN dir if present
	if ncnn_files_exist(ncnn_dir):
		try:
			print(f"Loading NCNN model from {ncnn_dir}")
			return YOLO(ncnn_dir, task=task)
		except Exception as e:
			print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

	# Otherwise attempt conversion (one-time). If it fails, fall back to .pt.
	exported = ensure_ncnn_export(weights_path, task)
	if exported:
		try:
			return YOLO(exported, task=task)
		except Exception as e:
			print(f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)")

	# Finally, fallback to direct .pt load
	print(f"Using original weights (PyTorch) at {weights_path}")
	return YOLO(weights_path, task=task)
# --------------------------------------------------------------------




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

clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
input_dir_ini = config['DEFAULT'].get('input_dir', 'input')
output_dir_ini = config['DEFAULT'].get('output_dir', 'output')
clips_dir = resolve_project_path(clips_dir_ini, 'clips')
input_folder = resolve_project_path(input_dir_ini, 'input')
output_folder = resolve_project_path(output_dir_ini, 'output')

# ---------- Raspberry Pi detection & picamera2 handling ----------
def is_raspberry_pi():
	"""Return True if running on Raspberry Pi hardware / Raspberry Pi OS."""
	try:
		if platform.system() != "Linux":
			return False
		# Primary check: /proc/device-tree/model exists on Pi images
		model_path = "/proc/device-tree/model"
		if os.path.exists(model_path):
			with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
				model = f.read().lower()
				if "raspberry pi" in model:
					return True
		# Fallback: /etc/os-release may contain raspbian or raspberry
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
			# Try a best-effort pip install (may still fail on some Pi images)
			print("picamera2 import failed:", e)
			print("Attempting pip install of picamera2 (best-effort)...")
			try:
				subprocess.check_call([sys.executable, "-m", "pip", "install", "picamera2"])
				from picamera2 import Picamera2  # noqa: F401
				PICAMERA2_AVAILABLE = True
				print("picamera2 installed via pip and imported successfully.")
			except Exception as e2:
				print("Automatic picamera2 install failed:", e2)
				print("If you are on Raspberry Pi OS, prefer installing from apt:")
				print("  sudo apt update && sudo apt install -y python3-picamera2 libcamera-apps")
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
	primary_colors = primary_static_colors + primary_motion_colors
	primary_hotkeys = primary_static_hotkeys + primary_motion_hotkeys

	secondary_classes = secondary_static_classes + secondary_motion_classes
	secondary_colors = secondary_static_colors + secondary_motion_colors
	secondary_hotkeys = secondary_static_hotkeys + secondary_motion_hotkeys

	primary_static_model_path = os.path.join('model_primary_static', "train", "weights", "best.pt")
	primary_motion_model_path = os.path.join('model_primary_motion', "train", "weights", "best.pt")

	ignore_secondary = [name.strip() for name in config['DEFAULT']['ignore_secondary'].split(',')]
	dominant_source = config['DEFAULT']['dominant_source'].lower()

	scale_factor = float(config['DEFAULT'].get('scale_factor','1.0'))
	use_ncnn = config['DEFAULT']['use_ncnn'].lower()
	expA = float(config['DEFAULT'].get('expA','0.5'))
	expB = float(config['DEFAULT'].get('expB','0.8'))
	lum_weight = float(config['DEFAULT'].get('lum_weight','0.7'))
	strategy = config['DEFAULT'].get('strategy','exponential')
	chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
	rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh','0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh','0.5'))
	match_distance_thresh = float(config['DEFAULT'].get('match_distance_thresh','200'))
	delete_after_missed = float(config['DEFAULT'].get('delete_after_missed','5'))
	centroid_merge_thresh = float(config['DEFAULT'].get('centroid_merge_thresh','50'))
	iou_thresh = float(config['DEFAULT'].get('iou_thresh','0.95'))
	line_thickness = int(config['DEFAULT'].get('line_thickness','1'))
	font_size = float(config['DEFAULT'].get('font_size','0.5'))
	frame_skip = int(config['DEFAULT'].get('frame_skip','0'))

	process_noise_pos = float(config['kalman'].get('process_noise_pos','0.01'))
	process_noise_vel = float(config['kalman'].get('process_noise_vel','0.1'))
	measurement_noise = float(config['kalman'].get('measurement_noise','0.1'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold','0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")

# Validation (kept)
if dominant_source not in ('motion','static','confidence'):
	raise ValueError("dominant_source must be motion, static, or confidence")

os.makedirs(clips_dir, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

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
		self.max_missed = max_missed

	def _create_kf(self, initial_pt):
		kf = cv2.KalmanFilter(4,2)
		kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
		kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
		kf.processNoiseCov = np.diag([process_noise_pos, process_noise_pos, process_noise_vel, process_noise_vel]).astype(np.float32)
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
		preds = self.predict_all()
		track_ids = [t[0] for t in preds]
		pred_pts   = [t[1] for t in preds]

		if pred_pts and detections:
			cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
			for i, p in enumerate(pred_pts):
				for j, d in enumerate(detections):
					cost[i,j] = np.hypot(p[0]-d[0], p[1]-d[1])
			row_idx, col_idx = linear_sum_assignment(cost)
		else:
			row_idx = np.array([], dtype=int); col_idx = np.array([], dtype=int)
			cost = np.array([])

		assigned_detects = {}; matched_tracks = set(); matched_dets = set()

		for r,c in zip(row_idx, col_idx):
			if cost[r,c] < self.dist_thresh:
				tid = track_ids[r]
				matched_tracks.add(tid); matched_dets.add(c); assigned_detects[c] = tid
				dpt = detections[c]
				meas = np.array([[np.float32(dpt[0])],[np.float32(dpt[1])]])
				self.tracks[tid]['kf'].correct(meas)
				self.tracks[tid]['missed'] = 0
				self.prev_positions[tid] = (dpt[0], dpt[1])

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
				tid = self.next_id
				kf = self._create_kf(dpt)
				self.tracks[tid] = {'kf': kf, 'missed': 0}
				assigned_detects[i] = tid
				self.prev_positions[tid] = (dpt[0], dpt[1])
				matched_tracks.add(tid)
				self.next_id += 1

		for tid in list(self.tracks.keys()):
			if tid not in matched_tracks:
				self.tracks[tid]['missed'] += 1
				noise_scale = min(2.0, 1.0 + self.tracks[tid]['missed'] * 0.2)
				kf = self.tracks[tid]['kf']; new_noise = kf.processNoiseCov.copy(); new_noise *= noise_scale; kf.processNoiseCov = new_noise
				if self.tracks[tid]['missed'] > self.max_missed:
					del self.tracks[tid]
					if tid in self.prev_positions: del self.prev_positions[tid]

		return assigned_detects

# -------------------------
# Camera processing
# -------------------------
class CameraProcessor:
	def __init__(self):
		self.cap = None
		self.camera_index = 0
		self.thread = None
		self.stop_event = threading.Event()
		self.running = False
		self.classifier_enabled = True
		self.manual_recording = False
		self.manual_writer = None
		self.display_mode = 'static'
		self.show_window_name = "BehaveAI Live"
		self.frame_lock = threading.Lock()
		self.frame_timestamps = deque()
		self.latest_fps = 0.0
		self.desired_fps = 0.0
		self.desired_resolution = (640,480)
		self.detection_recording_enabled = False
		self.detection_writer = None
		self.detection_active = False
		self.detection_last_seen = 0.0
		self.detection_stop_seconds = 3.0
		self.show_detections_in_recording = True

		# Tracker & models
		self.tracker = KalmanTracker(match_distance_thresh, delete_after_missed)
		self.model_static = None
		self.model_motion = None
		# secondary models keyed by PRIMARY class name (this matches train-time naming)
		self.secondary_static_models = {}
		self.secondary_motion_models = {}

		# frame buffers
		self.prev_frames = None

		# ensure output dirs
		os.makedirs(clips_dir, exist_ok=True)
		os.makedirs(output_folder, exist_ok=True)
		# load primary & secondary models (if present)
		self._maybe_load_models()

	def set_display_mode(self, mode: str):
		"""Set display_mode safely from GUI. Accepts 'static', 'motion', or 'disabled'."""
		if mode not in ('static', 'motion', 'disabled'):
			print("set_display_mode: invalid mode:", mode)
			return
		# small lock just in case GUI calls this while thread is running
		with self.frame_lock:
			self.display_mode = mode
		print(f"Display mode set to: {mode}")

	def set_show_detections_in_recording(self, flag: bool):
		"""Toggle whether saved manual/detection recordings include annotation overlays."""
		with self.frame_lock:
			self.show_detections_in_recording = bool(flag)
		print(f"Show detections in recordings set to: {self.show_detections_in_recording}")
		

	def _maybe_load_models(self):
		# Primary models
		try:
			if len(primary_static_classes) > 0 and os.path.exists(primary_static_model_path):
				# ~ self.model_static = YOLO(primary_static_model_path)
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
				# ~ self.model_motion = YOLO(primary_motion_model_path)
				# ~ print("Loaded primary motion model")
		except Exception as e:
			print("Primary model load failed:", e)
			self.model_static = None; self.model_motion = None

		# Load secondary models per PRIMARY class (matching naming used by annotation/training)
		if hierarchical_mode:
			for p in primary_classes:
				if p in ignore_secondary: 
					continue
				# static secondary model for this primary
				try:
					model_dir = f"model_secondary_static_{p}"
					model_path = os.path.join(model_dir, "train", "weights", "best.pt")
					if os.path.exists(model_path):
						# ~ self.secondary_static_models[p] = YOLO(model_path)
						# ~ print(f"Loaded secondary static model for primary '{p}'")
						if use_ncnn == 'true':
							self.secondary_static_models[p] = load_model_with_ncnn_preference(model_path, "classify")
							print(f"Loaded secondary static NCNN model for primary '{p}'")
						else:
							self.secondary_static_models[p] = YOLO(model_path)
							print(f"Loaded secondary static YOLO model for primary '{p}'")

				except Exception as e:
					print(f"Secondary static model load failed for {p}: {e}")
				# motion secondary model for this primary
				try:
					model_dir = f"model_secondary_motion_{p}"
					model_path = os.path.join(model_dir, "train", "weights", "best.pt")
					if os.path.exists(model_path):
						# ~ self.secondary_motion_models[p] = YOLO(model_path)
						# ~ print(f"Loaded secondary motion model for primary '{p}'")
						if use_ncnn == 'true':
							self.secondary_motion_models[p] = load_model_with_ncnn_preference(model_path, "classify")
							print(f"Loaded secondary motion NCNN model for primary '{p}'")
						else:
							self.secondary_static_models[p] = YOLO(model_path)
							print(f"Loaded secondary motion YOLO model for primary '{p}'")
				except Exception as e:
					print(f"Secondary motion model load failed for {p}: {e}")

	# ~ def set_camera(self, index):
		# ~ with self.frame_lock:
			# ~ if self.running:
				# ~ print("set_camera: processor already running; camera index change ignored.")
				# ~ return
			# ~ self.camera_index = int(index)

	def set_resolution(self, wh_tuple):
		if isinstance(wh_tuple, str):
			try:
				w,h = map(int, wh_tuple.split('x'))
			except Exception:
				return
		else:
			w,h = wh_tuple
		self.desired_resolution = (int(w), int(h))
		# If already open, try to apply once
		if getattr(self,'cap',None) is not None and self.cap.isOpened():
			try:
				self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
				self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
				time.sleep(0.05)
				print(f"set_resolution: applied {w}x{h} to open camera")
			except Exception:
				pass
		else:
			print(f"set_resolution: desired set to {w}x{h}; will apply when camera opens")

	def toggle_classifier(self, enable: bool):
		self.classifier_enabled = bool(enable)
		if self.classifier_enabled and (self.model_static is None and self.model_motion is None):
			self._maybe_load_models()

	def toggle_manual_recording(self, enable: bool):
		if enable and not self.manual_recording:
			timestamp = time.strftime("%Y%m%d-%H%M%S")
			path = os.path.join(clips_dir, f"manual_clip_{timestamp}.mp4")
			self.manual_writer_path = path
			self.manual_recording = True
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
			self.detection_writer.release(); self.detection_writer = None; self.detection_active = False

	def start(self):
		if getattr(self,'thread',None) is not None and self.thread.is_alive():
			print("CameraProcessor.start: thread already running; start ignored.")
			return
		self.stop_event.clear()
		self.thread = threading.Thread(target=self._run_loop, daemon=True)
		self.running = True
		self.thread.start()

	def stop(self):
		# used on quit only
		self.stop_event.set()
		if getattr(self,'thread',None) is not None:
			try:
				if self.thread.is_alive():
					self.thread.join(timeout=2.0)
			except Exception:
				pass
		self.running = False
		try:
			if getattr(self,'manual_writer',None): self.manual_writer.release(); self.manual_writer=None
		except: pass
		try:
			if getattr(self,'detection_writer',None): self.detection_writer.release(); self.detection_writer=None
		except: pass
		try:
			if getattr(self,'cap',None): self.cap.release(); self.cap=None
		except: pass
		self.thread = None

	# ~ def _open_camera(self, timeout=5.0, retry_interval=0.2):
		# ~ deadline = time.time() + float(timeout)
		# ~ while not self.stop_event.is_set() and time.time() < deadline:
			# ~ try:
				# ~ if getattr(self,'cap',None):
					# ~ try: self.cap.release()
					# ~ except: pass
					# ~ self.cap = None
				# ~ self.cap = cv2.VideoCapture(int(self.camera_index), cv2.CAP_ANY)
				# ~ time.sleep(0.06)
				# ~ if self.cap is not None and self.cap.isOpened():
					# ~ w,h = self.desired_resolution
					# ~ try:
						# ~ self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
						# ~ self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
						# ~ time.sleep(0.05)
						# ~ print(f"_open_camera: camera opened (requested {w}x{h})")
					# ~ except Exception:
						# ~ pass
					# ~ return True
				# ~ else:
					# ~ try: 
						# ~ if self.cap: self.cap.release()
					# ~ except: pass
					# ~ self.cap = None
			# ~ except Exception as e:
				# ~ last_err = e
			# ~ time.sleep(retry_interval)
		# ~ print(f"_open_camera: failed to open camera {self.camera_index} within timeout")
		# ~ return False

	def set_camera(self, index):
		with self.frame_lock:
			if self.running:
				print("set_camera: processor already running; camera index change ignored.")
				return
			# accept either numeric index or the string "picamera"
			# store as string for consistency with scan_cameras output
			self.camera_index = index if isinstance(index, str) else str(index)

	def _open_camera(self, timeout=5.0, retry_interval=0.2):
		"""
		Open either an OpenCV VideoCapture (when camera_index is numeric string)
		or the picamera2 wrapper when camera_index == "picamera".
		Returns True on success, False on failure.
		"""
		deadline = time.time() + float(timeout)
		while not self.stop_event.is_set() and time.time() < deadline:
			try:
				# Clean up existing capture if present
				if getattr(self, "cap", None):
					try: self.cap.release()
					except: pass
					self.cap = None

				# Picamera2 path
				if str(self.camera_index) == "picamera" and IS_RPI and PICAMERA2_AVAILABLE:
					try:
						res = self.desired_resolution if getattr(self, "desired_resolution", None) else (640, 480)
						# choose requested fps: prefer processor.desired_fps if >0, else try 30
						fps_req = self.desired_fps if getattr(self, "desired_fps", 0.0) else 30.0
						# create wrapper and start (will raise on repeated failure)
						self.cap = Picamera2Wrapper(resolution=tuple(map(int, res)), fps=int(round(fps_req)), retries=4, retry_delay=0.5)
						self.cap.start()
						if self.cap.isOpened():
							print(f"_open_camera: picamera opened (requested {res[0]}x{res[1]} @ {fps_req}fps)")
							# When using Picamera2 wrapper, we can't rely on cap.get(cv2.CAP_PROP_...) calls.
							# Set w,h,fps variables elsewhere in the code using self.desired_resolution and fps_req.
							return True
					except Exception as e:
						print("_open_camera: picamera2 startup error:", e)
						try:
							if getattr(self, "cap", None):
								try: self.cap.release()
								except: pass
						except Exception:
							pass
						self.cap = None
				
				else:
					# OpenCV path: camera_index should be numeric string like "0"
					try:
						idx = int(self.camera_index)
					except Exception:
						# fallback to 0
						idx = 0
					self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
					time.sleep(0.06)
					if self.cap is not None and self.cap.isOpened():
						w,h = self.desired_resolution
						try:
							self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
							self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
							time.sleep(0.05)
							print(f"_open_camera: camera opened (requested {w}x{h})")
						except Exception:
							pass
						return True
					else:
						try:
							if self.cap: self.cap.release()
						except: pass
						self.cap = None
			except Exception as e:
				last_err = e
			time.sleep(retry_interval)
		print(f"_open_camera: failed to open camera {self.camera_index} within timeout")
		return False

	def _start_manual_writer(self, w, h, fps):
		if not hasattr(self,'manual_writer_path'): return
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.manual_writer = cv2.VideoWriter(self.manual_writer_path, fourcc, fps, (w,h))
		delattr(self, 'manual_writer_path')

	def _start_detection_writer(self, w, h, fps):
		timestamp = time.strftime("%Y%m%d-%H%M%S")
		path = os.path.join(output_folder, f"det_{timestamp}.mp4")
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.detection_writer = cv2.VideoWriter(path, fourcc, fps, (w,h))
		self.detection_active = True
		print("Detection recording started:", path)

	def _run_loop(self):
		if not self._open_camera(timeout=5.0, retry_interval=0.2):
			print("CameraProcessor._run_loop: failed to open camera; aborting.")
			self.running = False
			return

		cap = self.cap
		# use requested or actual sizes scaled by scale_factor
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
		fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
		if fps <= 0: fps = 20.0

		timestamp_session = time.strftime("%Y%m%d-%H%M%S")
		csv_path = os.path.join(output_folder, f"det_{timestamp_session}.csv")
		csv_file = open(csv_path, 'w', newline='')
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow([
			"frame","id","x","y",
			"primary_static_class","primary_static_conf",
			"primary_motion_class","primary_motion_conf",
			"secondary_static_class","secondary_static_conf",
			"secondary_motion_class","secondary_motion_conf"
		])

		prev_frames = None
		frame_idx = 0
		frame_count = 0

		cv2.namedWindow(self.show_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)

		while not self.stop_event.is_set():
			frame_start_time = time.time()
			ret, raw_frame = cap.read()
			if not ret:
				time.sleep(0.01); continue
			frame_idx += 1

			if scale_factor != 1.0:
				raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)

			gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
			frame = raw_frame.copy()

			# initialize prev_frames (identical behaviour)
			if prev_frames is None:
				prev_frames = [gray.copy() for _ in range(3)]
				motion_image = frame.copy()
				tnow = time.time(); self.frame_timestamps.append(tnow)
				while self.frame_timestamps and (tnow - self.frame_timestamps[0]) > 1.0:
					self.frame_timestamps.popleft()
				self.latest_fps = float(len(self.frame_timestamps))
				display_frame = frame if self.display_mode=='static' else (motion_image if self.display_mode=='motion' else np.zeros_like(frame))
				if self.manual_recording or self.detection_active:
					self._draw_record_icon(display_frame, self.manual_recording, self.detection_active)
				cv2.imshow(self.show_window_name, display_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					self.stop_event.set()
				if self.desired_fps > 0.0:
					target = 1.0 / self.desired_fps; elapsed = time.time() - frame_start_time
					if elapsed < target: time.sleep(target - elapsed)
				continue

			# Motion processing
			if primary_motion_classes[0] != '0' or secondary_motion_classes[0] != '0':
				diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
				if strategy == 'exponential':
					prev_frames[0] = gray
					prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, 1 - expA, 0)
					prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, 1 - expB, 0)
				else:
					prev_frames[2] = prev_frames[1]; prev_frames[1] = prev_frames[0]; prev_frames[0] = gray

				if chromatic_tail_only == 'true':
					tb = cv2.subtract(diffs[0], diffs[1]); tr = cv2.subtract(diffs[2], diffs[1]); tg = cv2.subtract(diffs[1], diffs[0])
					blue = cv2.addWeighted(gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
					red = cv2.addWeighted(gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
				else:
					blue = cv2.addWeighted(gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
					red = cv2.addWeighted(gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)

				motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
			else:
				motion_image = frame.copy()

			# Primary detections
			all_detections = []
			if self.classifier_enabled:
				# static model on frame
				if primary_static_classes[0] != '0' and self.model_static is not None:
					try:
						results_static = self.model_static.predict(frame, conf=primary_conf_thresh, verbose=False)
						for box in results_static[0].boxes:
							x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
							cls_idx = int(box.cls[0]); conf = float(box.conf[0])
							class_name = primary_static_classes[cls_idx]
							all_detections.append({'coords':(x1,y1,x2,y2),'primary_class':class_name,'primary_conf':conf,'source':'static'})
					except Exception as e:
						print("Static model inference error:", e)

				# motion model on motion_image
				if primary_motion_classes[0] != '0' and self.model_motion is not None:
					try:
						results_motion = self.model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
						for box in results_motion[0].boxes:
							x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
							cls_idx = int(box.cls[0]); conf = float(box.conf[0])
							class_name = primary_motion_classes[cls_idx]
							all_detections.append({'coords':(x1,y1,x2,y2),'primary_class':class_name,'primary_conf':conf,'source':'motion'})
					except Exception as e:
						print("Motion model inference error:", e)

			# Merge detections between static and motion (same logic)
			merged_detections = []
			for det in all_detections:
				x1,y1,x2,y2 = det['coords']; cx,cy = (x1+x2)//2, (y1+y2)//2
				matched = False
				for md in merged_detections:
					md_cx, md_cy = md['centroid']; dist = np.hypot(cx-md_cx, cy-md_cy)
					md_x1, md_y1, md_x2, md_y2 = md['coords']; overlap = iou((x1,y1,x2,y2),(md_x1,md_y1,md_x2,md_y2))
					ms_source = md.get('source','')
					if dist < centroid_merge_thresh or overlap > iou_thresh:
						# decide which to prefer based on dominant_source or confidence
						if det['source'] == ms_source or dominant_source == 'confidence':
							if det['primary_conf'] > md.get('primary_conf',0):
								md['primary_class_combined'] = md.get('primary_class','')
								md['primary_conf_combined'] = md.get('primary_conf',0)
								md['primary_class'] = det['primary_class']
								md['primary_conf'] = det['primary_conf']
								md['coords'] = det['coords']; md['centroid'] = (cx,cy); md['source'] = det['source']
						elif det['source'] == 'static' and dominant_source == 'static':
							md['primary_class_combined'] = md.get('primary_class',''); md['primary_conf_combined'] = md.get('primary_conf',0)
							md['primary_class'] = det['primary_class']; md['primary_conf'] = det['primary_conf']; md['coords'] = det['coords']; md['centroid'] = (cx,cy); md['source'] = det['source']
						elif det['source'] == 'motion' and dominant_source == 'motion':
							md['primary_class_combined'] = md.get('primary_class',''); md['primary_conf_combined'] = md.get('primary_conf',0)
							md['primary_class'] = det['primary_class']; md['primary_conf'] = det['primary_conf']; md['coords'] = det['coords']; md['centroid'] = (cx,cy); md['source'] = det['source']
						matched = True; break
				if not matched:
					new_det = {'coords':det['coords'],'centroid':(cx,cy),'source':det['source'],'primary_class':det['primary_class'],'primary_conf':det['primary_conf'],'primary_class_combined':'','primary_conf_combined':0.0}
					merged_detections.append(new_det)

			# Secondary classification, tracking and drawing
			processed_detections = []
			for det in merged_detections:
				coords = det['coords']; source = det['source']
				primary_class = det['primary_class']; primary_conf = det['primary_conf']
				# populate per-source primary fields
				if source == 'static':
					det['primary_static_class'] = primary_class; det['primary_static_conf'] = primary_conf
					det['primary_motion_class'] = det.get('primary_class_combined',''); det['primary_motion_conf'] = det.get('primary_conf_combined',0.0)
				else:
					det['primary_motion_class'] = primary_class; det['primary_motion_conf'] = primary_conf
					det['primary_static_class'] = det.get('primary_class_combined',''); det['primary_static_conf'] = det.get('primary_conf_combined',0.0)

				# Secondary classification (only if hierarchical)
				if hierarchical_mode:
					x1,y1,x2,y2 = coords
					sec_model = None; crop_img = None
					# Prefer secondary model types consistent with source, fall back to the other if needed
					if source == 'static':
						if primary_class in self.secondary_static_models:
							sec_model = self.secondary_static_models.get(primary_class)
							crop_img = frame
						elif primary_class in self.secondary_motion_models:
							sec_model = self.secondary_motion_models.get(primary_class)
							crop_img = motion_image if primary_motion_classes[0] != '0' else frame
					else:
						if primary_class in self.secondary_motion_models:
							sec_model = self.secondary_motion_models.get(primary_class)
							crop_img = motion_image
						elif primary_class in self.secondary_static_models:
							sec_model = self.secondary_static_models.get(primary_class)
							crop_img = frame

					# Safe crop bounds
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
								idx = sec_results[0].probs.top1
								secondary_conf = sec_results[0].probs.top1conf.item()
								# sec_model.names maps class indices -> class names
								secondary_class = sec_model.names[idx]
						except Exception:
							pass

					# put results into det
					if source == 'static':
						det['secondary_static_class'] = secondary_class; det['secondary_static_conf'] = secondary_conf
						det['secondary_motion_class'] = '' ; det['secondary_motion_conf'] = 0.0
					else:
						det['secondary_motion_class'] = secondary_class; det['secondary_motion_conf'] = secondary_conf
						det['secondary_static_class'] = '' ; det['secondary_static_conf'] = 0.0

				processed_detections.append(det)

			# Tracking by centroid
			cents = [d['centroid'] for d in processed_detections]
			assignment = self.tracker.update(cents)

			# Draw and write CSV rows
			for idx, det in enumerate(processed_detections):
				tid = assignment.get(idx, None)
				if tid is None or tid not in self.tracker.tracks: continue
				x1,y1,x2,y2 = det['coords']; cx,cy = det['centroid']
				ps_class = det.get('primary_static_class',''); ps_conf = det.get('primary_static_conf',0)
				pm_class = det.get('primary_motion_class',''); pm_conf = det.get('primary_motion_conf',0)
				ss_class = det.get('secondary_static_class',''); ss_conf = det.get('secondary_static_conf',0)
				sm_class = det.get('secondary_motion_class',''); sm_conf = det.get('secondary_motion_conf',0)
				p_source = det.get('source','')

				primary_cls = ps_class if p_source=='static' else pm_class
				primary_col = primary_colors[primary_classes.index(primary_cls)] if primary_cls in primary_classes else (0,255,0)
				secondary_col = (255,255,255)
				if hierarchical_mode:
					secondary_cls = ''
					if sm_class and sm_class != primary_cls:
						secondary_cls = sm_class; secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
					if ss_class and ss_class != primary_cls:
						secondary_cls = ss_class; secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
					if primary_cls in ignore_secondary:
						label = f"{tid} {primary_cls.upper()}"
						label_size,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
						lw,lh = label_size
						cv2.rectangle(frame, (x1-line_thickness, y1 - lh - line_thickness*4), (x1 + lw + line_thickness*2, y1), (0,0,0), -1)
						cv2.rectangle(frame, (x1,y1), (x2,y2), primary_col, line_thickness)
						cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_col, line_thickness, cv2.LINE_AA)
					else:
						outer_t = line_thickness + 2
						cv2.rectangle(frame, (x1-outer_t, y1-outer_t), (x2+outer_t, y2+outer_t), primary_col, outer_t)
						label = f"{tid} {primary_cls.upper()} {secondary_cls}"
						label_size,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
						lw,lh = label_size
						cv2.rectangle(frame, (x1-line_thickness, y1 - lh - line_thickness*4), (x1 + lw + line_thickness*2, y1), (0,0,0), -1)
						cv2.rectangle(frame, (x1,y1), (x2,y2), secondary_col, line_thickness)
						cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, secondary_col, line_thickness, cv2.LINE_AA)
				else:
					label = f"{tid} {primary_cls}"
					label_size,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
					lw,lh = label_size
					cv2.rectangle(frame, (x1-line_thickness, y1 - lh - line_thickness*4), (x1 + lw + line_thickness*2, y1), (0,0,0), -1)
					cv2.rectangle(frame, (x1,y1), (x2,y2), primary_col, line_thickness)
					cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_col, line_thickness, cv2.LINE_AA)

				# draw track state vectors
				if tid in self.tracker.tracks:
					sp = self.tracker.tracks[tid]['kf'].statePost
					sx, sy = int(sp[0,0]), int(sp[1,0])
					vx, vy = float(sp[2,0]), float(sp[3,0])
					nx, ny = int(sx + vx), int(sy + vy)
					light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
					cv2.line(frame, (sx,sy), (nx,ny), primary_col, line_thickness)
					cv2.circle(frame, (nx,ny), 3, light_color, -line_thickness)
					cv2.circle(frame, (int(cx), int(cy)), 3, primary_col, -line_thickness)

				# ~ csv_writer.writerow([frame_idx, tid, cx, cy, ps_class, f"{ps_conf:.3f}", pm_class, f"{pm_conf:.3f}", ss_class, f"{ss_conf:.3f}", sm_class, f"{sm_conf:.3f}"])
				# Only record CSV rows while "Record on detection (output)" checkbox is enabled.
				# The CSV file remains the same single session file, but we only append rows
				# when the GUI's detection-recording option is on.
				if self.detection_recording_enabled:
					csv_writer.writerow([
						frame_idx, tid, cx, cy,
						ps_class, f"{ps_conf:.3f}",
						pm_class, f"{pm_conf:.3f}",
						ss_class, f"{ss_conf:.3f}",
						sm_class, f"{sm_conf:.3f}"
					])
				


			# Frame counter label
			text_color = (255,255,255)
			label = str(frame_idx)
			label_size,_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			lw,lh = label_size
			cv2.rectangle(frame, (0,0), (lw + line_thickness*4, lh + line_thickness*4), (0,0,0), -1)
			cv2.putText(frame, label, (line_thickness*2, lh + line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, line_thickness)

			display_frame = frame if self.display_mode=='static' else (motion_image if self.display_mode=='motion' else np.zeros_like(frame))

			# ~ if self.manual_recording and self.manual_writer is None:
				# ~ self._start_manual_writer(w, h, fps)
			# ~ if self.manual_writer is not None:
				# ~ try: self.manual_writer.write(display_frame)
				# ~ except Exception as e: print("Manual writer write failed:", e)
			record_frame = display_frame if self.show_detections_in_recording else raw_frame
			
			if self.manual_recording and self.manual_writer is None:
				self._start_manual_writer(w, h, fps)
			if self.manual_writer is not None:
				try:
					self.manual_writer.write(record_frame)
				except Exception as e:
					print("Manual writer write failed:", e)
					
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
				try:
					self.detection_writer.write(record_frame)
				except Exception as e:
					print("Detection writer write failed:", e)
			# ~ if self.detection_writer is not None:
				# ~ try: self.detection_writer.write(display_frame)
				# ~ except Exception as e: print("Detection writer write failed:", e)

			if self.manual_recording or self.detection_active:
				self._draw_record_icon(display_frame, self.manual_recording, self.detection_active)

			cv2.imshow(self.show_window_name, display_frame)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('q'):
				self.stop_event.set()
				break

			tnow = time.time(); self.frame_timestamps.append(tnow)
			while self.frame_timestamps and (tnow - self.frame_timestamps[0]) > 1.0: self.frame_timestamps.popleft()
			self.latest_fps = float(len(self.frame_timestamps))

			if self.desired_fps and self.desired_fps > 0.0:
				target_period = 1.0 / float(self.desired_fps)
				elapsed = time.time() - frame_start_time
				if elapsed < target_period: time.sleep(target_period - elapsed)

			frame_count += 1
			if frame_count > frame_skip: frame_count = 0

		csv_file.close()
		if self.manual_writer: self.manual_writer.release(); self.manual_writer = None
		if self.detection_writer: self.detection_writer.release(); self.detection_writer = None
		try: cv2.destroyWindow(self.show_window_name)
		except: pass
		print("Camera processing stopped. CSV saved to:", csv_path)

	def _draw_record_icon(self, img, manual_active: bool, detection_active: bool):
		h, w = img.shape[:2]; margin = 8; dot_radius = 8; label = "REC"
		if manual_active and detection_active: label = "REC M+D"
		elif manual_active: label = "REC M"
		elif detection_active: label = "REC D"
		font = cv2.FONT_HERSHEY_SIMPLEX; fscale = 0.5; fth = 1
		(text_w, text_h), _ = cv2.getTextSize(label, font, fscale, fth)
		rect_w = dot_radius * 2 + 6 + text_w + 6; rect_h = max(text_h + 6, dot_radius * 2 + 6)
		x1 = w - margin - rect_w; y1 = margin; x2 = w - margin; y2 = y1 + rect_h
		overlay = img.copy(); cv2.rectangle(overlay, (x1,y1),(x2,y2),(0,0,0), -1); alpha = 0.45
		cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
		dot_cx = x1 + 6 + dot_radius; dot_cy = y1 + rect_h // 2
		cv2.circle(img, (dot_cx, dot_cy), dot_radius, (0,0,255), -1)
		cv2.circle(img, (dot_cx + 1, dot_cy - 1), max(1, dot_radius // 3), (255,255,255), -1)
		text_x = dot_cx + dot_radius + 6; text_y = dot_cy + text_h // 2 - 2
		cv2.putText(img, label, (text_x, text_y), font, fscale, (200,200,200), fth, cv2.LINE_AA)

# -------------------------
# Camera enumeration helper
# -------------------------
class Picamera2Wrapper:
    """
    Picamera2 wrapper that mimics minimal cv2.VideoCapture API:
      - start()
      - isOpened()
      - read() -> (ret, frame) (BGR)
      - release()
      - get(prop)  # supports CAP_PROP_FRAME_WIDTH, _HEIGHT, _FPS
      - set(prop, value)  # best-effort, tries to reconfigure if running
    """
    def __init__(self, resolution=(640, 480), fps=30, retries=3, retry_delay=0.5):
        from picamera2 import Picamera2
        self.Picamera2 = Picamera2
        self.resolution = tuple(map(int, resolution))
        self.target_fps = float(fps) if fps and fps > 0 else 30.0
        self.retries = int(retries)
        self.retry_delay = float(retry_delay)
        self.picam = None
        self._started = False

        # small internal state to behave like cv2 capture
        self._width = float(self.resolution[0])
        self._height = float(self.resolution[1])
        self._fps = float(self.target_fps)

        try:
            self.picam = self.Picamera2()
        except Exception as e:
            raise RuntimeError(f"Picamera2 construction failed: {e}") from e

    def configure_and_start(self):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                cfg = self.picam.create_video_configuration(
                    main={"size": (int(self._width), int(self._height)), "format": "BGR888"},
                    controls={"FrameRate": int(round(self._fps))}
                )
                self.picam.configure(cfg)
                self.picam.start()
                self._started = True

                # warm-up: allow libcamera to stabilise and discard a couple of frames
                import time
                time.sleep(0.05)
                for _ in range(2):
                    try:
                        _ = self.picam.capture_array()
                    except Exception:
                        time.sleep(0.02)
                return True
            except Exception as e:
                last_exc = e
                # best-effort cleanup
                try:
                    if self._started:
                        try: self.picam.stop()
                        except Exception: pass
                except Exception:
                    pass
                # recreate picamera object for next attempt
                try:
                    self.picam = self.Picamera2()
                except Exception:
                    last_exc = RuntimeError(f"Failed to recreate Picamera2 after attempt {attempt}: {e}")
                    break
                import time
                time.sleep(self.retry_delay)
        raise RuntimeError(f"Picamera2 configure/start failed after {self.retries} attempts: {last_exc}") from last_exc

    def start(self):
        if self._started:
            return True
        return self.configure_and_start()

    def isOpened(self):
        return bool(self._started)

    def read(self):
        """Return (ret, frame) where frame is BGR (OpenCV-compatible)."""
        if not self._started:
            return False, None
        try:
            arr = self.picam.capture_array()
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            # transient capture errors can happen; return False so caller can skip
            # (caller may implement small sleep/retry)
            # Optionally: print("Picamera2Wrapper.read error:", e)
            return False, None

    def release(self):
        try:
            if self._started:
                try: self.picam.stop()
                except Exception: pass
            try:
                if hasattr(self.picam, "close"):
                    self.picam.close()
            except Exception:
                pass
        except Exception:
            pass
        self._started = False

    # --- cv2-like property get/set ---
    def get(self, prop):
        # Use OpenCV constants so code stays readable
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        # fallback: unknown property -> 0.0
        return 0.0

    def set(self, prop, value):
        """
        Best-effort: update internal values and, if already started, attempt to reconfigure.
        Returns True on success-like, False on failure.
        """
        try:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self._width = float(value)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self._height = float(value)
            elif prop == cv2.CAP_PROP_FPS:
                self._fps = float(value)
            else:
                # unsupported property — accept but no effect
                return False

            # if running, attempt to reconfigure pipeline to new size/fps
            if self._started:
                try:
                    # stop, reconfigure, then start again
                    try: self.picam.stop()
                    except Exception: pass
                    cfg = self.picam.create_video_configuration(
                        main={"size": (int(self._width), int(self._height)), "format": "RGB888"},
                        controls={"FrameRate": int(round(self._fps))}
                    )
                    self.picam.configure(cfg)
                    time_sleep = 0.05
                    import time
                    time.sleep(time_sleep)
                    self.picam.start()
                    # warm up a frame quickly
                    try:
                        _ = self.picam.capture_array()
                    except Exception:
                        time.sleep(0.02)
                except Exception:
                    # if reconfigure fails, mark as not started and return False
                    self._started = False
                    return False
            return True
        except Exception:
            return False



# -------------------------
# Camera enumeration helper (replaced)
# returns list of strings ("0","1",...,"picamera")
# -------------------------
def scan_cameras(max_search=6):
	available = []
	# First enumerate OpenCV (USB / V4L2) devices
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
			if ret:
				available.append(str(idx))
		except Exception:
			continue
	# If running on a Raspberry Pi and picamera2 is available, include it.
	if IS_RPI and PICAMERA2_AVAILABLE:
		available.append("picamera")
	# If nothing found, return a sensible default (string "0")
	return available if available else ["0"]


# -------------------------
# GUI
# -------------------------
class ControlGUI:
	def __init__(self, root, processor: CameraProcessor):
		self.root = root; self.processor = processor
		root.title("BehaveAI - Live Camera Controls")
		frm = ttk.Frame(root, padding=8); frm.grid(sticky="nsew")

		ttk.Label(frm, text="Camera:").grid(row=0,column=0,sticky="w")
		cams = scan_cameras(8)
		self.cam_combo = ttk.Combobox(frm, values=cams, state="readonly", width=6)
		self.cam_combo.set(cams[0]); self.cam_combo.grid(row=0,column=1,sticky="w")

		ttk.Label(frm, text="Resolution:").grid(row=0,column=2,sticky="w", padx=(12,0))
		res_strs = [f"{w}x{h}" for (w,h) in RES_LIST]
		if "640x480" in res_strs:
			res_strs.remove("640x480"); res_strs.insert(0,"640x480")
		self.res_combo = ttk.Combobox(frm, values=res_strs, state="readonly", width=12)
		self.res_combo.set("640x480"); self.res_combo.grid(row=0,column=3,sticky="w")

		# One-shot Start Camera button
		self.start_btn = ttk.Button(frm, text="Start Camera", command=self._start_camera)
		self.start_btn.grid(row=0, column=4, padx=8)

		self.manual_rec_var = tk.BooleanVar(value=False)
		self.manual_btn = ttk.Checkbutton(frm, text="Manual recording (clips)", variable=self.manual_rec_var, command=self._toggle_manual)
		self.manual_btn.grid(row=1, column=0, columnspan=2, sticky="w")

		self.classifier_var = tk.BooleanVar(value=True)
		self.classifier_chk = ttk.Checkbutton(frm, text="Enable classifier", variable=self.classifier_var, command=self._toggle_classifier)
		self.classifier_chk.grid(row=1, column=2, sticky="w")

		self.detect_rec_var = tk.BooleanVar(value=False)
		self.detect_chk = ttk.Checkbutton(frm, text="Record on detection (output)", variable=self.detect_rec_var, command=self._toggle_detect_recording)
		self.detect_chk.grid(row=2, column=0, columnspan=2, sticky="w")
		
		# Show detections in recording
		self.show_det_var = tk.BooleanVar(value=True)
		self.show_det_chk = ttk.Checkbutton(frm, text="Show detections in recordings", variable=self.show_det_var, command=self._toggle_show_detections)
		self.show_det_chk.grid(row=2, column=2, columnspan=2, sticky="w")	

		ttk.Label(frm, text="Display stream:").grid(row=3, column=0, sticky="w")
		self.display_var = tk.StringVar(value='static')
		rb_static = ttk.Radiobutton(frm, text="Static", variable=self.display_var, value='static', command=self._set_display)
		rb_motion = ttk.Radiobutton(frm, text="Motion", variable=self.display_var, value='motion', command=self._set_display)
		rb_disabled = ttk.Radiobutton(frm, text="Disabled", variable=self.display_var, value='disabled', command=self._set_display)
		rb_static.grid(row=3,column=1,sticky="w"); rb_motion.grid(row=3,column=2,sticky="w"); rb_disabled.grid(row=3,column=3,sticky="w")

		ttk.Label(frm, text="FPS:").grid(row=4,column=0,sticky="w")
		self.fps_label = ttk.Label(frm, text="0.0"); self.fps_label.grid(row=4,column=1,sticky="w")

		ttk.Label(frm, text="Throttle FPS (0 = off):").grid(row=5,column=0,sticky="w")
		self.throttle_var = tk.DoubleVar(value=0.0)
		self.throttle_spin = tk.Spinbox(frm, from_=0, to=120, increment=1, textvariable=self.throttle_var, width=6, command=self._on_throttle_change)
		self.throttle_spin.grid(row=5, column=1, sticky="w")
		self.throttle_spin.bind("<Return>", lambda e: self._on_throttle_change())

		self.quit_btn = ttk.Button(frm, text="Quit", command=self._quit)
		self.quit_btn.grid(row=6, column=4, sticky="e")

		self._update_gui()

	# ~ def _start_camera(self):
		# ~ # one-shot start: apply selection, start processor, then disable camera/res controls
		# ~ cam_index = int(self.cam_combo.get())
		# ~ self.processor.set_camera(cam_index)
		# ~ try:
			# ~ w,h = map(int, self.res_combo.get().split('x'))
			# ~ self.processor.set_resolution((w,h))
		# ~ except Exception:
			# ~ pass
		# ~ self.processor.start()
		# ~ # disable controls that could trigger re-start
		# ~ self.start_btn.config(text="Camera Started", state="disabled")
		# ~ self.cam_combo.config(state="disabled"); self.res_combo.config(state="disabled")

	def _start_camera(self):
		# one-shot start: apply selection, start processor, then disable camera/res controls
		cam_val = self.cam_combo.get()
		# cam_val will be string: either "0","1",... or "picamera"
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
		self.processor.start()
		# disable controls that could trigger re-start
		self.start_btn.config(text="Camera Started", state="disabled")
		self.cam_combo.config(state="disabled"); self.res_combo.config(state="disabled")
	

	def _toggle_manual(self):
		self.processor.toggle_manual_recording(self.manual_rec_var.get())

	def _toggle_classifier(self):
		self.processor.toggle_classifier(self.classifier_var.get())

	def _toggle_detect_recording(self):
		self.processor.toggle_detection_recording(self.detect_rec_var.get())

	def _toggle_show_detections(self):
		self.processor.set_show_detections_in_recording(self.show_det_var.get())


	def _set_display(self):
		self.processor.set_display_mode(self.display_var.get())

	def _on_throttle_change(self):
		val = self.throttle_var.get()
		try:
			fps_val = float(val)
		except:
			fps_val = 0.0
		if fps_val < 0: fps_val = 0.0
		self.processor.desired_fps = fps_val

	def _update_gui(self):
		fps = self.processor.latest_fps
		self.fps_label.config(text=f"{fps:.1f}")
		self.root.after(250, self._update_gui)

	def _quit(self):
		if messagebox.askokcancel("Quit", "Stop and exit?"):
			self.processor.stop()
			self.root.quit()

# -------------------------
# Main
# -------------------------
def main():
	processor = CameraProcessor()
	root = tk.Tk()
	gui = ControlGUI(root, processor)
	cams = scan_cameras(8)
	if cams:
		gui.cam_combo.set(cams[0]); processor.set_camera(cams[0])
	processor.set_resolution((640,480))
	root.protocol("WM_DELETE_WINDOW", gui._quit)
	root.mainloop()

if __name__ == '__main__':
	main()
