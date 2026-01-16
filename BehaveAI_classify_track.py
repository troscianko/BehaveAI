import cv2
import numpy as np
import csv
import os
import glob
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import configparser
import time
import shutil
import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
# ~ import config_watcher
import sys


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



# ---------- Project-aware configuration loading --------------------------

def pick_ini_via_dialog():
	root = tk.Tk()
	root.withdraw()
	path = filedialog.askopenfilename(
		title="Select BehaveAI settings INI",
		filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
	)
	root.destroy()
	return path

# Determine config_path (accept project dir or direct INI path)
if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	if os.path.isdir(arg):
		config_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		config_path = arg
else:
	config_path = pick_ini_via_dialog()
	if not config_path:
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
	tk.messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
	sys.exit(1)

# Set project directory to the INI parent and make it the working directory
project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
print(f"Working directory set to project dir: {project_dir}")
print(f"Using settings file: {config_path}")

# Load configuration
config = configparser.ConfigParser()
config.optionxform = str  # keep case
config.read(config_path)

# Helper: resolve a path from INI (absolute or relative to project_dir)
def resolve_project_path(value, fallback):
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	if os.path.isabs(value):
		return os.path.normpath(value)
	return os.path.normpath(os.path.join(project_dir, value))

# Read dataset / directory keys from INI (defaults are relative names inside the project)
clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
input_dir_ini = config['DEFAULT'].get('input_dir', 'input')
output_dir_ini = config['DEFAULT'].get('output_dir', 'output')

clips_dir = resolve_project_path(clips_dir_ini, 'clips')
input_folder = resolve_project_path(input_dir_ini, 'input')
output_folder = resolve_project_path(output_dir_ini, 'output')


# Read parameters
try:
	
	primary_motion_classes = [name.strip() for name in config['DEFAULT']['primary_motion_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('primary_motion_colors', '').split(';') if c.strip()]
	primary_motion_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	primary_motion_hotkeys = [key.strip() for key in config['DEFAULT']['primary_motion_hotkeys'].split(',')]
	
	secondary_motion_classes = [name.strip() for name in config['DEFAULT']['secondary_motion_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('secondary_motion_colors', '').split(';') if c.strip()]
	secondary_motion_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	secondary_motion_hotkeys = [key.strip() for key in config['DEFAULT']['secondary_motion_hotkeys'].split(',')]
	
	primary_static_classes = [name.strip() for name in config['DEFAULT']['primary_static_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('primary_static_colors', '').split(';') if c.strip()]
	primary_static_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	primary_static_hotkeys = [key.strip() for key in config['DEFAULT']['primary_static_hotkeys'].split(',')]
	
	secondary_static_classes = [name.strip() for name in config['DEFAULT']['secondary_static_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('secondary_static_colors', '').split(';') if c.strip()]
	secondary_static_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	secondary_static_hotkeys = [key.strip() for key in config['DEFAULT']['secondary_static_hotkeys'].split(',')]

	if len(secondary_motion_classes) >= 2 or len(secondary_static_classes) >= 2:
		hierarchical_mode = True
		motion_cropped_base_dir = 'annot_motion_crop'
		static_cropped_base_dir = 'annot_static_crop'
		
		# secondary classes need more than one value, so clear if there's only one value
		if len(secondary_motion_classes) == 1:
			secondary_motion_classes = []
			secondary_motion_colors = []
			secondary_motion_hotkeys = []
					
		if len(secondary_static_classes) == 1:
			secondary_static_classes = []
			secondary_static_colors = []
			secondary_static_hotkeys = []

	else: hierarchical_mode = False


	primary_classes = primary_static_classes + primary_motion_classes
	primary_colors = primary_static_colors + primary_motion_colors
	primary_hotkeys = primary_static_hotkeys + primary_motion_hotkeys
	
	secondary_classes = secondary_static_classes + secondary_motion_classes
	secondary_colors = secondary_static_colors + secondary_motion_colors
	secondary_hotkeys = secondary_static_hotkeys + secondary_motion_hotkeys

	primary_static_project_path = 'model_primary_static'
	primary_static_model_path = os.path.join('model_primary_static', "train", "weights", "best.pt")
	primary_static_yaml_path = 'static_annotations.yaml'
	
	primary_motion_project_path = 'model_primary_motion'
	primary_motion_model_path = os.path.join('model_primary_motion', "train", "weights", "best.pt")
	primary_motion_yaml_path = 'motion_annotations.yaml'
	
	ignore_secondary = [name.strip() for name in config['DEFAULT']['ignore_secondary'].split(',')]
	dominant_source = config['DEFAULT']['dominant_source'].lower()

	primary_classifier = config['DEFAULT'].get('primary_classifier', 'yolo11s.pt') 
	primary_epochs = int(config['DEFAULT'].get('primary_epochs', '50'))
	secondary_classifier = config['DEFAULT'].get('secondary_classifier', 'yolo11s-cls.pt')  
	secondary_epochs = int(config['DEFAULT'].get('secondary_epochs', '50'))

	if hierarchical_mode:
		secondary_static_project_path = 'model_secondary_static'
		secondary_static_data_path = 'annot_static_crop'
		secondary_static_model_path = os.path.join('model_secondary_static', "train", "weights", "best.pt")
		
		secondary_motion_project_path = 'model_secondary_motion'
		secondary_motion_data_path = 'annot_motion_crop'
		secondary_motion_model_path = os.path.join('model_secondary_motion', "train", "weights", "best.pt")
		
		secondary_class_ids = list(range(len(secondary_classes)))
		paired = list(zip(secondary_classes, secondary_colors, secondary_class_ids, secondary_hotkeys))
		paired_sorted = sorted(paired, key=lambda x: x[0].lower())
		secondary_classes, secondary_colors, secondary_class_ids, secondary_hotkeys = zip(*paired_sorted)
		# Convert back to lists
		secondary_classes = list(secondary_classes)
		secondary_colors = list(secondary_colors)	
		secondary_class_ids = list(secondary_class_ids)
		secondary_hotkeys = list(secondary_hotkeys)
	
	# Common parameters
	scale_factor = float(config['DEFAULT'].get('scale_factor', '1.0'))
	expA = float(config['DEFAULT'].get('expA', '0.5'))
	expB = float(config['DEFAULT'].get('expB', '0.8'))
	lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
	strategy = config['DEFAULT'].get('strategy', 'exponential')
	chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
	rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	use_ncnn = config['DEFAULT']['use_ncnn'].lower()
	primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))
	match_distance_thresh = float(config['DEFAULT'].get('match_distance_thresh', '200'))
	delete_after_missed = float(config['DEFAULT'].get('delete_after_missed', '5'))
	centroid_merge_thresh = float(config['DEFAULT'].get('centroid_merge_thresh', '50'))
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
	font_size = float(config['DEFAULT'].get('font_size', '0.5'))
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))

	process_noise_pos = float(config['kalman'].get('process_noise_pos', '0.01'))
	process_noise_vel = float(config['kalman'].get('process_noise_vel', '0.1'))
	measurement_noise = float(config['kalman'].get('measurement_noise', '0.1'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")


# Validate configuration

if len(primary_motion_classes) != len(primary_motion_colors) or len(primary_motion_classes) != len(primary_motion_hotkeys):
	raise ValueError("Primary motion classes, colors and hotkeys must match in configuration.")
if len(secondary_motion_classes) != len(secondary_motion_colors) or len(secondary_motion_classes) != len(secondary_motion_hotkeys):
	raise ValueError("Secondary motion classes, colors and hotkeys must match in configuration.")
if len(primary_static_classes) != len(primary_static_colors) or len(primary_static_classes) != len(primary_static_hotkeys):
	raise ValueError("Primary static classes, colors and hotkeys must match in configuration.")
if len(secondary_static_classes) != len(secondary_static_colors) or len(secondary_static_classes) != len(secondary_static_hotkeys):
	raise ValueError("Secondary static classes, colors and hotkeys must match in configuration.")
if dominant_source != 'motion' and dominant_source != 'static' and dominant_source != 'confidence':
	raise ValueError("dominant_source must be motion, static, or confidence")

if len(primary_static_classes) > 0:
	if not os.path.exists(primary_static_yaml_path):
		print(f"Error: Primary static YAML file not found. Run the Annotation script once to fix this")
		sys.exit(1)

if len(primary_motion_classes) > 0:
	if not os.path.exists(primary_motion_yaml_path):
		print(f"Error: Primary motion YAML file not found. Run the Annotation script once to fix this")
		sys.exit(1)


# ~ # check whether settings have been changed, and motion annotation library needs rebuilding 
# ~ settings_changed = config_watcher.check_settings_changed(current_config_path=config_path, saved_config_path=None, model_dirs=['model_primary_motion'])
# ~ # Globals for prompting/behaviour inside maybe_retrain
# ~ regen_prompt_shown = False
# ~ force_rebuild_motion = False


global_response = 0 # if 'yes' is selected for any model re-training, retraining should be perfoemd for all models

def count_images_in_dataset(path):
	## Count images in a dataset, handling both YAML-based and directory-based datasets
	# If path is a YAML file (primary models)
	if path.endswith('.yaml'):
		try:
			import yaml
			with open(path, 'r') as f:
				data = yaml.safe_load(f)
			
			# Get the path to the training images
			train_path = data['train']
			base_dir = os.path.dirname(path)
			abs_train_path = os.path.join(base_dir, train_path)
			
			# Handle different dataset formats
			if abs_train_path.endswith('.txt'):
				# Text file with image paths
				with open(abs_train_path, 'r') as f:
					return len(f.readlines())
			else:
				# Directory with images
				image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
				return len([f for f in os.listdir(abs_train_path) 
							if os.path.splitext(f)[1].lower() in image_exts])
		except Exception as e:
			print(f"Error counting images: {e}")
			return 0
	
	# If path is a directory (secondary models)
	elif os.path.isdir(path):
		total_count = 0
		image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
		
		# Walk through all subdirectories
		for root, dirs, files in os.walk(path):
			# Only count files in leaf directories (class directories)
			if not dirs:  # This is a leaf directory (no subdirectories)
				count = sum(1 for f in files 
						   if os.path.splitext(f)[1].lower() in image_exts)
				total_count += count
				
		return total_count
	
	else:
		print(f"Unsupported dataset format: {path}")
		return 0


def maybe_retrain(model_type, yaml_path, project_path, model_path, classifier, epochs, imgsz):
	"""
	Decide whether to (re)train a model based on existence and image counts.
	- If model_path exists and the recorded train_count differs from the current dataset,
	  prompt the user to retrain (Yes/No).
	- If model_path does not exist, perform first-time training.
	Returns True if a training run was performed, False otherwise.
	"""

	# Determine whether this is a motion model by naming
	is_motion_model = ('motion' in model_type.lower()) or ('secondary_motion' in project_path.lower()) or ('primary_motion' in project_path.lower())

	# If model exists: compare recorded image count (train_count.txt) with current dataset
	if os.path.exists(model_path):
		if os.path.exists(os.path.join(project_path, 'train_count.txt')):
			try:
				with open(os.path.join(project_path, 'train_count.txt'), 'r') as f:
					last_count = int(f.read().strip())
			except Exception:
				last_count = -1
		else:
			last_count = -1

		current_count = count_images_in_dataset(yaml_path)

		if current_count != last_count:
			# Ask user whether to retrain
			root = tk.Tk(); root.withdraw()
			msg = (
				f"New annotations detected for '{model_type}' model.\n"
				f"Image count changed from {last_count} to {current_count}.\n\n"
				"Do you want to re-train this model?"
			)
			response = messagebox.askyesno("Retrain model?", msg)
			root.destroy()

			if response:
				# Backup existing model dir/project and retrain from its weights
				backup_dir = project_path + "_backup"
				i = 1
				while os.path.exists(f"{backup_dir}{i}"):
					i += 1
				final_backup = f"{backup_dir}{i}"
				try:
					shutil.copytree(project_path, final_backup)
					print(f"Existing model copied to {final_backup}")
				except Exception as e:
					print(f"Warning: failed to backup {project_path}: {e}")

				start_weights = os.path.join(final_backup, "train", "weights", "best.pt")
				print(f'Training new {model_type} model using existing weights...')
				model = YOLO(start_weights)
				model.train(
					data=yaml_path,
					epochs=epochs,
					imgsz=imgsz,
					project=project_path,
					name="train",
					exist_ok=True
				)
				print(f'Done training {model_type} model')
				# Update saved train count
				with open(os.path.join(project_path, 'train_count.txt'), 'w') as f:
					f.write(str(current_count))
				# copy existing settings ini file for reference (so you know which settings were used for each model)
				os.makedirs(project_path, exist_ok=True)
				# ~ dst = os.path.join(project_path, os.path.basename(config_path))
				dst = os.path.join(project_path, 'saved_settings.ini')
				try:
					shutil.copy2(config_path, dst)
					print(f"Saved settings snapshot to {dst}")
				except Exception as e:
					print(f"Warning: could not copy settings to model dir: {e}")
				return True

		# else counts match -> nothing to do
		return False

	else:
		# Model missing -> do first-time training
		print(f'{model_type} model not found, building it...')
		model = YOLO(classifier)
		model.train(
			data=yaml_path,
			epochs=epochs,
			imgsz=imgsz,
			project=project_path,
			name="train",
			exist_ok=True
		)
		print(f'Done training {model_type} model')

		current_count = count_images_in_dataset(yaml_path)
		os.makedirs(project_path, exist_ok=True)
		with open(os.path.join(project_path, 'train_count.txt'), 'w') as f:
			f.write(str(current_count))

		# copy existing settings ini file for reference (so you know which settings were used for each model)
		os.makedirs(project_path, exist_ok=True)
		# ~ dst = os.path.join(project_path, os.path.basename(config_path))
		dst = os.path.join(project_path, 'saved_settings.ini')
		try:
			shutil.copy2(config_path, dst)
			print(f"Saved settings snapshot to {dst}")
		except Exception as e:
			print(f"Warning: could not copy settings to model dir: {e}")

		return True


# Train secondary classifiers for each static class
secondary_static_models = None
secondary_motion_models = None

if hierarchical_mode:
	secondary_static_models = {}
	static_class_map = [[None] * len(secondary_classes) for _ in range(len(primary_classes))]
	if len(secondary_static_classes) >= 2:
		for primary_class in primary_classes:
			idx = primary_classes.index(primary_class)
			hotkey = primary_hotkeys[idx]
			if hotkey in secondary_hotkeys: 
				continue
				
			if primary_class in ignore_secondary:
				continue
			
			data_dir = os.path.join(secondary_static_data_path, primary_class)
			# Skip if directory doesn't exist
			if not os.path.isdir(data_dir):
				continue
			
			# Create model directory for this static class
			model_dir = f"model_static_static_{primary_class}"
			weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
			
			maybe_retrain(model_dir, data_dir, model_dir, 
				weights_path, secondary_classifier, secondary_epochs, 224)

			# Load the trained model
			if use_ncnn == 'true':
				secondary_static_models[primary_class] = load_model_with_ncnn_preference(weights_path, "classify")
			else:
				secondary_static_models[primary_class] = YOLO(weights_path)


		# ~ print(f"secondary_static_models {secondary_static_models}")
		
	secondary_motion_models = {}
	motion_class_map = [[None] * len(secondary_classes) for _ in range(len(primary_classes))]
	if len(secondary_motion_classes) >= 2:
		for primary_class in primary_classes:
			idx = primary_classes.index(primary_class)
			hotkey = primary_hotkeys[idx]
			if hotkey in secondary_hotkeys: 
				continue
			
			if primary_class in ignore_secondary:
				continue			
			
			data_dir = os.path.join(secondary_motion_data_path, primary_class)
			# Skip if directory doesn't exist
			if not os.path.isdir(data_dir):
				continue
			
			# Create model directory for this static class
			model_dir = f"model_secondary_motion_{primary_class}"
			weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
			
			maybe_retrain(model_dir, data_dir, model_dir, 
				weights_path, secondary_classifier, secondary_epochs, 224)

			# Load the trained model
			if use_ncnn == 'true':
				secondary_motion_models[primary_class] = load_model_with_ncnn_preference(weights_path, "classify")
			else:
				secondary_motion_models[primary_class] = YOLO(weights_path)
				
		# ~ print(f"secondary_motion_models {secondary_motion_models}")

#-------CHECK PRIMARY MODEL EXISTS----------
if primary_static_classes[0] != '0':
	maybe_retrain('primary static', primary_static_yaml_path, primary_static_project_path, 
		primary_static_model_path, primary_classifier, primary_epochs, 640)


if primary_motion_classes[0] != '0':
	maybe_retrain('primary motion', primary_motion_yaml_path, primary_motion_project_path, 
		primary_motion_model_path, primary_classifier, primary_epochs, 640)


# --- PARAMETERS -----------------------------------------------------------

expA2 = 1 - expA
expB2 = 1 - expB

input_folder = "./input/"
output_folder = "./output/"

progress_update = 10 # print progress every n frames

def iou(box1, box2):
	xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
	xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
	inter = max(0, xb-xa) * max(0, yb-ya)
	area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
	area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
	prop1 = inter/area1 
	prop2 = inter/area2
	# return the larger proportional overlap - e.g. if one box is entirely inside another, this will return a 1.0, whereas the previous wouldn't
	if prop1 > prop2:
		return prop1 if prop1 > 0 else 0
	else:
		return prop2 if prop2 > 0 else 0
	# ~ union = area1 + area2 - inter
	# ~ return inter/union if union > 0 else 0
	

# --- TRACKER CLASS -------------------------------------------------------
class KalmanTracker:
	def __init__(self, dist_thresh, max_missed):
		self.next_id = 1
		self.tracks = {}  # tid -> {'kf': KalmanFilter, 'missed': int}
		self.prev_positions = {}  # Track previous positions
		self.dist_thresh = dist_thresh
		self.max_missed = max_missed

	def _create_kf(self, initial_pt):

		#Create a 4D state (x, y, vx, vy) Kalman Filter measuring (x, y).
		kf = cv2.KalmanFilter(4, 2)
		# State transition: x' = x + vx, y' = y + vy
		kf.transitionMatrix = np.array([[1, 0, 1, 0],
										[0, 1, 0, 1],
										[0, 0, 1, 0],
										[0, 0, 0, 1]], dtype=np.float32)
		# Measurement: we only observe x, y
		kf.measurementMatrix = np.array([[1, 0, 0, 0],
										 [0, 1, 0, 0]], dtype=np.float32)
		# Tune these covariances to your scene

		kf.processNoiseCov = np.diag([process_noise_pos, process_noise_pos, process_noise_vel, process_noise_vel]).astype(np.float32)	
		kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
		# Initialize state
		kf.statePre  = np.array([[initial_pt[0]],
								 [initial_pt[1]],
								 [0.],
								 [0.]], dtype=np.float32)
		kf.statePost = kf.statePre.copy()
		return kf
		
		self._prune_duplicate_tracks()

	def predict_all(self):
		"""
		Predict the next position for every track.
		Returns list of (tid, predicted_pt).
		"""
		preds = []
		for tid, tr in self.tracks.items():
			pred = tr['kf'].predict()
			preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
		return preds

		
		
	def _prune_duplicate_tracks(self):
		"""
		Merge any two tracks whose current posteriors are very close.
		Call this at the end of update().
		"""
		tids = list(self.tracks.keys())
		posts = {}
		for tid in tids:
			sp = self.tracks[tid]['kf'].statePost
			posts[tid] = (float(sp[0,0]), float(sp[1,0]))
		to_drop = set()
		for i, t1 in enumerate(tids):
			x1, y1 = posts[t1]
			for t2 in tids[i+1:]:
				x2, y2 = posts[t2]
				if np.hypot(x1-x2, y1-y2) < self.dist_thresh * 0.5:
					# mark the higher ID for deletion
					to_drop.add(max(t1, t2))
		for tid in to_drop:
			del self.tracks[tid]

	def update(self, detections):

		#detections: list of (x, y) centroids
		#Returns a dict: detection_index -> track_id

		# 1) Predict all tracks forward one step
		preds = self.predict_all()  # list of (tid, (px, py))
		track_ids = [t[0] for t in preds]
		pred_pts   = [t[1] for t in preds]

		# 2) Build cost matrix = Euclidean distance
		if pred_pts and detections:
			cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
			for i, p in enumerate(pred_pts):
				for j, d in enumerate(detections):
					cost[i, j] = np.hypot(p[0] - d[0], p[1] - d[1])
			row_idx, col_idx = linear_sum_assignment(cost)
		else:
			row_idx = np.array([], dtype=int)
			col_idx = np.array([], dtype=int)

		assigned_detects = {}
		matched_tracks = set()
		matched_dets   = set()

		# 3) Associate tracks ↔ detections
		for r, c in zip(row_idx, col_idx):
			if cost[r, c] < self.dist_thresh:
				tid = track_ids[r]
				matched_tracks.add(tid)
				matched_dets.add(c)
				assigned_detects[c] = tid

				# Get the measurement point
				dpt = detections[c]
				meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
				
				# Correct KF with the detection measurement
				self.tracks[tid]['kf'].correct(meas)
				self.tracks[tid]['missed'] = 0
				
				# Update previous position
				self.prev_positions[tid] = (dpt[0], dpt[1])

		# 4) Process unassigned detections
		for i, dpt in enumerate(detections):
			if i in matched_dets:
				continue
				
			# try to find an existing track under the threshold
			best_tid, best_dist = None, float('inf')
			for tid, (px, py) in preds:
				d = np.hypot(dpt[0]-px, dpt[1]-py)
				if d < best_dist:
					best_dist, best_tid = d, tid

			if best_dist < self.dist_thresh:
				assigned_detects[i] = best_tid
				self.tracks[best_tid]['missed'] = 0
				meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
				self.tracks[best_tid]['kf'].correct(meas)
				
				# Update previous position
				self.prev_positions[best_tid] = (dpt[0], dpt[1])
				matched_tracks.add(best_tid)  # Add to matched tracks

			else:
				# New track
				tid = self.next_id
				kf = self._create_kf(dpt)
				self.tracks[tid] = {'kf': kf, 'missed': 0}
				assigned_detects[i] = tid
				self.prev_positions[tid] = (dpt[0], dpt[1])  # Initialize position
				matched_tracks.add(tid)  # Add to matched tracks
				self.next_id += 1

		# 5) Handle unmatched tracks
		for tid in list(self.tracks.keys()):
			if tid not in matched_tracks:
				self.tracks[tid]['missed'] += 1
				# Increase uncertainty when missing detections
				noise_scale = min(2.0, 1.0 + self.tracks[tid]['missed'] * 0.2)
				
				# FIXED: Preserve matrix type and structure
				kf = self.tracks[tid]['kf']
				new_noise = kf.processNoiseCov.copy()
				new_noise *= noise_scale
				kf.processNoiseCov = new_noise
					
				# Remove track if missed too many times
				if self.tracks[tid]['missed'] > self.max_missed:
					del self.tracks[tid]
					if tid in self.prev_positions:
						del self.prev_positions[tid]

		return assigned_detects


# --- MAIN PROCESSING -----------------------------------------------------
def process_video(file):
	os.makedirs(output_folder, exist_ok=True)
	base = os.path.splitext(os.path.basename(file))[0]
	cap = cv2.VideoCapture(file)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if not cap.isOpened(): return
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale_factor)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale_factor)
	fps = cap.get(cv2.CAP_PROP_FPS)
	writer = cv2.VideoWriter(
		os.path.join(output_folder, base + "_detected.mp4"),
		cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
	)

	if primary_static_classes[0] != '0':
		if use_ncnn == 'true':
			model_static = load_model_with_ncnn_preference(primary_static_model_path, "detect")
		else:
			model_static = YOLO(primary_static_model_path)
		 
	if primary_motion_classes[0] != '0':
		if use_ncnn == 'true':
			model_motion = load_model_with_ncnn_preference(primary_motion_model_path, "detect")
		else:
			model_motion = YOLO(primary_motion_model_path)
		
		
	tracker = KalmanTracker(match_distance_thresh, delete_after_missed)

	prev_frames, frame_idx = None, 0
	csv_file = open(os.path.join(output_folder, base + "_tracking.csv"), 'w', newline='')
	csv_writer = csv.writer(csv_file)
	# Updated CSV header with four streams
	csv_writer.writerow([
		"frame", "id", "x", "y",
		"primary_static_class", "primary_static_conf",
		"primary_motion_class", "primary_motion_conf",
		"secondary_static_class", "secondary_static_conf",
		"secondary_motion_class", "secondary_motion_conf"
	])

	print(f"Processing video: {file}")
	print('Initialising')
	current_frame = 0
	print_tick = 0
	start_time = time.time()

	frame_count = 0
	
	while True:
		ret, raw_frame = cap.read()
		if not ret: break
		frame_idx += 1
		if frame_count == 0:
			if scale_factor != 1.0:
				raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)
			gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
			frame = raw_frame.copy()
			if prev_frames is None:
				prev_frames = [gray.copy() for _ in range(3)]
				continue
			
			# only process motion information if necessary
			if primary_motion_classes[0] != '0':
	
				diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
				
				if strategy == 'exponential':
					prev_frames[0] = gray
					prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, expA2, 0)
					prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, expB2, 0)
				elif strategy == 'sequential':
					prev_frames[2] = prev_frames[1]
					prev_frames[1] = prev_frames[0]
					prev_frames[0] = gray


				if chromatic_tail_only == 'true':
					tb = cv2.subtract(diffs[0], diffs[1])	
					tr = cv2.subtract(diffs[2], diffs[1])
					tg = cv2.subtract(diffs[1], diffs[0])
							
					blue = cv2.addWeighted(gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
					red = cv2.addWeighted(gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
				else:
					blue = cv2.addWeighted(gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
					red = cv2.addWeighted(gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)

				motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
	
			# Collect all primary detections
			all_detections = []
			
			# Primary static detection
			if primary_static_classes[0] != '0':
				results_static = model_static.predict(frame, conf=primary_conf_thresh, verbose=False)
				for box in results_static[0].boxes:
					coords = tuple(map(int, box.xyxy[0].tolist()))
					class_idx = int(box.cls[0])
					class_name = primary_static_classes[class_idx]
					conf = float(box.conf[0])
					all_detections.append({
						'coords': coords,
						'primary_class': class_name,
						'primary_conf': conf,
						'source': 'static',
						'primary_class_combined': '',
						'primary_conf_combined': 0.0
					})
					
			# Primary motion detection
			if primary_motion_classes[0] != '0':
				results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
				for box in results_motion[0].boxes:
					coords = tuple(map(int, box.xyxy[0].tolist()))
					class_idx = int(box.cls[0])
					class_name = primary_motion_classes[class_idx]
					conf = float(box.conf[0])
					all_detections.append({
						'coords': coords,
						'primary_class': class_name,
						'primary_conf': conf,
						'source': 'motion',
						'primary_class_combined': '',
						'primary_conf_combined': 0.0
					})
	
	
			# Merge detections based on proximity
			merged_detections = []
			for det in all_detections:
				x1, y1, x2, y2 = det['coords']
				cx, cy = (x1+x2)//2, (y1+y2)//2
				
				# Find matching existing detection
				matched = False
				for md in merged_detections:
					md_cx, md_cy = md['centroid']
					dist = np.hypot(cx - md_cx, cy - md_cy)
					
					# Calculate IOU
					md_x1, md_y1, md_x2, md_y2 = md['coords']
					overlap = iou((x1, y1, x2, y2), (md_x1, md_y1, md_x2, md_y2))
					ms_source = md['source']
					
					if dist < centroid_merge_thresh or overlap > iou_thresh:
						# Merge classes - keep highest confidence detection for each source
						if det['source'] == ms_source or dominant_source == 'confidence': # mathcing sources so select best, or confidence strategy used
							if det['source'] == 'static':
								# Keep highest confidence static detection
								if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
									md['primary_class_combined'] = md['primary_class'] # retain the combined primary
									md['primary_conf_combined'] = md['primary_conf']
									md['primary_class'] = det['primary_class']
									md['primary_conf'] = det['primary_conf']
									md['coords'] = det['coords']  # Update to higher conf box
									md['centroid'] = (cx, cy)
									md['source'] = det['source']
							else:  # motion source
								# Keep highest confidence motion detection
								if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
									md['primary_class_combined'] = md['primary_class'] # retain the combined primary
									md['primary_conf_combined'] = md['primary_conf']
									md['primary_class'] = det['primary_class']
									md['primary_conf'] = det['primary_conf']
									md['coords'] = det['coords']  # Update to higher conf box
									md['centroid'] = (cx, cy)
									md['source'] = det['source']
						elif det['source'] == 'static' and dominant_source == 'static':
								# Keep static detection
								md['primary_class_combined'] = md['primary_class'] # retain the combined primary
								md['primary_conf_combined'] = md['primary_conf']
								md['primary_class'] = det['primary_class']
								md['primary_conf'] = det['primary_conf']
								md['coords'] = det['coords']  # Update to higher conf box
								md['centroid'] = (cx, cy)
								md['source'] = det['source']
						elif det['source'] == 'motion' and dominant_source == 'motion':
								# Keep motion detection
								md['primary_class_combined'] = md['primary_class'] # retain the combined primary
								md['primary_conf_combined'] = md['primary_conf']
								md['primary_class'] = det['primary_class']
								md['primary_conf'] = det['primary_conf']
								md['coords'] = det['coords']  # Update to higher conf box
								md['centroid'] = (cx, cy)
								md['source'] = det['source']
	
							
						matched = True
						break
						
				if not matched:
					# Add as new detection
					new_det = {
						'coords': det['coords'],
						'centroid': (cx, cy),
						'source': det['source'],
						'primary_class_combined': '',
						'primary_conf_combined': 0.0
					}
					if det['source'] == 'static':
						new_det['primary_class'] = det['primary_class']
						new_det['primary_conf'] = det['primary_conf']
						# ~ if 'secondary_static_class' in det:
							# ~ new_det['secondary_static_class'] = det['secondary_static_class']
							# ~ new_det['secondary_static_conf'] = det['secondary_static_conf']
					else:  # motion source
						new_det['primary_class'] = det['primary_class']
						new_det['primary_conf'] = det['primary_conf']
						# ~ if 'secondary_motion_class' in det:
							# ~ new_det['secondary_motion_class'] = det['secondary_motion_class']
							# ~ new_det['secondary_motion_conf'] = det['secondary_motion_conf']
					merged_detections.append(new_det)
	
	
	
			# Run secondary classification on each primary detection
			processed_detections = []
			for det in merged_detections:
				coords = det['coords']
				primary_class = det['primary_class']
				primary_conf = det['primary_conf']
				source = det['source']
				primary_class_combined = det['primary_class_combined']
				primary_conf_combined = det['primary_conf_combined']
	
				
				if source == 'static':
					det['primary_static_class'] = primary_class
					det['primary_static_conf'] = primary_conf
					det['primary_motion_class'] = primary_class_combined
					det['primary_motion_conf'] = primary_conf_combined
				else:
					det['primary_motion_class'] = primary_class
					det['primary_motion_conf'] = primary_conf
					det['primary_static_class'] = primary_class_combined
					det['primary_static_conf'] = primary_conf_combined
	
				if hierarchical_mode:
					x1, y1, x2, y2 = coords
					
					# Determine which secondary model to use based on source and configuration
					sec_model = None
					sec_classes = []
					crop_img = None
					
					if source == 'static':
						# Use static secondary model if configured
						if len(secondary_static_classes) >= 2:
							sec_model = secondary_static_models.get(primary_class, None)
							sec_classes = secondary_static_classes
							crop_img = frame
						# Fallback to motion secondary model if static not available
						elif len(secondary_motion_classes) >= 2:
							sec_model = secondary_motion_models.get(primary_class, None)
							sec_classes = secondary_motion_classes
							crop_img = motion_image if primary_motion_classes[0] != '0' else frame
					else:  # motion source
						# Use motion secondary model if configured
						if len(secondary_motion_classes) >= 2:
							sec_model = secondary_motion_models.get(primary_class, None)
							sec_classes = secondary_motion_classes
							crop_img = motion_image
						# Fallback to static secondary model if motion not available
						elif len(secondary_static_classes) >= 2:
							sec_model = secondary_static_models.get(primary_class, None)
							sec_classes = secondary_static_classes
							crop_img = frame
					
					# Get the cropped region
					crop = None
					if crop_img is not None:
						crop = crop_img[y1:y2, x1:x2]
					
					secondary_class = primary_class
					secondary_conf = 1.0
					
					# Run secondary classification if we have a model and valid crop
					if sec_model and crop is not None and crop.size > 0:
						sec_results = sec_model.predict(crop, verbose=False)
						if sec_results[0].probs is not None:
							secondary_class_idx = sec_results[0].probs.top1
							secondary_conf = sec_results[0].probs.top1conf.item()
							secondary_class = sec_model.names[secondary_class_idx]
	
					# Add secondary results to detection
					if source == 'static':
						det['secondary_static_class'] = secondary_class
						det['secondary_static_conf'] = secondary_conf
					else:  # motion source
						det['secondary_motion_class'] = secondary_class
						det['secondary_motion_conf'] = secondary_conf
					
				processed_detections.append(det)
	
	
			# Prepare for tracking
			cents = [d['centroid'] for d in processed_detections]
			assignment = tracker.update(cents)
	
			# ~ frame = motion_image ## enable this line ot save the motion video instead of static 
	
			# Process tracked objects
			for idx, det in enumerate(processed_detections):
				tid = assignment.get(idx, None)
				if tid is None or tid not in tracker.tracks:
					continue
					
				x1, y1, x2, y2 = det['coords']
				cx, cy = det['centroid']
	
				# Get all class info with default values
				ps_class = det.get('primary_static_class', '')
				ps_conf = det.get('primary_static_conf', 0)
				pm_class = det.get('primary_motion_class', '')
				pm_conf = det.get('primary_motion_conf', 0)
				ss_class = det.get('secondary_static_class', '')
				ss_conf = det.get('secondary_static_conf', 0)
				sm_class = det.get('secondary_motion_class', '')
				sm_conf = det.get('secondary_motion_conf', 0)
				p_source = det.get('source', '')
	
	
				# Create display label
				label_parts = []
				# ~ if ps_class: 
				if p_source == 'static': 
					label_parts.append(f"{ps_class.upper()}")
					primary_cls = ps_class
				else:
					label_parts.append(f"{pm_class.upper()}")
					primary_cls = pm_class
				
				primary_col = primary_colors[primary_classes.index(primary_cls)]
				secondary_col = (255, 255, 255)
	
				if hierarchical_mode:
					
					if sm_class != '' and sm_class != primary_cls:
						secondary_cls = sm_class
						secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
					if ss_class != '' and ss_class != primary_cls:
						secondary_cls = ss_class
						secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
					
	
					if primary_cls in ignore_secondary:
						label = f"{tid} {primary_cls.upper()}"
						label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
						label_w, label_h = label_size
						cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
						cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, line_thickness)
						cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
									font_size, primary_col, line_thickness, cv2.LINE_AA)
					else:
						# Draw outer static box (slightly larger)
						outer_thickness = line_thickness + 2
						cv2.rectangle(frame, (x1-outer_thickness, y1-outer_thickness), 
									 (x2+outer_thickness, y2+outer_thickness), 
									primary_col, outer_thickness)
						label = f"{tid} {primary_cls.upper()} {secondary_cls}"
						label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
						label_w, label_h = label_size
						cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
						cv2.rectangle(frame, (x1, y1), (x2, y2), secondary_col, line_thickness)
						cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
									font_size, secondary_col, line_thickness, cv2.LINE_AA)
				else:
					label = f"{tid} {primary_cls}"
					label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
					label_w, label_h = label_size
					cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
					cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, line_thickness)
					cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, 
								font_size, primary_col, line_thickness, cv2.LINE_AA)
	
	
	
		
				# Draw motion vector (if tracking available)
				if tid in tracker.tracks:
					state_post = tracker.tracks[tid]['kf'].statePost
					x, y = state_post[0, 0], state_post[1, 0]
					vx, vy = state_post[2, 0], state_post[3, 0]
					next_x = x + vx
					next_y = y + vy
					
					light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
					cv2.line(frame, (int(x), int(y)), (int(next_x), int(next_y)), primary_col, line_thickness)
					cv2.circle(frame, (int(next_x), int(next_y)), 3, light_color, -line_thickness)
					cv2.circle(frame, (int(cx), int(cy)), 3, primary_col, -line_thickness)
				
				# Write to CSV
				csv_writer.writerow([
					frame_idx, tid, cx, cy,
					ps_class, f"{ps_conf:.3f}",
					pm_class, f"{pm_conf:.3f}",
					ss_class, f"{ss_conf:.3f}",
					sm_class, f"{sm_conf:.3f}"
				])


			# ~ # print frame number
			text_color = (255, 255, 255)  # white text
			label = str(current_frame)
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			cv2.rectangle(frame, (0, 0), 
						 (label_w + line_thickness*4, label_h + line_thickness*4), (0, 0, 0), -1)
			cv2.putText(frame, label, (line_thickness*2, label_h + line_thickness*2), 
					   cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, line_thickness)
						   
			writer.write(frame)
			
			if print_tick > progress_update:
				elapsed = time.time() - start_time
				current_fps = current_frame / elapsed if elapsed > 0 else 0
				pc_done = 100 * (frame_skip+1) * current_frame / total_frames
				print(f"Progress: {pc_done:.2f}% | {current_fps:.1f} FPS", end='\r', flush=True)
				print_tick = 0
			current_frame += 1
			print_tick += 1

		frame_count += 1
		
		if frame_count > frame_skip:
			frame_count = 0

	cap.release()
	writer.release()
	csv_file.close()
	print(f"Done processing {base} | {current_fps:.1f} FPS")

if __name__ == '__main__':
	for vid in glob.glob(os.path.join(input_folder, "*.*")):
		process_video(vid)
