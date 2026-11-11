#!/usr/bin/env python3


import os
import sys
import time
import traceback
import cv2
import numpy as np
import configparser
import random
from collections import deque

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from index_annotations import AnnotationIndex
import box_geometry as bg


# Try to import YOLO
try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

#--- Configuration parsing ----------
def choose_ini_path_from_dialog():
	root = tk.Tk(); root.withdraw()
	ini_path = filedialog.askopenfilename(
		title="Select BehaveAI settings INI",
		filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
	)
	root.destroy()
	return ini_path

if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	if os.path.isdir(arg):
		config_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		config_path = arg
else:
	config_path = choose_ini_path_from_dialog()
	if not config_path:
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
	try:
		root = tk.Tk(); root.withdraw()
		messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
		root.destroy()
	except Exception:
		print(f"Configuration file not found: {config_path}")
	sys.exit(1)

project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_path)

def resolve_project_path(value, fallback):
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	if os.path.isabs(value):
		return os.path.normpath(value)
	return os.path.normpath(os.path.join(project_dir, value))

clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
clips_dir = resolve_project_path(clips_dir_ini, 'clips')




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

	primary_static_project_path = 'model_primary_static'
	primary_static_model_path = os.path.join('model_primary_static', "train", "weights", "best.pt")
	primary_static_yaml_path = 'static_annotations.yaml'
	
	primary_motion_project_path = 'model_primary_motion'
	primary_motion_model_path = os.path.join('model_primary_motion', "train", "weights", "best.pt")
	primary_motion_yaml_path = 'motion_annotations.yaml'
	
	ignore_secondary = [name.strip() for name in config['DEFAULT']['ignore_secondary'].split(',')]
	dominant_source = config['DEFAULT']['dominant_source'].lower()


	motion_cropped_base_dir = 'annot_motion_crop'
	static_cropped_base_dir = 'annot_static_crop'
	motion_ornt_base_dir = 'annot_motion_ornt'
	static_ornt_base_dir = 'annot_static_ornt'

	oriented = config['DEFAULT'].get('box_shape', 'boxes').lower() == 'oriented'
	disambiguate_head_tail = oriented and config['DEFAULT'].get('disambiguate_head_tail', 'false').lower() == 'true'

	if len(secondary_motion_classes) >= 2 or len(secondary_static_classes) >= 2:
		hierarchical_mode = True
		
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

			
	static_train_images_dir = 'annot_static/images/train'
	static_val_images_dir = 'annot_static/images/val'
	static_train_labels_dir = 'annot_static/labels/train'
	static_val_labels_dir = 'annot_static/labels/val'
		
	motion_train_images_dir = 'annot_motion/images/train'
	motion_val_images_dir = 'annot_motion/images/val'
	motion_train_labels_dir = 'annot_motion/labels/train'
	motion_val_labels_dir = 'annot_motion/labels/val'
	
	# Common parameters
	scale_factor = float(config['DEFAULT'].get('scale_factor', '1.0'))
	expA = float(config['DEFAULT'].get('expA', '0.5'))
	expB = float(config['DEFAULT'].get('expB', '0.8'))
	val_frequency = float(config['DEFAULT'].get('val_frequency', '0.1'))

	lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
	strategy = config['DEFAULT'].get('strategy', 'exponential')
	chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
	primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))
	rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
	font_size = float(config['DEFAULT'].get('font_size', '0.5'))
	# ~ cross_blocking = config['DEFAULT']['cross_blocking'].lower()
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	centroid_merge_ratio = float(config['DEFAULT'].get('centroid_merge_ratio', '0.7'))
	motion_blocks_static = config['DEFAULT']['motion_blocks_static'].lower()
	static_blocks_motion = config['DEFAULT']['static_blocks_motion'].lower()
	save_empty_frames = config['DEFAULT']['save_empty_frames'].lower()
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))
	
except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")



if motion_blocks_static not in ('true', 'false'):
	raise ValueError("motion_blocks_static must be 'true' or 'false'")
if static_blocks_motion not in ('true', 'false'):
	raise ValueError("static_blocks_motion must be 'true' or 'false'")
if save_empty_frames not in ('true', 'false'):
	raise ValueError("save_empty_frames must be 'true' or 'false'")

primary_classes_info = list(zip(primary_hotkeys, primary_classes))
secondary_classes_info = list(zip(secondary_hotkeys, secondary_classes))
primary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(primary_classes_info)}
secondary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(secondary_classes_info)}

# initial selections - default to the first enabled ('0' = disabled placeholder) primary class,
# not just "index 1 whenever static has <=1 entries" (that conflated "static disabled" with
# "static has exactly one real class", leaving active_primary pointing at a disabled placeholder).
active_primary = next((i for i, name in enumerate(primary_classes) if name != '0'), 0)
active_secondary = 0


annotation_index = AnnotationIndex(
	static_train_images_dir,
	static_val_images_dir,
	static_train_labels_dir,
	static_val_labels_dir,
	motion_train_images_dir,
	motion_val_images_dir,
	motion_train_labels_dir,
	motion_val_labels_dir,
	motion_cropped_base_dir,
	static_cropped_base_dir,
	clips_dir,
	primary_static_classes,
	primary_classes,
	secondary_classes,
	hierarchical_mode,
	ignore_secondary=ignore_secondary,
	oriented=oriented
)

items = annotation_index.list_images_labels_and_masks()

# Build quick lookup: video_label -> set(of frame numbers that have annotations)
def build_annot_index_map(items_list):
	m = {}
	for it in items_list:
		base = it.get('basename', '')
		if '_' not in base:
			continue
		vlabel, tail = base.rsplit('_', 1)
		try:
			frm = int(tail)
		except Exception:
			continue
		m.setdefault(vlabel, set()).add(frm)
	return m

# initial annotated frames map (used to draw ticks on seek)
annotated_frames_map = build_annot_index_map(items)




# Open video
root_tmp = tk.Tk(); root_tmp.withdraw()
initial_dir = clips_dir if os.path.isdir(clips_dir) else os.getcwd()
video_path = filedialog.askopenfilename(title="Select video file", initialdir=initial_dir)
root_tmp.destroy()
if not video_path:
	print("No video selected. Exiting.")
	sys.exit(0)




capture = cv2.VideoCapture(video_path)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# frameWindow logic
right_frame_width = max(96, int(video_height / 3))
frameWindow = 4
if strategy == 'exponential':
	if expA > 0.2 or expB > 0.2:
		frameWindow = 5
	if expA > 0.5 or expB > 0.5:
		frameWindow = 10
	if expA > 0.7 or expB > 0.7:
		frameWindow = 15
	if expA > 0.8 or expB > 0.8:
		frameWindow = 20
	if expA > 0.9 or expB > 0.9:
		frameWindow = 45

raw_buf = deque(maxlen=frameWindow)
frameWindow = frameWindow * (frame_skip + 1)
frame_number = min(max(frameWindow - 1, 0), total_frames - 1)
frame_updated = True

# state
boxes = []
grey_boxes = []
original_frame = None
fr = None
motion_image = None

video_label = os.path.splitext(os.path.basename(video_path))[0]

bottom_bar_height = int(20 + font_size * 20)
grey_mode = False
annot_count = 1
auto_ann_switch = 1
show_mode = 1  # 1 = motion false color, -1 = static RGB
zoom_hide = 0
disp_scale_factor = 1.0

last_mouse_move = 0.0
ANIM_STILL_THRESHOLD = 0.5
ANIM_FPS = 8
last_anim_draw = 0.0
ANIM_DT = 1.0 / ANIM_FPS

# Load models
model_static = None
model_motion = None
if YOLO is not None:
	if os.path.exists(primary_static_model_path):
		try:
			model_static = YOLO(primary_static_model_path)
		except Exception as e:
			print("Failed to load primary static model:", e)
	if os.path.exists(primary_motion_model_path):
		try:
			model_motion = YOLO(primary_motion_model_path)
		except Exception as e:
			print("Failed to load primary motion model:", e)

secondary_static_models = {}
secondary_motion_models = {}

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
			if not os.path.isdir(data_dir):
				continue
			
			# Create model directory for this static class
			model_dir = f"model_secondary_static_{primary_class}"
			weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
			
			# Check if model exists
			if not os.path.exists(weights_path):
				print(f'Secondary static model for "{primary_class}" not found')
				# ~ secondary_motion_models[primary_class] = '0'
			else:
				print(f'Secondary static model for "{primary_class}" found')
				# Load the trained model
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
			if not os.path.isdir(data_dir):
				continue
			
			disk_classes = sorted(os.listdir(data_dir))
			
			# Create model directory for this static class
			model_dir = f"model_secondary_motion_{primary_class}"
			weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
			
			# Check if model exists
			if not os.path.exists(weights_path):
				print(f'Secondary motion model for "{primary_class}" not found')
				# ~ secondary_motion_models[primary_class] = '0'
			else:
				print(f'Secondary motion model for "{primary_class}" found')
				# Load the trained model
				secondary_motion_models[primary_class] = YOLO(weights_path)
				# ~ motion_model_count += 1
				
		# ~ print(f"secondary_motion_models {secondary_motion_models}")

ornt_static_models = {}
ornt_motion_models = {}

if disambiguate_head_tail:
	for primary_class in primary_classes:
		data_dir = os.path.join(static_ornt_base_dir, primary_class)
		if os.path.isdir(data_dir):
			weights_path = os.path.join(f"model_ornt_static_{primary_class}", "train", "weights", "best.pt")
			if os.path.exists(weights_path):
				try:
					ornt_static_models[primary_class] = YOLO(weights_path)
				except Exception as e:
					print(f"Failed to load head/tail static model for '{primary_class}':", e)

		data_dir = os.path.join(motion_ornt_base_dir, primary_class)
		if os.path.isdir(data_dir):
			weights_path = os.path.join(f"model_ornt_motion_{primary_class}", "train", "weights", "best.pt")
			if os.path.exists(weights_path):
				try:
					ornt_motion_models[primary_class] = YOLO(weights_path)
				except Exception as e:
					print(f"Failed to load head/tail motion model for '{primary_class}':", e)


# Helper: convert BGR -> PhotoImage

def cv2_to_photoimage(bgr_img):
	rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	pil = Image.fromarray(rgb)
	return ImageTk.PhotoImage(pil)


# ------------------------------
# Load saved labels for a given base (video_label_frame)
# ------------------------------
def norm_to_pixels(xc, yc, bw, bh, w, h):
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p/2); y1 = int(cy - bh_p/2)
	x2 = int(cx + bw_p/2); y2 = int(cy + bh_p/2)
	x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1)); x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
	return x1, y1, x2, y2


def _orient_conf(model, img):
	"""Probability the given whole crop is in the 'correct' (head-left, tail-right) orientation,
	per the per-primary-class orientation classifier."""
	if img is None or img.size == 0:
		return 0.0
	res = model.predict(img, verbose=False)
	probs = res[0].probs
	if probs is None:
		return 0.0
	correct_idx = None
	for idx, name in model.names.items():
		if name == 'correct':
			correct_idx = idx
			break
	if correct_idx is None:
		return 0.0
	return float(probs.data[correct_idx])


def _disambiguate_head_tail(crop_img, head, tail, width, primary_class, is_motion):
	"""Resolve both the mod-180 direction ambiguity AND which axis is actually the head-tail
	axis of a raw OBB detection (its longer dimension isn't necessarily the true head-tail axis -
	e.g. a wide-bodied animal can be wider than it is long) using the trained per-primary-class
	whole-crop orientation classifier. Falls back to the (arbitrary) input head/tail/width when
	disambiguation is off/unavailable. Returns (head, tail, width)."""
	if not disambiguate_head_tail:
		return head, tail, width
	model = (ornt_motion_models if is_motion else ornt_static_models).get(primary_class)
	if model is None:
		return head, tail, width
	return bg.best_head_tail_orientation(crop_img, head, tail, width, lambda img: _orient_conf(model, img))


def _attach_secondary_for_auto_annotate(box, primary_class, is_motion):
	sec_model = None
	crop_img = None
	if is_motion:
		if len(secondary_motion_classes) >= 2:
			sec_model = secondary_motion_models.get(primary_class)
			crop_img = motion_image
		elif len(secondary_static_classes) >= 2:
			sec_model = secondary_static_models.get(primary_class)
			crop_img = fr
	else:
		if len(secondary_static_classes) >= 2:
			sec_model = secondary_static_models.get(primary_class)
			crop_img = fr
		elif len(secondary_motion_classes) >= 2:
			sec_model = secondary_motion_models.get(primary_class)
			crop_img = motion_image if primary_motion_classes[0] != '0' else fr

	secondary_class_idx = -1
	secondary_conf = -1.0
	if sec_model is not None and crop_img is not None:
		corners = bg.corners_from_head_tail_width(box['head'], box['tail'], box['width'])
		crop = bg.crop_black_masked(crop_img, corners)
		if crop is not None and crop.size > 0:
			sec_results = sec_model.predict(crop, verbose=False)
			if sec_results[0].probs is not None:
				secondary_class_idx = sec_results[0].probs.top1
				secondary_conf = sec_results[0].probs.top1conf.item()
	box['sec_cls'] = secondary_class_idx
	box['sec_conf'] = secondary_conf
	return box


def _auto_annotate_local_oriented():
	global boxes
	all_detections = []

	if primary_static_classes[0] != '0' and model_static is not None:
		results_static = model_static.predict(fr, conf=primary_conf_thresh, verbose=False)
		if results_static[0].obb is not None:
			for obb_box in results_static[0].obb:
				class_idx = int(obb_box.cls[0])
				primary_class = primary_static_classes[class_idx]
				conf = float(obb_box.conf[0])
				coords = tuple(map(int, obb_box.xyxy[0].tolist()))
				cx, cy, bw, bh, angle = [float(v) for v in obb_box.xywhr[0]]
				head, tail, width = bg.head_tail_width_from_xywhr_rad(cx, cy, bw, bh, angle)
				head, tail, width = _disambiguate_head_tail(fr, head, tail, width, primary_class, is_motion=False)
				all_detections.append({'coords': coords, 'primary_class': primary_class, 'primary_conf': conf,
										'source': 'static', 'head': head, 'tail': tail, 'width': width})

	if primary_motion_classes[0] != '0' and model_motion is not None:
		results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
		if results_motion[0].obb is not None:
			for obb_box in results_motion[0].obb:
				class_idx = int(obb_box.cls[0])
				primary_class = primary_motion_classes[class_idx]
				conf = float(obb_box.conf[0])
				coords = tuple(map(int, obb_box.xyxy[0].tolist()))
				cx, cy, bw, bh, angle = [float(v) for v in obb_box.xywhr[0]]
				head, tail, width = bg.head_tail_width_from_xywhr_rad(cx, cy, bw, bh, angle)
				head, tail, width = _disambiguate_head_tail(motion_image, head, tail, width, primary_class, is_motion=True)
				all_detections.append({'coords': coords, 'primary_class': primary_class, 'primary_conf': conf,
										'source': 'motion', 'head': head, 'tail': tail, 'width': width})

	# Merge static/motion detections of the same physical object, respecting dominant_source -
	# shared with BehaveAI_classify_track.py/BehaveAI_live.py (box_geometry.py), matching by the
	# OBB's axis-aligned envelope while carrying the true rotated geometry through unchanged
	merged = bg.merge_source_detections(all_detections, centroid_merge_ratio, iou_thresh, dominant_source)

	new_boxes = []
	for det in merged:
		primary_class = det['primary_class']
		source = det['source']
		if source == 'static':
			cls = primary_static_classes.index(primary_class)
		else:
			cls = len(primary_static_classes) + primary_motion_classes.index(primary_class)
		box = {'head': det['head'], 'tail': det['tail'], 'width': det['width'], 'cls': cls, 'conf': det['primary_conf']}
		if hierarchical_mode:
			box = _attach_secondary_for_auto_annotate(box, primary_class, is_motion=(source == 'motion'))
		new_boxes.append(box)

	boxes = new_boxes


# Auto-annotate: uses model_static / model_motion and per-primary secondary models
def auto_annotate_local():
	global boxes
	if oriented:
		_auto_annotate_local_oriented()
		return

	all_detections = []

	# Primary static detection
	if primary_static_classes[0] != '0' and model_static is not None:
		results_static = model_static.predict(fr, conf=primary_conf_thresh, verbose=False)
		for box in results_static[0].boxes:
			class_idx = int(box.cls[0])
			primary_class = primary_static_classes[class_idx]
			conf = float(box.conf[0])
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			all_detections.append({'coords': (x1, y1, x2, y2), 'primary_class': primary_class,
									'primary_conf': conf, 'source': 'static'})

	# Primary motion detection
	if primary_motion_classes[0] != '0' and model_motion is not None:
		results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
		for box in results_motion[0].boxes:
			class_idx = int(box.cls[0])
			primary_class = primary_motion_classes[class_idx]
			conf = float(box.conf[0])
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			all_detections.append({'coords': (x1, y1, x2, y2), 'primary_class': primary_class,
									'primary_conf': conf, 'source': 'motion'})

	# Merge static/motion detections of the same physical object, respecting dominant_source -
	# shared with BehaveAI_classify_track.py/BehaveAI_live.py (box_geometry.py) so auto-annotate
	# resolves conflicting static/motion detections the same way batch processing does
	merged = bg.merge_source_detections(all_detections, centroid_merge_ratio, iou_thresh, dominant_source)

	new_boxes = []
	for det in merged:
		x1, y1, x2, y2 = det['coords']
		primary_class = det['primary_class']
		conf = det['primary_conf']
		source = det['source']
		if source == 'static':
			class_idx = primary_static_classes.index(primary_class)
		else:
			class_idx = len(primary_static_classes) + primary_motion_classes.index(primary_class)

		if hierarchical_mode:
			# prefer the secondary model matching this detection's own source, falling back to
			# the other stream if it isn't configured - matches BehaveAI_classify_track.py
			sec_model = None
			crop_img = None
			if source == 'static':
				if len(secondary_static_classes) >= 2:
					sec_model = secondary_static_models.get(primary_class, None)
					crop_img = fr
				elif len(secondary_motion_classes) >= 2:
					sec_model = secondary_motion_models.get(primary_class, None)
					crop_img = motion_image if primary_motion_classes[0] != '0' else fr
			else:
				if len(secondary_motion_classes) >= 2:
					sec_model = secondary_motion_models.get(primary_class, None)
					crop_img = motion_image
				elif len(secondary_static_classes) >= 2:
					sec_model = secondary_static_models.get(primary_class, None)
					crop_img = fr

			crop = crop_img[y1:y2, x1:x2] if crop_img is not None else None

			secondary_conf = 1.0
			secondary_class_idx = -1
			if sec_model and crop is not None and crop.size > 0:
				sec_results = sec_model.predict(crop, verbose=False)
				if sec_results[0].probs is not None:
					secondary_class_idx = sec_results[0].probs.top1
					secondary_conf = sec_results[0].probs.top1conf.item()

			new_boxes.append((x1, y1, x2, y2, class_idx, secondary_class_idx, conf, secondary_conf))
		else:
			new_boxes.append((x1, y1, x2, y2, class_idx, conf))

	boxes = new_boxes



# draw boxes onto a frame copy
def _draw_oriented_boxes_on_image(out):
	for box in boxes:
		corners = bg.corners_from_head_tail_width(box['head'], box['tail'], box['width'])
		primary_cls = box['cls']
		pcol = primary_colors[primary_cls] if (primary_cls is not None and primary_cls < len(primary_colors)) else (255,255,255)
		conf = box.get('conf', -1)

		if hierarchical_mode:
			secondary_cls = box.get('sec_cls', -1)
			sec_conf = box.get('sec_conf', -1)
			scol = secondary_colors[secondary_cls] if (secondary_cls is not None and secondary_cls != -1 and secondary_cls < len(secondary_colors)) else pcol

			outer_th = max(1, line_thickness + 2)
			outer_corners = bg.inset_corners(corners, -outer_th)
			cv2.polylines(out, [np.array(outer_corners, dtype=np.int32)], True, pcol, outer_th, cv2.LINE_AA)

			label = [(f"{primary_classes[primary_cls].upper()}", pcol)]
			if conf != -1 and conf is not None:
				try:
					label[0] = (label[0][0] + f" {conf:.2f}", pcol)
				except Exception:
					pass

			if primary_classes[primary_cls] not in ignore_secondary:
				# Secondary box is nudged outward and drawn a shade thicker than a bare
				# line_thickness so it isn't visually swallowed by the thicker primary ring.
				inner_th = max(1, line_thickness + 1)
				inner_corners = bg.inset_corners(corners, -1)
				cv2.polylines(out, [np.array(inner_corners, dtype=np.int32)], True, scol, inner_th, cv2.LINE_AA)

				if secondary_cls is not None and secondary_cls != -1 and secondary_cls < len(secondary_classes):
					label2 = f" {secondary_classes[secondary_cls]}"
					if sec_conf != -1 and sec_conf is not None:
						try:
							label2 = label2 + f" {sec_conf:.2f}"
						except Exception:
							pass
					label.append((label2, scol))

			cv2.circle(out, (int(box['head'][0]), int(box['head'][1])), int(max(3, line_thickness * 2) * 1.5), pcol, -1, cv2.LINE_AA)
			anchor, angle_deg, align = bg.label_anchor_and_angle(corners, box['head'])
			bg.draw_rotated_text(out, label, anchor, angle_deg, pcol, align=align, font_scale=font_size, thickness=line_thickness)
		else:
			label = f"{primary_classes[primary_cls]}"
			if conf != -1 and conf is not None:
				try:
					label = label + f" {conf:.2f}"
				except Exception:
					pass
			bg.draw_oriented_box(out, corners, box['head'], pcol, thickness=line_thickness, label=label, label_color=pcol)


def _draw_temp_obb_preview(out, temp_obb):
	stage, head, tail, cursor, width_value, shift_amount = temp_obb
	if stage == 1 and head is not None and cursor is not None:
		cv2.line(out, (int(head[0]), int(head[1])), (int(cursor[0]), int(cursor[1])), (255,255,255), max(1, line_thickness))
		cv2.circle(out, (int(head[0]), int(head[1])), int(max(3, line_thickness * 2) * 1.5), (255,255,255), -1, cv2.LINE_AA)
		# full-frame crosshair line orthogonal to the current head->cursor (candidate tail) axis,
		# through the cursor - helps align the tail placement against the rest of the frame,
		# matching the axis-aligned crosshair shown for square boxes
		hx, hy = head
		dx, dy = hx - cursor[0], hy - cursor[1]
		length = (dx*dx + dy*dy) ** 0.5
		if length > 1e-6:
			perp = (-dy / length, dx / length)
			clipped = bg.clip_line_to_rect(cursor, perp, video_width, video_height)
			if clipped is not None:
				(x1, y1), (x2, y2) = clipped
				cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), max(1, line_thickness))
	elif stage == 2 and head is not None and tail is not None:
		draw_head, draw_tail = bg.shift_head_tail_sideways(head, tail, shift_amount)
		corners = bg.corners_from_head_tail_width(draw_head, draw_tail, max(1.0, width_value))
		bg.draw_oriented_box(out, corners, draw_head, (255,255,255), thickness=max(1, line_thickness))


def draw_boxes_on_image(base_img, temp_obb=None):
	"""
	Draw hierarchical boxes onto a *copy* of base_img.
	- Outer rectangle uses primary color (slightly thicker)
	- Inner rectangle uses secondary color (if present)
	- Label shows PRIMARY conf SECONDARY conf (primary uppercased)
	"""
	out = base_img.copy()

	if oriented:
		_draw_oriented_boxes_on_image(out)
		if temp_obb is not None:
			_draw_temp_obb_preview(out, temp_obb)
		for gx1, gy1, gx2, gy2 in grey_boxes:
			overlay = out.copy()
			cv2.rectangle(overlay, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (128,128,128), -1)
			cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
		return out

	for box in boxes:


		if hierarchical_mode:
			x1, y1, x2, y2, primary_cls, secondary_cls, conf, sec_conf = box
			# primary colour (BGR tuple) if available
			pcol = primary_colors[primary_cls] if (primary_cls is not None and primary_cls < len(primary_colors)) else (255,255,255)
			# if secondary present choose its colour, otherwise use primary for inner too
			scol = None
			if secondary_cls is not None and secondary_cls != -1 and secondary_cls < len(secondary_colors):
				scol = secondary_colors[secondary_cls]
			else:
				scol = pcol

			# draw outer box (primary) slightly thicker
			outer_th = max(1, line_thickness + 2)
			cv2.rectangle(out, (int(x1)-outer_th, int(y1)-outer_th), (int(x2)+outer_th, int(y2)+outer_th), pcol, outer_th)

			# draw inner box (secondary or primary) - nudged outward and a shade thicker than a
			# bare line_thickness so it isn't visually swallowed by the thicker primary ring.
			if primary_classes[primary_cls] not in ignore_secondary:
				inner_th = max(1, line_thickness + 1)
				cv2.rectangle(out, (int(x1)-1, int(y1)-1), (int(x2)+1, int(y2)+1), scol, inner_th)

			# compose label: PRIMARY (upper) [+ conf] in primary colour, then secondary [+ conf] in secondary colour
			label_segments = [(f"{primary_classes[primary_cls].upper()}", pcol)]
			if conf != -1 and conf is not None:
				try:
					label_segments[0] = (label_segments[0][0] + f" {conf:.2f}", pcol)
				except Exception:
					pass

			if primary_classes[primary_cls] not in ignore_secondary:
				if secondary_cls is not None and secondary_cls != -1 and secondary_cls < len(secondary_classes):
					label2 = f" {secondary_classes[secondary_cls]}"
					if sec_conf != -1 and sec_conf is not None:
						try:
							label2 = label2 + f" {sec_conf:.2f}"
						except Exception:
							pass
					label_segments.append((label2, scol))

			bg.draw_label_with_background(out, label_segments, int(x1), int(y1), font_size, line_thickness)

		else:
			x1, y1, x2, y2, primary_cls, conf = box
			pcol = primary_colors[primary_cls] if (primary_cls is not None and primary_cls < len(primary_colors)) else (255,255,255)
			cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), pcol, line_thickness)
			label = f"{primary_classes[primary_cls]}"
			if conf != -1 and conf is not None:
				try:
					label = label + f" {conf:.2f}"
				except Exception:
					pass
			cv2.putText(out, label, (int(x1), max(int(y1) - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, font_size, pcol, line_thickness, cv2.LINE_AA)

	# grey masks (as previously)
	for gx1, gy1, gx2, gy2 in grey_boxes:
		overlay = out.copy()
		cv2.rectangle(overlay, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (128,128,128), -1)
		cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

	return out
	
	
def _save_ornt_crops(box, primary_class_name, motion_ann_frame, static_ann_frame):
	"""Save whole-crop orientation-classifier training images for one box: the correctly-oriented
	upright crop (head-left, tail-right) as a 'correct' example, plus its 90/180/270-degree
	rotations as 'null' examples. Whole-crop (not head-vs-tail half-crop) so the classifier can
	use the entire animal's shape/appearance to judge orientation, not just a local head/tail
	feature - useful when there's no single obviously distinct 'head' marking to compare halves
	on. Independent of hierarchical_mode - this is a per-primary-class feature, not per-secondary-class."""
	head, tail, width = box['head'], box['tail'], box['width']
	for base_dir, src_frame in ((motion_ornt_base_dir, motion_ann_frame), (static_ornt_base_dir, static_ann_frame)):
		upright = bg.crop_rotated_upright(src_frame, head, tail, width)
		if upright is None or upright.size == 0:
			continue
		correct_dir = os.path.join(base_dir, primary_class_name, 'correct')
		os.makedirs(correct_dir, exist_ok=True)
		cv2.imwrite(os.path.join(correct_dir, f"{video_label}_{frame_number}.jpg"), upright)

		null_dir = os.path.join(base_dir, primary_class_name, 'null')
		os.makedirs(null_dir, exist_ok=True)
		for i, rotated in enumerate(bg.rotate_90_variants(upright)):
			cv2.imwrite(os.path.join(null_dir, f"{video_label}_{frame_number}_{i}.jpg"), rotated)


def _save_annotation_oriented():
	global annot_count
	randVal = random.random()
	is_val = randVal < val_frequency

	motion_target_img_dir = motion_val_images_dir if is_val else motion_train_images_dir
	motion_target_lbl_dir = motion_val_labels_dir if is_val else motion_train_labels_dir
	static_target_img_dir = static_val_images_dir if is_val else static_train_images_dir
	static_target_lbl_dir = static_val_labels_dir if is_val else static_train_labels_dir
	annot_type = 'validation' if is_val else 'training'

	motion_ann_frame = original_frame.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	static_ann_frame = fr.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	static_count = 0
	motion_count = 0

	for box in boxes:
		primary_cls = box['cls']
		corners = bg.corners_from_head_tail_width(box['head'], box['tail'], box['width'])
		poly = np.array(corners, dtype=np.int32)
		if primary_cls < len(primary_static_classes):
			static_count += 1
			if static_blocks_motion == 'true':
				cv2.fillPoly(motion_ann_frame, [poly], (128, 128, 128))
		else:
			motion_count += 1
			if motion_blocks_static == 'true':
				cv2.fillPoly(static_ann_frame, [poly], (128, 128, 128))

	h, w = original_frame.shape[:2]
	base_filename = f"{video_label}_{frame_number}"

	deleted = annotation_index.delete_frame(base_filename)
	if deleted:
		print("Overwriting existing annotation")

	if static_count > 0 or save_empty_frames == 'true':
		cv2.imwrite(os.path.join(static_target_img_dir, f"{base_filename}.jpg"), static_ann_frame)
		with open(os.path.join(static_target_lbl_dir, f"{base_filename}.txt"), 'w') as f:
			for box in boxes:
				if box['cls'] < len(primary_static_classes):
					corners = bg.corners_from_head_tail_width(box['head'], box['tail'], box['width'])
					corners_n = bg.norm_corners(corners, w, h)
					f.write(bg.write_label_line({'cls': box['cls'], 'corners_n': corners_n}, oriented=True))

	if motion_count > 0 or save_empty_frames == 'true':
		cv2.imwrite(os.path.join(motion_target_img_dir, f"{base_filename}.jpg"), motion_ann_frame)
		with open(os.path.join(motion_target_lbl_dir, f"{base_filename}.txt"), 'w') as f:
			for box in boxes:
				if box['cls'] >= len(primary_static_classes):
					corners = bg.corners_from_head_tail_width(box['head'], box['tail'], box['width'])
					corners_n = bg.norm_corners(corners, w, h)
					cls_in_file = box['cls'] - len(primary_static_classes)
					f.write(bg.write_label_line({'cls': cls_in_file, 'corners_n': corners_n}, oriented=True))

	if static_blocks_motion == 'true':
		motion_ann_frame = original_frame.copy()
		for gx1, gy1, gx2, gy2 in grey_boxes:
			cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	if motion_blocks_static == 'true':
		static_ann_frame = fr.copy()
		for gx1, gy1, gx2, gy2 in grey_boxes:
			cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	for box in boxes:
		primary_cls = box['cls']
		primary_class_name = primary_classes[primary_cls]
		corners = bg.corners_from_head_tail_width(box['head'], box['tail'], box['width'])
		x1, y1, _, _ = bg.axis_aligned_bounds(corners)

		if hierarchical_mode:
			secondary_cls = box.get('sec_cls', -1)
			if secondary_cls is not None and secondary_cls != -1:
				secondary_class_name = secondary_classes[secondary_cls]

				motion_crop = bg.crop_black_masked(motion_ann_frame, corners)
				if motion_crop is not None and motion_crop.size > 0:
					motion_class_dir = os.path.join(motion_cropped_base_dir, primary_class_name, secondary_class_name)
					os.makedirs(motion_class_dir, exist_ok=True)
					cv2.imwrite(os.path.join(motion_class_dir, f"{video_label}_{frame_number}_{x1}_{y1}.jpg"), motion_crop)

				static_crop = bg.crop_black_masked(static_ann_frame, corners)
				if static_crop is not None and static_crop.size > 0:
					static_class_dir = os.path.join(static_cropped_base_dir, primary_class_name, secondary_class_name)
					os.makedirs(static_class_dir, exist_ok=True)
					cv2.imwrite(os.path.join(static_class_dir, f"{video_label}_{frame_number}_{x1}_{y1}.jpg"), static_crop)

		if oriented:
			# always saved for oriented projects, regardless of disambiguate_head_tail, so the
			# training data is already there if the user later switches this setting on
			_save_ornt_crops(box, primary_class_name, motion_ann_frame, static_ann_frame)

	static_mask_dir = static_target_lbl_dir.replace('labels', 'masks')
	motion_mask_dir = motion_target_lbl_dir.replace('labels', 'masks')
	os.makedirs(static_mask_dir, exist_ok=True)
	os.makedirs(motion_mask_dir, exist_ok=True)

	mask_content = ""
	for gx1, gy1, gx2, gy2 in grey_boxes:
		mask_content += f"{gx1} {gy1} {gx2} {gy2}\n"

	mask_filename = f"{base_filename}.mask.txt"
	with open(os.path.join(static_mask_dir, mask_filename), 'w') as f:
		f.write(mask_content)
	with open(os.path.join(motion_mask_dir, mask_filename), 'w') as f:
		f.write(mask_content)

	print(f"Saved #{annot_count} frame {frame_number} -> {annot_type}")
	annot_count += 1


def save_annotation():
	global annot_count
	if original_frame is None or (not boxes and not grey_boxes) and save_empty_frames == 'false':
		return
	if oriented:
		_save_annotation_oriented()
		return
	# randomly assign to valdiation
	randVal = random.random()
	is_val = randVal < val_frequency
	
	motion_target_img_dir = motion_val_images_dir if is_val else motion_train_images_dir
	motion_target_lbl_dir = motion_val_labels_dir if is_val else motion_train_labels_dir

	static_target_img_dir = static_val_images_dir if is_val else static_train_images_dir
	static_target_lbl_dir = static_val_labels_dir if is_val else static_train_labels_dir
		
	annot_type = 'validation' if is_val else 'training'


	# Save image with grey overlays
	motion_ann_frame = original_frame.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)
		
	static_ann_frame = fr.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)
	

	# fill static boxes with grey (to avoid cross-training on similar motion & static things)
	static_count = 0
	motion_count = 0

	for box in boxes:
		if hierarchical_mode:
			x1, y1, x2, y2, primary_cls, _ , _ , _ = box
		else:
			x1, y1, x2, y2, primary_cls, _ = box
		if primary_cls < len(primary_static_classes): # primary class is static
			static_count +=1
			if static_blocks_motion == 'true':
				cv2.rectangle(motion_ann_frame, (x1, y1), (x2, y2), (128, 128, 128), -line_thickness)
		else:  # primary class is motion
			motion_count +=1
			if motion_blocks_static == 'true':
				cv2.rectangle(static_ann_frame, (x1, y1), (x2, y2), (128, 128, 128), -line_thickness)

	

	h, w = original_frame.shape[:2]
	base_filename = f"{video_label}_{frame_number}"
	
	## delete any existing annotations for this frame
	deleted = annotation_index.delete_frame(base_filename)
	if deleted:
		print("Overwriting existing annotation")
	
	if static_count > 0 or save_empty_frames == 'true': # don't save blank images	
		static_img_path = os.path.join(static_target_img_dir, f"{base_filename}.jpg")
		cv2.imwrite(static_img_path, static_ann_frame)	

		
		# Save static labels
		static_ann_path = os.path.join(static_target_lbl_dir, f"{base_filename}.txt")
		with open(static_ann_path, 'w') as f:
			for box in boxes:
				if hierarchical_mode:
					x1, y1, x2, y2, primary_cls, _ , _ , _ = box
				else:
					x1, y1, x2, y2, primary_cls, _  = box
				if primary_cls < len(primary_static_classes):
					# ~ if y1 < button_height:
						# ~ continue
					xc = (x1 + x2) / 2 / w
					yc = (y1 + y2) / 2 / h
					bw = abs(x2 - x1) / w
					bh = abs(y2 - y1) / h
					f.write(f"{primary_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

	if motion_count > 0 or save_empty_frames == 'true': # don't save blank images
		img_path = os.path.join(motion_target_img_dir, f"{base_filename}.jpg")
		cv2.imwrite(img_path, motion_ann_frame)
				
		# Save motion labels
		motion_ann_path = os.path.join(motion_target_lbl_dir, f"{base_filename}.txt")
		with open(motion_ann_path, 'w') as f:
			for box in boxes:
				if hierarchical_mode:
					x1, y1, x2, y2, primary_cls, _ , _ , _ = box
				else:
					x1, y1, x2, y2, primary_cls, _ = box
				if primary_cls >= len(primary_static_classes):
					# ~ if y1 < button_height:
						# ~ continue
					xc = (x1 + x2) / 2 / w
					yc = (y1 + y2) / 2 / h
					bw = abs(x2 - x1) / w
					bh = abs(y2 - y1) / h
					f.write(f"{primary_cls - len(primary_static_classes) } {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

	if static_blocks_motion == 'true':
		# ann_frames need re-making after greying out the static for the above primary training
		motion_ann_frame = original_frame.copy()
		for gx1, gy1, gx2, gy2 in grey_boxes:
			cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)
		
	if motion_blocks_static == 'true':		
		static_ann_frame = fr.copy()
		for gx1, gy1, gx2, gy2 in grey_boxes:
			cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)
							

	if hierarchical_mode:

		for box in boxes:
			x1, y1, x2, y2, primary_cls, secondary_cls, _ , _ = box
			####----Motion-----
			# ~ if secondary_cls > len(secondary_static_classes)-1:
			motion_crop = motion_ann_frame[y1:y2, x1:x2]
			if motion_crop.size == 0:
				continue
			
			# Create cropped image path
			primary_class_name = primary_classes[primary_cls]
			secondary_class_name = secondary_classes[secondary_cls]
			
			# Create target directory (static_class/motion_class)
			motion_class_dir = os.path.join(
				motion_cropped_base_dir, 
				primary_class_name, 
				secondary_class_name
			)

			os.makedirs(motion_class_dir, exist_ok=True)
			# Save image
			crop_path = os.path.join(
				motion_class_dir,
				f"{video_label}_{frame_number}_{x1}_{y1}.jpg"
			)
			cv2.imwrite(crop_path, motion_crop)
			
			####----Static-----
			# ~ if secondary_cls < len(secondary_static_classes)-1:
			static_crop = static_ann_frame[y1:y2, x1:x2]
			if static_crop.size == 0:
				continue
			
			# Create cropped image path
			primary_class_name = primary_classes[primary_cls]
			secondary_class_name = secondary_classes[secondary_cls]
			
			# Create target directory (static_class/motion_class)
			static_class_dir = os.path.join(
				static_cropped_base_dir, 
				primary_class_name, 
				secondary_class_name
			)
				
			os.makedirs(static_class_dir, exist_ok=True)
			# Save image
			crop_path = os.path.join(
				static_class_dir,
				f"{video_label}_{frame_number}_{x1}_{y1}.jpg"
			)
			cv2.imwrite(crop_path, static_crop)


	# Create mask directories
	static_mask_dir = static_target_lbl_dir.replace('labels', 'masks')
	motion_mask_dir = motion_target_lbl_dir.replace('labels', 'masks')
	os.makedirs(static_mask_dir, exist_ok=True)
	os.makedirs(motion_mask_dir, exist_ok=True)

	# Save grey box coordinates to mask files
	mask_content = ""
	for gx1, gy1, gx2, gy2 in grey_boxes:
		mask_content += f"{gx1} {gy1} {gx2} {gy2}\n"
	
	# Write mask files
	mask_filename = f"{base_filename}.mask.txt"
	static_mask_path = os.path.join(static_mask_dir, mask_filename)
	motion_mask_path = os.path.join(motion_mask_dir, mask_filename)
	
	with open(static_mask_path, 'w') as f:
		f.write(mask_content)
	with open(motion_mask_path, 'w') as f:
		f.write(mask_content)

	print(f"Saved #{annot_count} frame {frame_number} -> {annot_type}")

	annot_count += 1	
	

# ---------- Tk UI (composite single-image display) ----------
class AnnotatorTk:
	def __init__(self, root):
		self.root = root
		root.title(f"BehaveAI — {os.path.basename(video_path)}")
		
		# sensible default window geometry so the main video panel is visible on launch
		default_w = max(1000, int(video_width * 1.2))
		default_h = max(700, int(video_height * 1.2))
		root.geometry(f"{default_w}x{default_h}")
		root.minsize(900, 600)

		# main layout
		self.main = tk.Frame(root)
		self.main.pack(fill='both', expand=True)

		# left container which holds the single composite canvas
		self.left = tk.Frame(self.main)
		self.left.pack(side='left', fill='both', expand=True)
		self.left.pack_propagate(False)

		# conservative initial canvas size to avoid early thrash
		self.canvas = tk.Canvas(self.left, bg='black', highlightthickness=0,
								width=min(800, video_width), height=min(600, video_height))
		self.canvas.pack(fill='both', expand=True)

		# --- bottom control bar (seek + grey toggle) ---
		self.controls = tk.Frame(self.left)
		self.controls.pack(fill='x', pady=(4, 2))
		
		# grey toggle (left)
		self.grey_btn = tk.Button(self.controls, text="Grey (g)", width=10, command=self.toggle_grey)
		self.grey_btn.pack(side='left', padx=4)
		
		# frame number label (shows current frame number)
		self.frame_var = tk.StringVar(value=str(frame_number))
		self.frame_label = tk.Label(self.controls, textvariable=self.frame_var, width=8, anchor='w')
		self.frame_label.pack(side='left', padx=(0,6))

		# container for tickline + seek scale so ticks sit *above* the slider
		self.seek_container = tk.Frame(self.controls)
		self.seek_container.pack(side='left', fill='x', expand=True, padx=4)
		
		# small tick canvas sitting above the actual scale (height can be tuned)
		self.seek_ticks = tk.Canvas(self.seek_container, height=8, bg=self.controls.cget('bg'), highlightthickness=0)
		self.seek_ticks.pack(fill='x', padx=0, pady=(0,1))
		
		self.seek_ticks.bind('<Configure>', lambda e: self.draw_seek_ticks())

		
		# the real seek scale below the tick rail
		self.seek = ttk.Scale(
			self.seek_container,
			from_=0,
			to=max(0, total_frames - 1),
			orient='horizontal',
			command=self.on_seek
		)
		self.seek.pack(fill='x', expand=True)

		self.buttons_frame = tk.Frame(self.left)
		self.buttons_frame.pack(side='bottom', fill='x', pady=(4,4))

		self.primary_buttons = []
		self.secondary_buttons = []

		# create primary buttons
		col = 0
		for idx, name in enumerate(primary_classes):
			if name == '0':
				continue
			color_hex = None
			if idx < len(primary_colors):
				bgr = primary_colors[idx]
				color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
			btn = tk.Button(self.buttons_frame, text="{} ({})".format(name, primary_classes_info[idx][0]),
							width=12, relief='raised', command=lambda i=idx: self.select_primary(i))
			btn.grid(row=0, column=col, padx=2, pady=2)
			self.primary_buttons.append((btn, color_hex, idx))
			col += 1

		# secondary row
		if hierarchical_mode:
			col = 0
			for idx, name in enumerate(secondary_classes):
				color_hex = None
				if idx < len(secondary_colors):
					bgr = secondary_colors[idx]
					color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
				btn = tk.Button(self.buttons_frame, text="{} ({})".format(name, secondary_classes_info[idx][0]),
								width=12, relief='raised', command=lambda i=idx: self.select_secondary(i))
				btn.grid(row=1, column=col, padx=2, pady=2)
				self.secondary_buttons.append((btn, color_hex, idx))
				col += 1


		# bind events
		self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
		self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
		self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
		self.canvas.bind('<Button-3>', self.on_right_click)
		# Alt+left-click deletes too - an alternative to right-click, which is unreliable on some
		# macOS mice/trackpads. The more specific <Alt-ButtonPress-1> binding takes precedence over
		# plain <ButtonPress-1>, so the click doesn't also start drawing a box.
		self.canvas.bind('<Alt-ButtonPress-1>', self.on_alt_click)
		if root.tk.call('tk', 'windowingsystem') == 'aqua':
			# macOS: the Option key reports as Tk's "Option" modifier rather than "Alt", and Tk < 8.7
			# reports the right mouse button as Button-2 (it became Button-3 from Tk 8.7). macOS-only
			# because on X11 "Option" (Mod2) is usually NumLock and Button-2 is the middle button.
			self.canvas.bind('<Option-ButtonPress-1>', self.on_alt_click)
			if tk.TkVersion < 8.7:
				self.canvas.bind('<Button-2>', self.on_right_click)
		self.canvas.bind('<Motion>', self.on_motion)

		root.bind_all('<Key>', self.on_key_all)
		# ~ root.bind_all('<Left>', lambda e: self.key_step(-1))
		# ~ root.bind_all('<Right>', lambda e: self.key_step(1))
		root.bind_all('<space>', lambda e: self.toggle_show_mode())
		root.bind_all('<Return>', lambda e: self.key_save())

		# drawing/display state
		# ~ self.display_size = (video_width, video_height)
		self.display_size = (min(800, video_width), min(600, video_height))
		self.tk_img = None
		self.last_mouse = None
		self.drawing = False
		self.start_canvas_xy = None

		# oriented-box 3-click draw state (head -> tail -> width)
		self.obb_stage = 0
		self.obb_head = None
		self.obb_tail = None
		self.obb_shift_held = False
		# perpendicular cursor distance from the centre line drives width normally, or - while
		# Shift is held - drives the sideways shift instead; each is frozen while the other is
		# active. Driven by the CHANGE in cursor distance since a mode was last (re)entered
		# (tracked via the *_dist_offset anchors below), not the absolute distance, so toggling
		# Shift mid-drag doesn't snap the box to wherever the cursor happens to be.
		self.obb_width_value = 0.0
		self.obb_shift_amount = 0.0
		self.obb_width_dist_offset = 0.0
		self.obb_shift_dist_offset = 0.0
		self.obb_prev_shift_held = False

		# small layout tuning: padding between main and zoom column when composing
		self._composite_gap = 8

		# schedule loop
		self.root.after(30, self.loop)
		self.update_button_states()


	# button handlers
	def select_primary(self, class_idx):
		global active_primary, grey_mode, show_mode
		active_primary = class_idx
		grey_mode = False
		if active_primary < len(primary_static_classes):
			show_mode = -1
		else:
			show_mode = 1
		self.update_button_states()
		self.redraw()

	def select_secondary(self, class_idx):
		global active_secondary, grey_mode, show_mode
		active_secondary = class_idx
		grey_mode = False
		if class_idx < len(secondary_static_classes):
			show_mode = -1
		else:
			show_mode = 1
		self.update_button_states()
		self.redraw()

	def toggle_grey(self):
		global grey_mode
		grey_mode = not grey_mode
		self.update_button_states()

	def update_button_states(self):
		for btn, col, cls in self.primary_buttons:
			if cls == active_primary:
				btn.config(relief='sunken')
				if col:
					try:
						btn.config(bg=col)
					except Exception:
						pass
			else:
				btn.config(relief='raised', bg='#888888')
		for btn, col, cls in self.secondary_buttons:
			if cls == active_secondary:
				btn.config(relief='sunken')
				if col:
					try:
						btn.config(bg=col)
					except Exception:
						pass
			else:
				btn.config(relief='raised', bg='#888888')
		self.grey_btn.config(relief='sunken' if grey_mode else 'raised')

	def draw_seek_ticks(self):
		"""Draw small ticks for annotated frames and a red cursor for current frame."""
		try:
			self.seek_ticks.delete('all')
		except Exception:
			return
	
		w = self.seek_ticks.winfo_width()
		if w <= 2:
			# widget not yet realised — try again shortly
			self.root.after(100, self.draw_seek_ticks)
			return
	
		# get annotated frames for this video's video_label
		ann_set = annotated_frames_map.get(video_label, set())
		if not ann_set:
			return
	
		# draw ticks (color / height are adjustable)
		for frm in ann_set:
			if frm < 0 or frm >= max(1, total_frames):
				continue
			x = int(round((frm / float(max(1, total_frames - 1))) * (w - 1)))
			# short yellow tick (top-down)
			# ~ self.seek_ticks.create_line(x, 0, x, 6, fill='yellow', width=1)
			self.seek_ticks.create_line(x, 0, x, 10, fill='red', width=2)
	
		# draw current-frame cursor
		cur_x = int(round((frame_number / float(max(1, total_frames - 1))) * (w - 1)))
		# ~ self.seek_ticks.create_line(cur_x, 0, cur_x, 7, fill='red', width=2)
		self.seek_ticks.create_line(cur_x, 0, cur_x, 10, fill='black', width=2)
		
	def refresh_annotation_index_map(self):
		"""Rebuild global `items` and annotated_frames_map from the shared index."""
		try:
			global items, annotated_frames_map
			items = annotation_index.list_images_labels_and_masks()
			annotated_frames_map = build_annot_index_map(items)
		except Exception:
			pass
	
	def jump_to_annotated(self, direction):
		"""Jump to previous (direction=-1) or next (direction=+1) annotated frame for current video_label.
		   If none found, do nothing.
		"""
		try:
			ann_set = sorted(annotated_frames_map.get(video_label, []))
			if not ann_set:
				return
			cur = int(frame_number)
			if direction > 0:
				# next annotated frame strictly greater than cur
				for frm in ann_set:
					if frm > cur:
						self.seek.set(frm)
						self.on_seek(str(frm))
						return
				# wrap to first
				self.seek.set(ann_set[0])
				self.on_seek(str(ann_set[0]))
			else:
				# previous annotated frame strictly less than cur
				for frm in reversed(ann_set):
					if frm < cur:
						self.seek.set(frm)
						self.on_seek(str(frm))
						return
				# wrap to last
				self.seek.set(ann_set[-1])
				self.on_seek(str(ann_set[-1]))
		except Exception:
			pass


	def on_seek(self, val):
		global frame_number, frame_updated
		try:
			frame_number = int(float(val))
		except Exception:
			frame_number = 0
		frame_updated = True
		try:
			self.frame_var.set(f'Frame {str(frame_number)}')
		except Exception:
			pass
		# redraw ticks to show current cursor
		try:
			self.draw_seek_ticks()
		except Exception:
			pass
	


	def canvas_to_video(self, canvas_point):
		"""
		Map a canvas (x,y) into video coordinates (vx, vy).
		Accounts for the composite image being uniformly scaled to fit the canvas.
		Top-left anchored (composite drawn at 0,0).
		"""
		cx, cy = canvas_point
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1
	
		# fallback values if redraw hasn't set them yet
		disp_w, disp_h = getattr(self, 'display_size', (video_width, video_height))
		scale = getattr(self, 'composite_scale', 1.0)
	
		# scaled displayed video region (left part of composite)
		scaled_disp_w = max(1, int(round(disp_w * scale)))
		scaled_disp_h = max(1, int(round(disp_h * scale)))
	
		# if click is outside scaled main display, clamp to nearest edge
		if cx < 0: cx = 0
		if cy < 0: cy = 0
	
		# only map if inside scaled main display; if outside we still return nearest edge point
		# map back to display coords then to video coords
		display_x = min(cx, scaled_disp_w - 1) / scale
		display_y = min(cy, scaled_disp_h - 1) / scale
	
		vx = display_x * (video_width / float(max(1, disp_w)))
		vy = display_y * (video_height / float(max(1, disp_h)))
		return (vx, vy)
	


	def video_to_canvas(self, vx, vy):
		disp_w, disp_h = self.display_size
		cx = int(round((vx * disp_w / float(video_width))))
		cy = int(round((vy * disp_h / float(video_height))))
		return (cx, cy)

	# drawing handlers
	def on_mouse_down(self, event):
		if oriented and not grey_mode:
			self.on_obb_click(event)
			return
		self.drawing = True
		self.start_canvas_xy = (event.x, event.y)
		self.last_mouse = (event.x, event.y)

	def on_obb_click(self, event):
		"""3-click oriented-box draw flow: click1=head, click2=tail, click3=width (commit)."""
		v = self.canvas_to_video((event.x, event.y))
		v = (max(0, min(video_width - 1, v[0])), max(0, min(video_height - 1, v[1])))
		if self.obb_stage == 0:
			self.obb_head = v
			self.obb_stage = 1
		elif self.obb_stage == 1:
			self.obb_tail = v
			self.obb_stage = 2
			self.obb_width_value = 0.0
			self.obb_shift_amount = 0.0
			self.obb_width_dist_offset = 0.0
			self.obb_shift_dist_offset = 0.0
			self.obb_prev_shift_held = False
		elif self.obb_stage == 2:
			self.obb_shift_held = bool(event.state & 0x1)
			self._update_obb_width_or_shift(v)
			width = self.obb_width_value
			if width >= 2:
				final_head, final_tail = bg.shift_head_tail_sideways(self.obb_head, self.obb_tail, self.obb_shift_amount)
				if hierarchical_mode:
					boxes.append({'head': final_head, 'tail': final_tail, 'width': width,
								  'cls': active_primary, 'sec_cls': active_secondary, 'conf': -1, 'sec_conf': -1})
				else:
					boxes.append({'head': final_head, 'tail': final_tail, 'width': width,
								  'cls': active_primary, 'conf': -1})
			self.obb_stage = 0
			self.obb_head = None
			self.obb_tail = None
		self.redraw()

	def _update_obb_width_or_shift(self, cursor):
		"""Stage 2: the perpendicular distance from `cursor` to the (original, un-shifted) head-tail
		line drives WIDTH normally, or - while Shift is held - drives the sideways SHIFT instead.
		Whichever one isn't currently active is left frozen at its last value. Each is advanced by
		the CHANGE in cursor distance since its mode was last (re)entered (not the raw absolute
		distance) so that toggling Shift mid-drag doesn't snap the box to wherever the cursor
		currently is - the *_dist_offset anchors are reset to the cursor's current distance the
		instant a mode switch is detected, making that switch a zero-delta (no-jump) event."""
		if self.obb_stage != 2 or self.obb_head is None or self.obb_tail is None:
			return
		signed = bg.signed_perpendicular_distance(self.obb_head, self.obb_tail, cursor)
		if self.obb_shift_held != self.obb_prev_shift_held:
			if self.obb_shift_held:
				self.obb_shift_dist_offset = signed
			else:
				self.obb_width_dist_offset = abs(signed)
			self.obb_prev_shift_held = self.obb_shift_held
		if self.obb_shift_held:
			self.obb_shift_amount += signed - self.obb_shift_dist_offset
			self.obb_shift_dist_offset = signed
		else:
			dist = abs(signed)
			self.obb_width_value = max(0.0, self.obb_width_value + 2.0 * (dist - self.obb_width_dist_offset))
			self.obb_width_dist_offset = dist

	def cancel_obb_draw(self):
		if self.obb_stage != 0:
			self.obb_stage = 0
			self.obb_head = None
			self.obb_tail = None
			self.obb_width_value = 0.0
			self.obb_shift_amount = 0.0
			self.obb_width_dist_offset = 0.0
			self.obb_shift_dist_offset = 0.0
			self.obb_prev_shift_held = False
			return True
		return False

	def on_mouse_drag(self, event):
		if not self.drawing:
			return
		self.last_mouse = (event.x, event.y)
		self.redraw(temp_rect=(self.start_canvas_xy, (event.x, event.y)))

	def on_mouse_up(self, event):
		if not self.drawing:
			return
		self.drawing = False
		start_v = self.canvas_to_video(self.start_canvas_xy)
		end_v = self.canvas_to_video((event.x, event.y))
		x1, x2 = sorted([int(round(start_v[0])), int(round(end_v[0]))])
		y1, y2 = sorted([int(round(start_v[1])), int(round(end_v[1]))])
		x1 = max(0, min(video_width-1, x1)); x2 = max(0, min(video_width-1, x2))
		y1 = max(0, min(video_height-1, y1)); y2 = max(0, min(video_height-1, y2))
		if abs(x2-x1) > 5 and abs(y2-y1) > 5:
			if grey_mode:
				grey_boxes.append((x1, y1, x2, y2))
			else:
				if hierarchical_mode:
					boxes.append((x1, y1, x2, y2, active_primary, active_secondary, -1, -1))
				else:
					boxes.append((x1, y1, x2, y2, active_primary, -1))
		self.redraw()

	def on_alt_click(self, event):
		"""Alt/Option+left-click: same as right-click (cancel in-progress oriented box, or delete the box under the cursor)."""
		self.on_right_click(event)
		return "break"

	def on_right_click(self, event):
		if oriented and self.cancel_obb_draw():
			self.redraw()
			return
		v = self.canvas_to_video((event.x, event.y))
		x, y = int(v[0]), int(v[1])
		removed = False
		for i in range(len(boxes)-1, -1, -1):
			if oriented:
				corners = bg.corners_from_head_tail_width(boxes[i]['head'], boxes[i]['tail'], boxes[i]['width'])
				hit = bg.point_in_corners(x, y, corners)
			else:
				bx1, by1, bx2, by2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
				hit = bx1 <= x <= bx2 and by1 <= y <= by2
			if hit:
				del boxes[i]; removed = True; break
		if not removed:
			for i in range(len(grey_boxes)-1, -1, -1):
				gx1, gy1, gx2, gy2 = grey_boxes[i]
				if gx1 <= x <= gx2 and gy1 <= y <= gy2:
					del grey_boxes[i]; break
		try:
			self.redraw()
		except Exception:
			print("Error redrawing after right-click delete:")
			traceback.print_exc()

	def on_motion(self, event):
		self.last_mouse = (event.x, event.y)
		self.obb_shift_held = bool(event.state & 0x1)
		self.redraw()

	# keyboard
	def on_key_all(self, event):
		global active_primary, active_secondary, grey_mode, boxes, grey_boxes, frame_number, frame_updated, show_mode

		ch = event.char
		ks = event.keysym

		if ks == 'Escape':
			if oriented and self.cancel_obb_draw():
				self.redraw()
			return

		# Frame step - step larger when Shift is held (event.state & 0x1 tests Shift mask)
		# Support CTRL + Left/Right to jump to previous/next annotated frame (event.state & 0x4 tests CTRL mask on X11)
		if ks == 'Left':
			# CTRL jump to previous annotated frame
			if (event.state & 0x4):
				self.jump_to_annotated(-1)
				return
			step = -10 if (event.state & 0x1) else -1
			self.key_step(step)
			return
		if ks == 'Right':
			# CTRL jump to next annotated frame
			if (event.state & 0x4):
				self.jump_to_annotated(+1)
				return
			step = 10 if (event.state & 0x1) else 1
			self.key_step(step)
			return
		

		if ch:
			c_ord = ord(ch)
			if c_ord in primary_class_dict and c_ord in secondary_class_dict:
				if ch != '0':
					active_primary = primary_class_dict[c_ord]
					active_secondary = secondary_class_dict[c_ord]
					grey_mode = False
					if active_primary < len(primary_static_classes):
						show_mode = -1
					else:
						show_mode = 1
					self.update_button_states()
					return
			if c_ord in primary_class_dict:
				if ch != '0':
					active_primary = primary_class_dict[c_ord]
					grey_mode = False
					if active_primary < len(primary_static_classes):
						show_mode = -1
					else:
						show_mode = 1
					self.update_button_states()
					return
			if c_ord in secondary_class_dict:
				if ch != '0':
					active_secondary = secondary_class_dict[c_ord]
					grey_mode = False
					if active_secondary < len(secondary_static_classes):
						show_mode = -1
					else:
						show_mode = 1
					self.update_button_states()
					return

		if ch == 'u':
			if grey_mode:
				if grey_boxes: grey_boxes.pop()
			elif boxes:
				boxes.pop()
			self.redraw()
			return

		if ch == 'g':
			self.toggle_grey()
			return

		if ks == 'Return':
			save_annotation()
			boxes.clear(); grey_boxes.clear()
			frame_number = min(frame_number + 1, total_frames - 1)
			frame_updated = True
			self.seek.set(frame_number)
			
			try:
				# refresh index and redraw ticks immediately
				self.refresh_annotation_index_map()
				self.draw_seek_ticks()
			except Exception:
				pass
			self.redraw()
			return
			
		if ks == 'Delete':
			print("\nWARNING: This will delete ALL files for this frame!")
			print("Press ENTER to confirm, any other key to cancel...")
			# Wait for confirmation using a simple key binding approach
			self.root.bind('<Return>', self.confirm_delete)
			self.root.bind('<Escape>', self.cancel_delete)
			self.delete_pending = True
			return

	def confirm_delete(self, event=None):
		if hasattr(self, 'delete_pending') and self.delete_pending:
			base_filename = f"{video_label}_{frame_number}"
			# ~ if delete_frame_data(base_filename):
			deleted = annotation_index.delete_frame(base_filename)
			if deleted:
				# Clear the current display
				boxes.clear()
				grey_boxes.clear()
				print(f"All files for frame {frame_number} have been deleted")
				frame_updated = True
				try:
					# refresh index and redraw ticks immediately
					self.refresh_annotation_index_map()
					self.draw_seek_ticks()
				except Exception:
					pass
				self.redraw()
			self.delete_pending = False
			# Remove the temporary key bindings
			self.root.unbind('<Return>')
			self.root.unbind('<Escape>')
			# Prevent the save function from being called
			return "break"

	def cancel_delete(self, event=None):
		if hasattr(self, 'delete_pending') and self.delete_pending:
			print("Deletion cancelled")
			self.delete_pending = False
			# Remove the temporary key bindings
			self.root.unbind('<Return>')
			self.root.unbind('<Escape>')
			# Prevent the save function from being called
			return "break"			
	
	def key_step(self, delta):
		global frame_number, frame_updated
		frame_number = min(max(0, frame_number + delta), total_frames - 1)
		frame_updated = True
		self.seek.set(frame_number)

	def toggle_show_mode(self):
		global show_mode
		show_mode *= -1

	def key_save(self):
		save_annotation()
		boxes.clear(); grey_boxes.clear()
		global frame_number, frame_updated
		frame_number = min(frame_number + 1, total_frames - 1)
		frame_updated = True
		self.seek.set(frame_number)
		try:
			self.refresh_annotation_index_map()
			self.draw_seek_ticks()
		except Exception:
			pass
	

	def redraw(self, temp_rect=None):
		"""
		Compose main display + three zoom panes into a single composite image,
		scale that composite uniformly to fit the available canvas width/height,
		then display it anchored top-left. Draw crosshair and temp rect on the
		scaled composite so canvas coords match.
		"""
		global original_frame, fr, motion_image, last_mouse_move
	
		if original_frame is None:
			return
	
		# pick base image depending on current view mode (native video pixels)
		if show_mode == -1:
			base = fr.copy() if fr is not None else np.zeros((video_height, video_width, 3), dtype=np.uint8)
		else:
			base = motion_image.copy() if motion_image is not None else np.zeros((video_height, video_width, 3), dtype=np.uint8)
	
		# draw boxes/grey boxes onto base (works in video/native coords)
		temp_obb = None
		if oriented and self.obb_stage in (1, 2) and self.last_mouse is not None:
			try:
				cursor_v = self.canvas_to_video(self.last_mouse)
			except Exception:
				cursor_v = None
			if cursor_v is not None:
				if self.obb_stage == 2:
					self._update_obb_width_or_shift(cursor_v)
				temp_obb = (self.obb_stage, self.obb_head, self.obb_tail, cursor_v, self.obb_width_value, self.obb_shift_amount)
		display = draw_boxes_on_image(base, temp_obb=temp_obb)
	
		# initial desired main display size (before final uniform scaling)
		disp_w = max(1, int(self.display_size[0]))
		disp_h = max(1, int(self.display_size[1]))
	
		# resize main display (native composite size)
		disp_resized = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
	
		# --- prepare zoom panes (native size) ---
		MAG = 2.0
		MAG_ANIM = 1.0
	
		widget_size = max(32, int(disp_h / 3))
	
		display_scale = float(video_width) / float(max(1, disp_w))
		crop_vid = max(2, int(round(widget_size * display_scale / MAG)))
		crop_vid_anim = max(2, int(round(widget_size * display_scale / MAG_ANIM)))
	
		# prevent absurdly large crop sizes (memory blowouts).
		MAX_ZOOM_CROP = 2048
		crop_vid = min(crop_vid, MAX_ZOOM_CROP)
		crop_vid_anim = min(crop_vid_anim, MAX_ZOOM_CROP)
	
		# ~ def padded_crop(src, cx, cy, crop_size):
			# ~ h, w = src.shape[:2]
			# ~ x1 = cx - crop_size // 2
			# ~ y1 = cy - crop_size // 2
			# ~ x2 = x1 + crop_size
			# ~ y2 = y1 + crop_size
			# ~ sx1 = max(0, x1); sy1 = max(0, y1)
			# ~ sx2 = min(w, x2); sy2 = min(h, y2)
			# ~ out = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
			# ~ if sx2 > sx1 and sy2 > sy1:
				# ~ dst_x1 = sx1 - x1
				# ~ dst_y1 = sy1 - y1
				# ~ dst_x2 = dst_x1 + (sx2 - sx1)
				# ~ dst_y2 = dst_y1 + (sy2 - sy1)
				# ~ out[dst_y1:dst_y2, dst_x1:dst_x2] = src[sy1:sy2, sx1:sx2]
			# ~ return out, (x1, y1, x2, y2)

		def padded_crop(src, cx, cy, crop_size):
			h, w = src.shape[:2]

			# Defensive clamp in case upstream computed a large crop_size.
			MAX_PADDDED_CROP = 2048
			use_crop = int(min(crop_size, MAX_PADDDED_CROP))

			x1 = cx - crop_size // 2
			y1 = cy - crop_size // 2
			x2 = x1 + crop_size
			y2 = y1 + crop_size
			sx1 = max(0, x1); sy1 = max(0, y1)
			sx2 = min(w, x2); sy2 = min(h, y2)

			# create output at the clamped size but still compute box using original crop coords
			out = np.zeros((use_crop, use_crop, 3), dtype=np.uint8)
			if sx2 > sx1 and sy2 > sy1:
				# destination offsets must respect the difference between original and clamped size
				# compute offsets relative to the clamped output
				dst_x1 = sx1 - x1
				dst_y1 = sy1 - y1
				dst_x2 = dst_x1 + (sx2 - sx1)
				dst_y2 = dst_y1 + (sy2 - sy1)

				# If we clamped use_crop < crop_size, we may need to shift the destination region
				# ensure indices fit inside out array
				dst_x1 = max(0, dst_x1)
				dst_y1 = max(0, dst_y1)
				dst_x2 = min(use_crop, dst_x2)
				dst_y2 = min(use_crop, dst_y2)

				out[dst_y1:dst_y2, dst_x1:dst_x2] = src[sy1:sy2, sx1:sx2]
			return out, (x1, y1, x2, y2)
	
		# center of interest in video coords
		if self.last_mouse is not None:
			try:
				vx, vy = self.canvas_to_video(self.last_mouse)
				cx = int(min(max(0, vx), video_width - 1))
				cy = int(min(max(0, vy), video_height - 1))
			except Exception:
				cx, cy = video_width // 2, video_height // 2
		else:
			cx, cy = video_width // 2, video_height // 2
	
		# ~ # top zoom (static)
		z_top = None
		if fr is not None:
			crop_img, crop_box = padded_crop(fr, cx, cy, crop_vid)
			z_top = cv2.resize(crop_img, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
			rel_x = cx - crop_box[0]; rel_y = cy - crop_box[1]
			if 0 <= rel_x < crop_vid and 0 <= rel_y < crop_vid:
				zx = int(round(rel_x * widget_size / crop_vid))
				zy = int(round(rel_y * widget_size / crop_vid))
				cv2.line(z_top, (0, zy), (widget_size-1, zy), (255,255,255), 1)
				cv2.line(z_top, (zx, 0), (zx, widget_size-1), (255,255,255), 1)
			cv2.rectangle(z_top, (0, 0), (widget_size-1, widget_size-1), (0, 0, 0), 1)
		
		# mid zoom (motion)
		z_mid = None
		if original_frame is not None:
			crop_img, crop_box = padded_crop(original_frame, cx, cy, crop_vid)
			z_mid = cv2.resize(crop_img, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
			rel_x = cx - crop_box[0]; rel_y = cy - crop_box[1]
			if 0 <= rel_x < crop_vid and 0 <= rel_y < crop_vid:
				zx = int(round(rel_x * widget_size / crop_vid))
				zy = int(round(rel_y * widget_size / crop_vid))
				cv2.line(z_mid, (0, zy), (widget_size-1, zy), (255,255,255), 1)
				cv2.line(z_mid, (zx, 0), (zx, widget_size-1), (255,255,255), 1)
			cv2.rectangle(z_mid, (0, 0), (widget_size-1, widget_size-1), (0, 0, 0), 1)

		# bottom zoom (animation)
		z_bot = None
		if len(raw_buf) == raw_buf.maxlen:
			idx = int(((time.time() - last_mouse_move) * ANIM_FPS) % raw_buf.maxlen)
			small = raw_buf[idx]
			small_crop, crop_box = padded_crop(small, cx, cy, crop_vid_anim)
			z_bot = cv2.resize(small_crop, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
		else:
			z_bot = np.zeros((widget_size, widget_size, 3), dtype=np.uint8)
		# add single-pixel black border to bottom pane as well
		cv2.rectangle(z_bot, (0, 0), (widget_size-1, widget_size-1), (0, 0, 0), 1)

		gap = 0
		right_col_w = widget_size
		right_col_h = widget_size * 3  # no extra gap used here
		
		# composite size: main display + immediate right column
		composite_h = max(disp_h, right_col_h)
		composite_w = disp_w + right_col_w
		composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
		
		# place main display at top-left (no horizontal gap)
		composite[0:disp_h, 0:disp_w] = disp_resized
		
		# zoom column starts immediately after the main display
		zoom_x_off = disp_w
		zoom_y_off = 0
					

	
		def place_zoom(zi, x_off, y_off):
			if zi is None:
				return
			h_rem = composite.shape[0] - y_off
			w_rem = composite.shape[1] - x_off
			if h_rem <= 0 or w_rem <= 0:
				return
			zi_h, zi_w = zi.shape[:2]
			use_h = min(zi_h, h_rem)
			use_w = min(zi_w, w_rem)
			zi_crop = zi[0:use_h, 0:use_w]
			composite[y_off:y_off+use_h, x_off:x_off+use_w] = zi_crop
	
		place_zoom(z_top, zoom_x_off, zoom_y_off)
		place_zoom(z_mid, zoom_x_off, zoom_y_off + widget_size + gap)
		place_zoom(z_bot, zoom_x_off, zoom_y_off + 2 * (widget_size + gap))
	
		# --- scale composite to fit canvas ---
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1
		scale_w = float(c_w) / float(max(1, composite_w))
		scale_h = float(c_h) / float(max(1, composite_h))
		scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else 1.0
		# store for mapping functions
		self.composite_scale = scale
	
		scaled_w = max(1, int(round(composite_w * scale)))
		scaled_h = max(1, int(round(composite_h * scale)))
		scaled = cv2.resize(composite, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)


		# draw crosshair — but *limit* it to the main display area so it doesn't cross into the zoom column
		# (skipped in oriented mode: the angled head/tail-axis crosshair replaces it there - see
		# _draw_temp_obb_preview - the zoom-pane crosshairs below are unaffected)
		scaled_disp_w = max(1, int(round(disp_w * scale)))
		scaled_disp_h = max(1, int(round(disp_h * scale)))

		if not oriented and self.last_mouse is not None:
			mx, my = self.last_mouse
			# only draw if the mouse is inside the scaled main-display region
			if 0 <= mx < scaled_disp_w and 0 <= my < scaled_disp_h:
				cv2.line(scaled, (int(mx), 0), (int(mx), scaled_disp_h), (255,255,255), max(1, line_thickness))
				cv2.line(scaled, (0, int(my)), (scaled_disp_w, int(my)), (255,255,255), max(1, line_thickness))
		
	
		# determine temporary rectangle to draw (if drawing and no explicit temp_rect provided)
		if temp_rect is None and getattr(self, 'drawing', False):
			if self.start_canvas_xy is not None and self.last_mouse is not None:
				temp_rect = (self.start_canvas_xy, self.last_mouse)
	
		# draw temporary rect (coordinates are canvas coords; draw onto scaled image)
		if temp_rect is not None:
			(sx, sy), (ex, ey) = temp_rect
			# clip to scaled image
			rx1 = max(0, min(sx, scaled_w-1)); ry1 = max(0, min(sy, scaled_h-1))
			rx2 = max(0, min(ex, scaled_w-1)); ry2 = max(0, min(ey, scaled_h-1))
			cv2.rectangle(scaled, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255,255,255), max(1, line_thickness))
	
		# convert and display
		self.tk_img = cv2_to_photoimage(scaled)
		try:
			self.canvas.config(scrollregion=(0, 0, scaled_w, scaled_h))
		except Exception:
			pass
		self.canvas.delete('all')
		self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')

		try:
			self.draw_seek_ticks()
		except Exception:
			pass
		
	

	def loop(self):
		# The self-rescheduling after(30, self.loop) call must always run, even if
		# _loop_body() raises - otherwise the canvas silently stops refreshing forever
		# (Tk's default callback-exception handler just prints and swallows the error,
		# so an unhandled exception here previously killed the redraw loop for good).
		try:
			self._loop_body()
		except Exception:
			print("Error in redraw loop:")
			traceback.print_exc()
		self.root.after(30, self.loop)

	def _loop_body(self):
		global frame_updated, fr, original_frame, motion_image, raw_buf, last_mouse_move, last_anim_draw, boxes, grey_boxes

		need_ = False
		now = time.time()

		if frame_updated:
			frame_updated = False
			boxes.clear(); grey_boxes.clear()
			last_frame = frame_number
			# Sample every `step` frames so the LAST sampled (kept) frame lands exactly on
			# last_frame - matches the base_N/step math in Regenerate_annotations.py and
			# BehaveAI_inspect_dataset.py. The previous `last_frame - frameWindow + 1` start
			# (frameWindow already being base_N*step) put the final "keep" slot frame_skip
			# frames short of last_frame whenever frame_skip > 0, so the image actually shown/
			# saved for a given frame_number was really frame_number - frame_skip.
			step = frame_skip + 1
			base_N = raw_buf.maxlen
			start_frame = last_frame - (base_N - 1) * step
			if start_frame < 0:
				original_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
				fr = original_frame.copy()
				motion_image = original_frame.copy()
				raw_buf.clear()
				need_ = True
			else:
				capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
				prev_frames = [None] * 3
				motion_image = None
				frame_count = 0
				raw_buf.clear()
				for i in range((base_N - 1) * step + 1):
					ret, raw_frame = capture.read()
					if not ret:
						break
					if frame_count == 0:
						fr = raw_frame.copy()
						if scale_factor != 1.0:
							fr = cv2.resize(fr, (0,0), fx=scale_factor, fy=scale_factor)
						raw_buf.append(fr.copy())
						gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
						if i == 0:
							prev_frames = [gray.copy()] * 3
							frame_count += 1
							if frame_count > frame_skip:
								frame_count = 0
							continue
						diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
						if strategy == 'exponential':
							prev_frames[0] = gray
							prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, 1-expA, 0)
							prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, 1-expB, 0)
						else:
							prev_frames[2] = prev_frames[1]
							prev_frames[1] = prev_frames[0]
							prev_frames[0] = gray
					frame_count += 1
					if frame_count > frame_skip:
						frame_count = 0
				if 'diffs' in locals():
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
					original_frame = motion_image.copy()
					

					try:
						base = f"{video_label}_{frame_number}"
						boxes, grey_boxes = annotation_index.load_labels_for_basename(base, fr, original_frame)
					except Exception as e:
						print("Error loading saved annotations for", f"{video_label}_{frame_number},", e)
				
					if boxes or grey_boxes:
						# ~ print('Annotations found')
						pass
					else:
						if auto_ann_switch == 1:
							auto_annotate_local()
						need_ = True
						last_anim_draw = time.time()
					
		else:
			if (now - last_mouse_move) > ANIM_STILL_THRESHOLD and (now - last_anim_draw) >= ANIM_DT:
				last_anim_draw = now
				need_ = True

		# recompute display size preserving aspect ratio (main video area only)
		c_w = self.canvas.winfo_width() or 400
		c_h = self.canvas.winfo_height() or 300
		aspect = video_width / video_height
		if c_w / aspect <= c_h:
			disp_w = c_w
			disp_h = int(c_w / aspect)
		else:
			disp_h = c_h
			disp_w = int(c_h * aspect)
		self.display_size = (max(1, int(disp_w)), max(1, int(disp_h)))


		# ensure the temporary rectangle remains visible while mouse is held
		if getattr(self, 'drawing', False):
			need_ = True

		if need_:
			self.redraw()

# Launch app
root = tk.Tk()
app = AnnotatorTk(root)
root.mainloop()

capture.release()
print(f"Done annotating {video_label}")
