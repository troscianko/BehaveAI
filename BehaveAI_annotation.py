#!/usr/bin/env python3


import os
import sys
import time
import cv2
import numpy as np
import configparser
import random
from collections import deque

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from index_annotations import AnnotationIndex


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

# initial selections
active_primary = 0
if len(primary_static_classes) <= 1:
	active_primary = 1
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
	ignore_secondary=ignore_secondary
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
			model_dir = f"model_static_static_{primary_class}"
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


# Helper: convert BGR -> PhotoImage

def cv2_to_photoimage(bgr_img):
	rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	pil = Image.fromarray(rgb)
	return ImageTk.PhotoImage(pil)

## remove overlapping detections	
def non_max_suppression(box_list):
	"""Remove overlapping detections keeping highest confidence box."""
	if len(box_list) == 0:
		return []
	
	# Calculate overall confidence for each box
	confidences = []
	for box in box_list:
		if hierarchical_mode:
			conf = box[6]
		else:
			conf = box[5]

		confidences.append(conf)
	
	# Sort by confidence (descending)
	sorted_indices = sorted(range(len(box_list)), key=lambda i: confidences[i], reverse=True)
	suppressed = [False] * len(box_list)
	keep = []
	
	for i in range(len(sorted_indices)):
		idx_i = sorted_indices[i]
		if suppressed[idx_i]:
			continue
			
		keep.append(box_list[idx_i])
		box_i = box_list[idx_i]
		coords_i = (box_i[0], box_i[1], box_i[2], box_i[3])
		
		for j in range(i + 1, len(sorted_indices)):
			idx_j = sorted_indices[j]
			if suppressed[idx_j]:
				continue
				
			box_j = box_list[idx_j]
			coords_j = (box_j[0], box_j[1], box_j[2], box_j[3])
			
			if iou(coords_i, coords_j) > iou_thresh:
				suppressed[idx_j] = True
				
	return keep



def iou(box1, box2):
	xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
	xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
	inter = max(0, xb-xa) * max(0, yb-ya)
	area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
	area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
	prop1 = inter/area1
	prop2 = inter/area2
	if prop1 > prop2:
		return prop1 if prop1 > 0 else 0
	else:
		return prop2 if prop2 > 0 else 0



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


# Auto-annotate: uses model_static / model_motion and per-primary secondary models
def auto_annotate_local():
	# Collect all primary detections
	# ~ all_detections = []
	global boxes
	
	# Primary static detection
	if primary_static_classes[0] != '0' and model_static != None:
		results_static = model_static.predict(fr, conf=primary_conf_thresh, verbose=False)
		for box in results_static[0].boxes:
			# ~ coords = tuple(map(int, box.xyxy[0].tolist()))
			class_idx = int(box.cls[0])
			primary_class = primary_static_classes[class_idx]
			conf = float(box.conf[0])
			x1, y1, x2, y2 = map(int, box.xyxy[0])

			if hierarchical_mode:
				# crop and run secondary classifier on static image
				if len(secondary_static_classes) >= 2:
					sec_model = secondary_static_models.get(primary_class, None)
					sec_classes = secondary_static_classes
					crop_img = fr
				# Fallback to motion secondary model if static not available
				elif len(secondary_motion_classes) >= 2:
					sec_model = secondary_motion_models.get(primary_class, None)
					sec_classes = secondary_motion_classes
					crop_img = motion_image if primary_motion_classes[0] != '0' else fr

				# Get the cropped region
				crop = None
				if crop_img is not None:
					crop = crop_img[y1:y2, x1:x2]
				
				secondary_class = primary_class
				secondary_conf = 1.0
				secondary_class_idx = -1

				# Run secondary classification if we have a model and valid crop
				if sec_model and crop is not None and crop.size > 0:
					sec_results = sec_model.predict(crop, verbose=False)
					if sec_results[0].probs is not None:
						secondary_class_idx = sec_results[0].probs.top1
						secondary_conf = sec_results[0].probs.top1conf.item()
						secondary_class = sec_model.names[secondary_class_idx]

				boxes.append((x1, y1, x2, y2, class_idx, secondary_class_idx, conf, secondary_conf))#conf 1 & 2 need separating
				

			else:
				boxes.append((x1, y1, x2, y2, class_idx, conf))


	# Primary motion detection
	if primary_motion_classes[0] != '0' and model_motion != None:
		results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
		for box in results_motion[0].boxes:
			class_idx = int(box.cls[0])
			primary_class = primary_motion_classes[class_idx]
			conf = float(box.conf[0])
			x1, y1, x2, y2 = map(int, box.xyxy[0])

			if hierarchical_mode:
				# crop and run secondary classifier on static image
				if len(secondary_static_classes) >= 2:
					sec_model = secondary_static_models.get(primary_class, None)
					sec_classes = secondary_static_classes
					crop_img = fr
				# Fallback to motion secondary model if static not available
				elif len(secondary_motion_classes) >= 2:
					sec_model = secondary_motion_models.get(primary_class, None)
					sec_classes = secondary_motion_classes
					crop_img = motion_image if primary_motion_classes[0] != '0' else fr

				# Get the cropped region
				crop = None
				if crop_img is not None:
					crop = crop_img[y1:y2, x1:x2]
				
				secondary_class = primary_class
				secondary_conf = 1.0
				secondary_class_idx = -1

				# Run secondary classification if we have a model and valid crop
				if sec_model and crop is not None and crop.size > 0:
					sec_results = sec_model.predict(crop, verbose=False)
					if sec_results[0].probs is not None:
						secondary_class_idx = sec_results[0].probs.top1
						secondary_conf = sec_results[0].probs.top1conf.item()
						secondary_class = sec_model.names[secondary_class_idx]

				boxes.append((x1, y1, x2, y2, class_idx + len(primary_static_classes), secondary_class_idx, conf, secondary_conf))#conf 1 & 2 need separating
				
			else:
				boxes.append((x1, y1, x2, y2, class_idx + len(primary_static_classes), conf))

	if boxes:
		boxes = non_max_suppression(boxes)



# draw boxes onto a frame copy
def draw_boxes_on_image(base_img):
	"""
	Draw hierarchical boxes onto a *copy* of base_img.
	- Outer rectangle uses primary color (slightly thicker)
	- Inner rectangle uses secondary color (if present)
	- Label shows PRIMARY conf SECONDARY conf (primary uppercased)
	"""
	out = base_img.copy()
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

			# draw inner box (secondary or primary)
			if primary_classes[primary_cls] not in ignore_secondary:
				cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), scol, line_thickness)

			# compose label: PRIMARY (upper) [+ conf], then secondary [+ conf]
			label = f"{primary_classes[primary_cls].upper()}"
			if conf != -1 and conf is not None:
				try:
					label = label + f" {conf:.2f}"
				except Exception:
					pass
			
			if primary_classes[primary_cls] not in ignore_secondary:	
				if secondary_cls is not None and secondary_cls != -1 and secondary_cls < len(secondary_classes):
					label2 = f"{secondary_classes[secondary_cls]}"
					if sec_conf != -1 and sec_conf is not None:
						try:
							label2 = label2 + f" {sec_conf:.2f}"
						except Exception:
							pass
					label = label + " " + label2

			# draw label background and text
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			lx, ly = int(x1), int(y1)
			cv2.rectangle(out, (lx - line_thickness, ly - label_h - line_thickness*4), (lx + label_w + line_thickness*2, ly), (0,0,0), -1)
			# ~ cv2.putText(out, label, (lx, ly - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, scol, line_thickness, cv2.LINE_AA)
			cv2.putText(out, label, (lx, ly - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, pcol, line_thickness, cv2.LINE_AA)

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
	
	
def save_annotation():
	global annot_count
	if original_frame is None or (not boxes and not grey_boxes) and save_empty_frames == 'false':
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

		# seek bar
		# ~ self.seek = ttk.Scale(root, from_=0, to=max(0, total_frames - 1), orient='horizontal', command=self.on_seek)
		# ~ self.seek.pack(fill='x', padx=4, pady=4)

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
		
		# seek bar fills remaining width
		# ~ self.seek = ttk.Scale(
			# ~ self.controls,
			# ~ from_=0,
			# ~ to=max(0, total_frames - 1),
			# ~ orient='horizontal',
			# ~ command=self.on_seek
		# ~ )
		# ~ self.seek.pack(side='left', fill='x', expand=True, padx=4)
		
		# ~ self.seek.set(frame_number)

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

		# grey toggle
		# ~ self.grey_btn = tk.Button(self.buttons_frame, text="Grey (g)", width=10, command=self.toggle_grey)
		# ~ self.grey_btn.grid(row=2, column=0, padx=2, pady=2)

		# bind events
		self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
		self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
		self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
		self.canvas.bind('<Button-3>', self.on_right_click)
		self.canvas.bind('<Motion>', self.on_motion)

		root.bind_all('<Key>', self.on_key_all)
		# ~ root.bind_all('<Left>', lambda e: self.key_step(-1))
		# ~ root.bind_all('<Right>', lambda e: self.key_step(1))
		root.bind_all('<space>', lambda e: self.toggle_show_mode())
		root.bind_all('<Return>', lambda e: self.key_save())

		# drawing/display state
		self.display_size = (video_width, video_height)
		self.tk_img = None
		self.last_mouse = None
		self.drawing = False
		self.start_canvas_xy = None

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


	# seek
	# ~ def on_seek(self, val):
		# ~ global frame_number, frame_updated
		# ~ try:
			# ~ frame_number = int(float(val))
		# ~ except Exception:
			# ~ frame_number = 0
		# ~ frame_updated = True
		# ~ try:
			# ~ self.frame_var.set(f'Frame {str(frame_number)}')
		# ~ except Exception:
			# ~ pass

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
		self.drawing = True
		self.start_canvas_xy = (event.x, event.y)
		self.last_mouse = (event.x, event.y)

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

	def on_right_click(self, event):
		v = self.canvas_to_video((event.x, event.y))
		x, y = int(v[0]), int(v[1])
		removed = False
		for i in range(len(boxes)-1, -1, -1):
			bx1, by1, bx2, by2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
			if bx1 <= x <= bx2 and by1 <= y <= by2:
				del boxes[i]; removed = True; break
		if not removed:
			for i in range(len(grey_boxes)-1, -1, -1):
				gx1, gy1, gx2, gy2 = grey_boxes[i]
				if gx1 <= x <= gx2 and gy1 <= y <= gy2:
					del grey_boxes[i]; break
		self.redraw()

	def on_motion(self, event):
		self.last_mouse = (event.x, event.y)
		self.redraw()

	# keyboard
	def on_key_all(self, event):
		global active_primary, active_secondary, grey_mode, boxes, grey_boxes, frame_number, frame_updated, show_mode

		ch = event.char
		ks = event.keysym

		# Frame step - step larger when Shift is held (event.state & 0x1 tests Shift mask)
		# ~ if ks == 'Left':
			# ~ step = -10 if (event.state & 0x1) else -1
			# ~ self.key_step(step)
			# ~ return
		# ~ if ks == 'Right':
			# ~ step = 10 if (event.state & 0x1) else 1
			# ~ self.key_step(step)
			# ~ return
			
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
			
			
			
			return

	def key_step(self, delta):
		global frame_number, frame_updated
		frame_number = min(max(0, frame_number + delta), total_frames - 1)
		frame_updated = True
		self.seek.set(frame_number)

	def toggle_show_mode(self):
		global show_mode
		show_mode *= -1

	# ~ def key_save(self):
		# ~ save_annotation()
		# ~ boxes.clear(); grey_boxes.clear()
		# ~ global frame_number, frame_updated
		# ~ frame_number = min(frame_number + 1, total_frames - 1)
		# ~ frame_updated = True
		# ~ self.seek.set(frame_number)

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
		display = draw_boxes_on_image(base)
	
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
	
		def padded_crop(src, cx, cy, crop_size):
			h, w = src.shape[:2]
			x1 = cx - crop_size // 2
			y1 = cy - crop_size // 2
			x2 = x1 + crop_size
			y2 = y1 + crop_size
			sx1 = max(0, x1); sy1 = max(0, y1)
			sx2 = min(w, x2); sy2 = min(h, y2)
			out = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
			if sx2 > sx1 and sy2 > sy1:
				dst_x1 = sx1 - x1
				dst_y1 = sy1 - y1
				dst_x2 = dst_x1 + (sx2 - sx1)
				dst_y2 = dst_y1 + (sy2 - sy1)
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
		scaled_disp_w = max(1, int(round(disp_w * scale)))
		scaled_disp_h = max(1, int(round(disp_h * scale)))
		
		if self.last_mouse is not None:
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
		global frame_updated, fr, original_frame, motion_image, raw_buf, last_mouse_move, last_anim_draw, boxes, grey_boxes

		need_ = False
		now = time.time()

		if frame_updated:
			frame_updated = False
			boxes.clear(); grey_boxes.clear()
			last_frame = frame_number
			start_frame = last_frame - frameWindow + 1
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
				for i in range(frameWindow):
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


		self.root.after(30, self.loop)

# Launch app
root = tk.Tk()
app = AnnotatorTk(root)
root.mainloop()

capture.release()
print("Done annotating video.")
