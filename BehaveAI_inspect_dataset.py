#!/usr/bin/env python3

import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import configparser
import random
import time
from collections import deque
import sys
from PIL import Image, ImageTk
from index_annotations import AnnotationIndex


# Optional YOLO import
try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

# ---------- Determine settings INI path and project directory ----------
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
print(f"Using project directory: {project_dir}")
print(f"Using config file: {config_path}")

# Load config
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

# Read parameters (copied from your inspector; keep consistent)
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

	if len(secondary_motion_classes) >= 2 or len(secondary_static_classes) >= 2:
		hierarchical_mode = True
		motion_cropped_base_dir = 'annot_motion_crop'
		static_cropped_base_dir = 'annot_static_crop'
		if len(secondary_motion_classes) == 1:
			secondary_motion_classes = []
			secondary_motion_colors = []
			secondary_motion_hotkeys = []
		if len(secondary_static_classes) == 1:
			secondary_static_classes = []
			secondary_static_colors = []
			secondary_static_hotkeys = []
	else:
		hierarchical_mode = False
		motion_cropped_base_dir = ""
		static_cropped_base_dir = ""

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
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	motion_blocks_static = config['DEFAULT']['motion_blocks_static'].lower()
	static_blocks_motion = config['DEFAULT']['static_blocks_motion'].lower()
	save_empty_frames = config['DEFAULT']['save_empty_frames'].lower()
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")

# Basic validation
if len(primary_motion_classes) > len(primary_motion_colors) or len(primary_motion_classes) != len(primary_motion_hotkeys):
	raise ValueError("Primary motion classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if len(secondary_motion_classes) > len(secondary_motion_colors) or len(secondary_motion_classes) != len(secondary_motion_hotkeys):
	raise ValueError("Secondary motion classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if len(primary_static_classes) > len(primary_static_colors) or len(primary_static_classes) != len(primary_static_hotkeys):
	raise ValueError("Primary static classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if len(secondary_static_classes) > len(secondary_static_colors) or len(secondary_static_classes) != len(secondary_static_hotkeys):
	raise ValueError("Secondary static classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if motion_blocks_static not in ('true','false'):
	raise ValueError("motion_blocks_static must be true or false")
if static_blocks_motion not in ('true','false'):
	raise ValueError("static_blocks_motion must be true or false")
if save_empty_frames not in ('true','false'):
	raise ValueError("save_empty_frames must be true or false")

expA2 = 1 - expA
expB2 = 1 - expB

primary_classes_info = list(zip(primary_hotkeys, primary_classes))
secondary_classes_info = list(zip(secondary_hotkeys, secondary_classes))
primary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(primary_classes_info)}
secondary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(secondary_classes_info)}
active_primary = 0
if len(primary_static_classes) <= 1:
	active_primary = 1
active_secondary = 0

frameWindow = 4
if strategy == 'exponential':
	if expA > 0.2 or expB > 0.2: frameWindow = 5
	if expA > 0.5 or expB > 0.5: frameWindow = 10
	if expA > 0.7 or expB > 0.7: frameWindow = 15
	if expA > 0.8 or expB > 0.8: frameWindow = 20
	if expA > 0.9 or expB > 0.9: frameWindow = 45

raw_buf = deque(maxlen=frameWindow)

# ~ # create helper index (so we can use centralized listing/loading)
# ~ _ann_index = AnnotationIndex(
	# ~ static_train_images_dir, static_val_images_dir, static_train_labels_dir, static_val_labels_dir,
	# ~ motion_train_images_dir, motion_val_images_dir, motion_train_labels_dir, motion_val_labels_dir,
	# ~ motion_cropped_base_dir, static_cropped_base_dir, clips_dir,
	# ~ primary_static_classes, primary_classes, secondary_classes,
	# ~ hierarchical_mode
# ~ )

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


# ~ items = list_images_labels_and_masks()
items = annotation_index.list_images_labels_and_masks()
if not items:
	print("No annotated images found in the expected dataset directories.")
	print("Checked:", static_train_images_dir, static_val_images_dir, motion_train_images_dir, motion_val_images_dir)
	sys.exit(1)
	
	
# Build list of annotated images
def list_images_labels_and_masks():
	items = {}
	def add_dir(img_dir, lbl_dir):
		if not os.path.isdir(img_dir):
			return
		for fname in os.listdir(img_dir):
			if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
				base = os.path.splitext(fname)[0]
				img_path = os.path.join(img_dir, fname)
				lbl_path = os.path.join(lbl_dir, base + '.txt') if os.path.isdir(lbl_dir) else None
				mask_dir = lbl_dir.replace('labels', 'masks') if lbl_dir else None
				mask_path = os.path.join(mask_dir, base + '.mask.txt') if mask_dir and os.path.isdir(mask_dir) else None
				rec = items.setdefault(base, {})
				if 'static_img' not in rec:
					rec['static_img'] = img_path
					rec['static_lbl'] = lbl_path if lbl_path and os.path.exists(lbl_path) else None
					rec['static_mask'] = mask_path if mask_path and os.path.exists(mask_path) else None
					rec['static_origin_img_dir'] = img_dir
					rec['static_origin_lbl_dir'] = lbl_dir

	add_dir(static_train_images_dir, static_train_labels_dir)
	add_dir(static_val_images_dir, static_val_labels_dir)
	add_dir(motion_train_images_dir, motion_train_labels_dir)
	add_dir(motion_val_images_dir, motion_val_labels_dir)

	ordered = []
	for base, rec in sorted(items.items()):
		ordered.append({'basename': base, **rec})
	return ordered



current_idx = 0

# --- Determine initial sample image size so video_width/video_height exist before UI ---
sample_img_path = items[0].get('static_img') or items[0].get('motion_img')
if not sample_img_path:
	for it in items:
		p = it.get('static_img') or it.get('motion_img')
		if p and os.path.exists(p):
			sample_img_path = p
			break

if sample_img_path and os.path.exists(sample_img_path):
	sample = cv2.imread(sample_img_path)
	if sample is None:
		video_height, video_width = 480, 640
	else:
		video_height, video_width = sample.shape[:2]
else:
	video_height, video_width = 480, 640

right_frame_width = max(96, int(video_height / 3))
ts = cv2.getTextSize("XyZ", cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)[0]
bottom_bar_height = int(ts[1]) + 6 * line_thickness

# global state
drawing = False
cursor_pos = (video_width//2, video_height//2)   # in VIDEO PIXELS (not canvas coords)
ix = iy = -1
boxes = []
grey_boxes = []
frame_updated = True
original_frame = None
fr = None
video_label = ""
annot_count = 1
auto_ann_switch = 1
show_mode = 1
zoom_hide = 0
disp_scale_factor = 1.0
grey_mode = False
last_mouse_move = 0.0
ANIM_STILL_THRESHOLD = 0.5
ANIM_FPS = 8
last_anim_draw = 0.0
ANIM_DT = 1.0 / ANIM_FPS

# helper functions copied/adapted from your inspector code
def norm_to_pixels(xc, yc, bw, bh, w, h):
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p/2); y1 = int(cy - bh_p/2)
	x2 = int(cx + bw_p/2); y2 = int(cy + bh_p/2)
	x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1)); x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
	return x1, y1, x2, y2

def build_window_title(basename):
	elements = ["BehaveAI Annotations (inspect):", basename, "ESC=quit BACKSPACE=clear u=undo ENTER=save SPACE=toggle view LEFT/RIGHT </> seek"]
	return ' '.join(elements)


	if item.get('motion_lbl') and os.path.exists(item['motion_lbl']):
		with open(item['motion_lbl'], 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) < 5:
					continue
				cls = int(parts[0])
				xc, yc, bw, bh = parts[1:5]
				h, w = original_frame.shape[:2]
				x1,y1,x2,y2 = norm_to_pixels(xc, yc, bw, bh, w, h)
				global_primary_cls = cls + len(primary_static_classes)

				if hierarchical_mode:
					boxes.append((x1, y1, x2, y2, global_primary_cls, -1, -1, -1))
				else:
					boxes.append((x1, y1, x2, y2, global_primary_cls, -1))
										
	mask_path = item.get('static_mask') or item.get('motion_mask')
	if mask_path and os.path.exists(mask_path):
		with open(mask_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) >= 4:
					gx1, gy1, gx2, gy2 = map(int, parts[:4])
					grey_boxes.append((gx1, gy1, gx2, gy2))

video_capture = None
video_frame_index = None


def load_item(idx):
	"""Load images, labels, masks, and video preview buffer for given index."""
	global original_frame, fr, boxes, grey_boxes, raw_buf, video_label, video_capture, video_path, video_frame_index, video_height, video_width
	boxes = []
	grey_boxes = []
	raw_buf.clear()
	item = items[idx]
	video_label = item['basename']
	# load static image (fr) and motion image (original_frame) - if one missing copy the other
	static_img = item.get('static_img')
	motion_img = item.get('motion_img')
	# prefer static image to be fr and motion to be original_frame
	fr_img = None
	motion_img_cv = None
	if static_img and os.path.exists(static_img):
		fr_img = cv2.imread(static_img)
	if motion_img and os.path.exists(motion_img):
		motion_img_cv = cv2.imread(motion_img)
	# fallback: if only one exists use it for both
	if fr_img is None and motion_img_cv is None:
		# try to find image by searching both static and motion dirs using basename
		possible = []
		for d in [static_train_images_dir, static_val_images_dir, motion_train_images_dir, motion_val_images_dir]:
			p = os.path.join(d, item['basename'] + '.jpg')
			if os.path.exists(p):
				possible.append(p)
		if possible:
			fr_img = cv2.imread(possible[0])
			motion_img_cv = fr_img.copy()
	else:
		if fr_img is None and motion_img_cv is not None:
			fr_img = motion_img_cv.copy()
		if motion_img_cv is None and fr_img is not None:
			motion_img_cv = fr_img.copy()

	if fr_img is None:
		# create blank placeholder
		fr_img = np.zeros((video_height, video_width, 3), dtype=np.uint8)
	if motion_img_cv is None:
		motion_img_cv = fr_img.copy()

	# optionally resize if scale_factor used in original saving
	if scale_factor != 1.0:
		fr_img = cv2.resize(fr_img, (0, 0), fx=scale_factor, fy=scale_factor)
		motion_img_cv = cv2.resize(motion_img_cv, (0, 0), fx=scale_factor, fy=scale_factor)

	fr = fr_img
	original_frame = motion_img_cv

	# set video dims from loaded images (affects layout)
	video_height, video_width = original_frame.shape[:2]
	# refill raw_buf with static frame repeated (until we try video)
	for _ in range(raw_buf.maxlen):
		raw_buf.append(fr.copy())

	# load labels and masks
	# ~ load_labels_and_masks_for_item(item)
	# ~ boxes, grey_boxes = _ann_index.load_labels_and_masks_for_item(item, fr, original_frame)
	boxes, grey_boxes = annotation_index.load_labels_and_masks_for_item(items[current_idx], fr, original_frame)

	# ----------------- Link secondary crops to primary boxes -----------------
	# Only run when hierarchical mode is enabled
	if hierarchical_mode and boxes:
		# tolerance for small rounding/resizing differences (px)
		MATCH_TOL = 2

		# derive video_label and frame_number from item's basename (split on last underscore)
		if '_' in item['basename']:
			video_label_guess, tail = item['basename'].rsplit('_', 1)
			try:
				frame_number_guess = int(tail)
			except Exception:
				frame_number_guess = None
		else:
			video_label_guess = item['basename']
			frame_number_guess = None

		# build map of boxes keyed by (x1, y1, primary_name) -> list of box indices
		box_index = {}
		for bi, b in enumerate(boxes):
			# hierarchical box structure: (x1,y1,x2,y2, primary_cls, secondary_cls, conf, secondary_conf)
			bx1 = int(round(b[0])); by1 = int(round(b[1]))
			primary_idx = b[4] if len(b) > 4 else None
			primary_name = primary_classes[primary_idx] if primary_idx is not None and primary_idx < len(primary_classes) else None
			key = (bx1, by1, primary_name)
			box_index.setdefault(key, []).append(bi)

		# helper to parse crop filename format: <video_label>_<frame>_<x1>_<y1>.<ext>
		def _parse_crop_filename(fn):
			stem = os.path.splitext(fn)[0]
			parts = stem.split('_')
			if len(parts) < 4:
				return None
			try:
				y1 = int(parts[-1]); x1 = int(parts[-2]); frame = int(parts[-3])
				video_label_part = '_'.join(parts[:-3])
				return video_label_part, frame, x1, y1
			except Exception:
				return None

		# helper to map secondary dir-name to index in secondary_classes
		sec_name_to_idx = {name: idx for idx, name in enumerate(secondary_classes)}

		# search both cropped base dirs (motion then static)
		for base_crop_dir in (motion_cropped_base_dir, static_cropped_base_dir):
			if not base_crop_dir or not os.path.isdir(base_crop_dir):
				continue
			# primary class directories inside the cropped base dir
			for primary_name in os.listdir(base_crop_dir):
				prim_dir = os.path.join(base_crop_dir, primary_name)
				if not os.path.isdir(prim_dir):
					continue
				# secondary class directories under the primary dir
				for secondary_name in os.listdir(prim_dir):
					sec_dir = os.path.join(prim_dir, secondary_name)
					if not os.path.isdir(sec_dir):
						continue
					sec_idx = sec_name_to_idx.get(secondary_name)
					if sec_idx is None:
						# not in configured secondary list, skip
						continue
					# scan crop files
					for fn in os.listdir(sec_dir):
						lower = fn.lower()
						if not lower.endswith(('.jpg', '.jpeg', '.png')):
							continue
						parsed = _parse_crop_filename(fn)
						if parsed is None:
							continue
						vlabel_part, fn_frame, x1_fn, y1_fn = parsed
						# must be same video label and frame
						if vlabel_part != video_label_guess or fn_frame != frame_number_guess:
							continue
						# exact key match first (primary_name must match so we attach secondary to correct primary)
						key = (x1_fn, y1_fn, primary_name)
						matched = False
						if key in box_index:
							for bi in box_index[key]:
								b = boxes[bi]
								# update secondary index in place (preserve other fields)
								if len(b) >= 8:
									boxes[bi] = (b[0], b[1], b[2], b[3], b[4], sec_idx, b[6], b[7])
								else:
									# convert shorter tuple into hierarchical format
									primary_cls = b[4] if len(b) > 4 else 0
									conf = b[6] if len(b) > 6 else -1
									boxes[bi] = (b[0], b[1], b[2], b[3], primary_cls, sec_idx, conf, -1)
								matched = True
						if matched:
							continue
						# if not exact, try small neighbourhood search
						for dx in range(-MATCH_TOL, MATCH_TOL + 1):
							if matched:
								break
							for dy in range(-MATCH_TOL, MATCH_TOL + 1):
								cand = (x1_fn + dx, y1_fn + dy, primary_name)
								if cand in box_index:
									for bi in box_index[cand]:
										b = boxes[bi]
										if len(b) >= 8:
											boxes[bi] = (b[0], b[1], b[2], b[3], b[4], sec_idx, b[6], b[7])
										else:
											primary_cls = b[4] if len(b) > 4 else 0
											conf = b[6] if len(b) > 6 else -1
											boxes[bi] = (b[0], b[1], b[2], b[3], primary_cls, sec_idx, conf, -1)
										matched = True
										break
									if matched:
										break
							if matched:
								break
	# ----------------- END link secondary crops to primary boxes -----------------

	# ----------------- BEGIN: record original secondary crop files for this item -----------------
	# Build a set of full paths for any secondary crop files that exist now for this item.
	# Stored on item['_orig_secondary_crops'] so we can detect deletions later.
	orig_crops = set()
	# parse video_label and frame_number from basename
	if '_' in item['basename']:
		_video_label_part, _tail = item['basename'].rsplit('_', 1)
		try:
			_frame_num = int(_tail)
		except Exception:
			_frame_num = None
	else:
		_video_label_part = item['basename']
		_frame_num = None

	# scan both cropped bases and collect matching filenames for this video/frame
	for base_crop_dir in (motion_cropped_base_dir, static_cropped_base_dir):
		if not base_crop_dir or not os.path.isdir(base_crop_dir):
			continue
		for primary_name in os.listdir(base_crop_dir):
			primary_dir = os.path.join(base_crop_dir, primary_name)
			if not os.path.isdir(primary_dir):
				continue
			for secondary_name in os.listdir(primary_dir):
				sec_dir = os.path.join(primary_dir, secondary_name)
				if not os.path.isdir(sec_dir):
					continue
				for fn in os.listdir(sec_dir):
					low = fn.lower()
					if not low.endswith(('.jpg', '.jpeg', '.png')):
						continue
					# try to parse pattern: <video_label>_<frame>_<x1>_<y1>.<ext>
					stem = os.path.splitext(fn)[0]
					parts = stem.split('_')
					if len(parts) < 4:
						continue
					try:
						y1_fn = int(parts[-1]); x1_fn = int(parts[-2]); frame_fn = int(parts[-3])
						video_label_part_fn = '_'.join(parts[:-3])
					except Exception:
						continue
					# only keep files for this item (same video label + same frame)
					if video_label_part_fn == _video_label_part and frame_fn == _frame_num:
						full = os.path.join(sec_dir, fn)
						orig_crops.add(full)

	# attach to item for later comparison in save
	item['_orig_secondary_crops'] = orig_crops
	# ----------------- END: record original secondary crop files for this item -----------------


	# try to find and load video preview frames (replicating original sampling behaviour)
	# ~ video_path_found, guessed_frame = find_video_for_item(item)
	# ~ video_path_found, guessed_frame = _ann_index.find_video_for_item(item)
	video_path_found, guessed_frame = annotation_index.find_video_for_item(item)
	video_capture = None
	video_frame_index = guessed_frame
	if video_path_found and os.path.exists(video_path_found):
		try:
			video_capture = cv2.VideoCapture(video_path_found)

			# If guessed_frame is known and capture opened, we want to build a buffer
			# of base_N frames that *end* at guessed_frame (i.e. guessed_frame is the last frame).
			if guessed_frame is not None and video_capture.isOpened():
				total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
				base_N = raw_buf.maxlen				 # frameWindow_base (number of frames we want)
				step = frame_skip + 1				  # sampling interval
				total_to_read = base_N * step		  # how many raw frames to read in worst case

				def _read_buffer_ending_at(last_frame):
					"""
					Read frames so that the returned buffer contains up to `base_N` frames
					sampled every `step` frames, and the buffer *ends* at `last_frame`.
					Returns (buf_list, start_frame_used, last_index_appended).
					"""
					# compute start index so that last appended index would be last_frame
					# appended indices will be: s + 0, s + step, s + 2*step, ..., s + (len(buf)-1)*step
					start_frame = int(last_frame - (base_N - 1) * step)
					# clamp start
					start_frame = max(0, min(start_frame, max(0, total - 1)))
					video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
					buf = []
					read_count = 0
					idx = start_frame
					# read up to total_to_read raw frames, appending every 'step' frames
					while read_count < total_to_read:
						ret, f = video_capture.read()
						if not ret:
							break
						if (read_count % step) == 0:
							if scale_factor != 1.0:
								f = cv2.resize(f, (0, 0), fx=scale_factor, fy=scale_factor)
							buf.append(f.copy())
							if len(buf) >= base_N:
								# compute last appended frame index
								last_appended = start_frame + ( (len(buf) - 1) * step )
								return buf, start_frame, last_appended
						read_count += 1
						idx += 1
						if idx > total - 1:
							break
					# if we didn't reach base_N, compute last appended appropriately
					if buf:
						last_appended = start_frame + ( (len(buf) - 1) * step )
					else:
						last_appended = start_frame - 1
					return buf, start_frame, last_appended

				# Try a few candidates (centered on guessed_frame) to allow for small offsets.
				# Each candidate is a last-frame index we'd like the buffer to end at.
				candidates = [guessed_frame - 1, guessed_frame, guessed_frame + 1]
				best_buf = None
				best_start = None
				best_last = None
				best_score = float('inf')

				def _frame_diff(a, b):
					try:
						ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
						gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
						return float(np.mean(np.abs(ga - gb)))
					except Exception:
						return float('inf')

				for cand_last in candidates:
					# skip out-of-range candidates
					if cand_last < 0 or cand_last > total - 1:
						continue
					buf, s, last_idx = _read_buffer_ending_at(cand_last)
					if not buf:
						continue
					# IMPORTANT: compare the LAST buffer frame to fr, because basenames
					# now store the *last* frame of the motion window.
					if fr is not None:
						score = _frame_diff(buf[-1], fr)
					else:
						score = 0.0
					if score < best_score:
						best_score = score
						best_buf = buf
						best_start = s
						best_last = last_idx

				# fallback: try guessed_frame directly if none chosen
				if best_buf is None:
					buf, s, last_idx = _read_buffer_ending_at(guessed_frame)
					if buf:
						best_buf, best_start, best_last = buf, s, last_idx

				if best_buf is not None:
					raw_buf.clear()
					for f in best_buf:
						raw_buf.append(f)
					# video_frame_index is the last appended frame index (i.e. the saved frame number)
					video_frame_index = best_last

		except Exception:
			video_capture = None



# populate motion_img keys for items (unchanged)
for it in items:
	base = it['basename']
	p1 = os.path.join(motion_train_images_dir, base + '.jpg')
	p2 = os.path.join(motion_val_images_dir, base + '.jpg')
	if os.path.exists(p1):
		it['motion_img'] = p1; it['motion_lbl'] = os.path.join(motion_train_labels_dir, base + '.txt'); it['motion_mask'] = os.path.join(motion_train_labels_dir.replace('labels','masks'), base + '.mask.txt')
	elif os.path.exists(p2):
		it['motion_img'] = p2; it['motion_lbl'] = os.path.join(motion_val_labels_dir, base + '.txt'); it['motion_mask'] = os.path.join(motion_val_labels_dir.replace('labels','masks'), base + '.mask.txt')
	if it.get('static_img'):
		base_lbl = os.path.join(it['static_origin_lbl_dir'], it['basename'] + '.txt')
		it['static_lbl'] = base_lbl if os.path.exists(base_lbl) else None
		base_mask = os.path.join(it['static_origin_lbl_dir'].replace('labels','masks'), it['basename'] + '.mask.txt')
		it['static_mask'] = base_mask if os.path.exists(base_mask) else None

# load first item synchronously so UI has initial data
load_item(current_idx)
print("Starting inspection of annotation dataset. Items found:", len(items))

# Drawing helpers (adapted)
def draw_boxes(frame):
	for box in boxes:
		if hierarchical_mode:
			x1, y1, x2, y2, primary_cls, secondary_cls, conf, secondary_conf = box
			x1 = int(x1 * disp_scale_factor); y1 = int(y1 * disp_scale_factor)
			x2 = int(x2 * disp_scale_factor); y2 = int(y2 * disp_scale_factor)
			if primary_classes[primary_cls] in ignore_secondary:
				label = f"{primary_classes[primary_cls].upper()}"
				if conf != -1:
					label = label + f' {conf:.2f}'
				label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
				label_w, label_h = label_size
				cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
				cv2.rectangle(frame, (x1, y1), (x2, y2), primary_colors[primary_cls], line_thickness)
				cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_colors[primary_cls], line_thickness, cv2.LINE_AA)
			else:
				outer_thickness = line_thickness + 2
				cv2.rectangle(frame, (x1-outer_thickness, y1-outer_thickness), (x2+outer_thickness, y2+outer_thickness), primary_colors[primary_cls], outer_thickness)
				label = f"{primary_classes[primary_cls].upper()}"
				# ~ if conf != -1:
					# ~ label = label + f' {conf:.2f}'
				label = label + f" {secondary_classes[secondary_cls]}"
				# ~ if secondary_conf != -1:
					# ~ label = label + f' {secondary_conf:.2f}'
				label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
				label_w, label_h = label_size
				cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
				cv2.rectangle(frame, (x1, y1), (x2, y2), secondary_colors[secondary_cls], line_thickness)
				cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, font_size, secondary_colors[secondary_cls], line_thickness, cv2.LINE_AA)
		else:
			x1, y1, x2, y2, primary_cls, conf = box
			x1 = int(x1 * disp_scale_factor); y1 = int(y1 * disp_scale_factor)
			x2 = int(x2 * disp_scale_factor); y2 = int(y2 * disp_scale_factor)
			label = f"{primary_classes[primary_cls]}"
			if conf != -1:
				label = label + f' {conf:.2f}'
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
			cv2.rectangle(frame, (x1, y1), (x2, y2), primary_colors[primary_cls], line_thickness)
			cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_colors[primary_cls], line_thickness, cv2.LINE_AA)

	for gx1, gy1, gx2, gy2 in grey_boxes:
		overlay = frame.copy()
		cv2.rectangle(overlay, (int(gx1*disp_scale_factor), int(gy1*disp_scale_factor)), (int(gx2*disp_scale_factor), int(gy2*disp_scale_factor)), (128, 128, 128), -line_thickness)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


def draw_zoom(disp, cursor_pos_in):
	"""
	Right-hand zoom column:
	 - top = static (2x)
	 - mid  = motion (2x)
	 - bot  = animation (1x, updates continuously)
	Uses padded crops so off-screen areas are black.
	"""
	if cursor_pos_in is None:
		return
	cx, cy = cursor_pos_in
	cx = int(cx); cy = int(cy)
	h = int(video_height); w = int(video_width)

	# widget size chosen relative to video height (same idea as annotation redraw)
	widget_size = max(32, int(h / 3))

	# magnifications
	MAG = 2.0
	MAG_ANIM = 1.0

	# compute crop sizes in VIDEO pixels (smaller crop -> magnified when resized to widget_size)
	crop_vid = max(2, int(round(widget_size / MAG)))		 # top/mid  -> 2x
	crop_vid_anim = max(2, int(round(widget_size / MAG_ANIM)))  # anim -> 1x (same size)

	# padded crop helper (returns crop and original crop box (x1,y1,x2,y2) in video coords)
	def padded_crop(src, cx, cy, crop_size):
		h_src, w_src = src.shape[:2]
		x1 = int(cx - crop_size // 2)
		y1 = int(cy - crop_size // 2)
		x2 = x1 + crop_size
		y2 = y1 + crop_size
		sx1 = max(0, x1); sy1 = max(0, y1)
		sx2 = min(w_src, x2); sy2 = min(h_src, y2)
		out = np.zeros((crop_size, crop_size, 3), dtype=src.dtype)
		if sx2 > sx1 and sy2 > sy1:
			dst_x1 = sx1 - x1
			dst_y1 = sy1 - y1
			dst_x2 = dst_x1 + (sx2 - sx1)
			dst_y2 = dst_y1 + (sy2 - sy1)
			out[dst_y1:dst_y2, dst_x1:dst_x2] = src[sy1:sy2, sx1:sx2]
		return out, (x1, y1, x2, y2)

	# --- top zoom (static) ---
	z_top = None
	if fr is not None:
		crop_img, crop_box = padded_crop(fr, cx, cy, crop_vid)
		z_top = cv2.resize(crop_img, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
		rel_x = cx - crop_box[0]; rel_y = cy - crop_box[1]
		if 0 <= rel_x < crop_vid and 0 <= rel_y < crop_vid:
			zx = int(round(rel_x * widget_size / crop_vid))
			zy = int(round(rel_y * widget_size / crop_vid))
			# single-pixel crosshair inside zoom pane
			cv2.line(z_top, (0, zy), (widget_size-1, zy), (255,255,255), 1)
			cv2.line(z_top, (zx, 0), (zx, widget_size-1), (255,255,255), 1)
		# hairline black border
		cv2.rectangle(z_top, (0, 0), (widget_size-1, widget_size-1), (0,0,0), 1)

	# --- mid zoom (motion) ---
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
		cv2.rectangle(z_mid, (0, 0), (widget_size-1, widget_size-1), (0,0,0), 1)

	# --- bottom zoom (animation, 1x) ---
	z_bot = None
	if len(raw_buf) == raw_buf.maxlen and len(raw_buf) > 0:
		# always update animation (no gating)
		idx = int(((time.time() - last_mouse_move) * ANIM_FPS) % raw_buf.maxlen)
		small = raw_buf[idx]
		small_crop, crop_box = padded_crop(small, cx, cy, crop_vid_anim)
		# resize to widget_size (crop_vid_anim == widget_size so this is 1x or nearest)
		z_bot = cv2.resize(small_crop, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
	else:
		z_bot = np.zeros((widget_size, widget_size, 3), dtype=np.uint8)
	cv2.rectangle(z_bot, (0, 0), (widget_size-1, widget_size-1), (0,0,0), 1)

	# --- place zooms immediately to the right of the main display (no gap) ---
	# compute placement in the disp image (disp is in display pixels already)
	# use disp_scale_factor to compute pixel offset for main display width
	pos_x = int(round(video_width * disp_scale_factor))
	pos_y = 0

	h_disp, w_disp = disp.shape[:2]
	# place top
	if z_top is not None:
		zh, zw = z_top.shape[:2]
		if pos_x + zw <= w_disp and pos_y + zh <= h_disp:
			disp[pos_y:pos_y+zh, pos_x:pos_x+zw] = z_top[0:zh, 0:zw]
	# place mid
	if z_mid is not None:
		zh, zw = z_mid.shape[:2]
		y_off = pos_y + widget_size
		if pos_x + zw <= w_disp and y_off + zh <= h_disp:
			disp[y_off:y_off+zh, pos_x:pos_x+zw] = z_mid[0:zh, 0:zw]
	# place bot (animation)
	if z_bot is not None:
		zh, zw = z_bot.shape[:2]
		y_off = pos_y + 2 * widget_size
		if pos_x + zw <= w_disp and y_off + zh <= h_disp:
			disp[y_off:y_off+zh, pos_x:pos_x+zw] = z_bot[0:zh, 0:zw]

	
	

def refresh_display():
	global original_frame, fr, cursor_pos, disp_scale_factor
	if original_frame is None or fr is None:
		return None
	# cursor_pos is in VIDEO PIXELS already
	x, y = cursor_pos
	h, w = original_frame.shape[:2]
	# build composite (video area + bottom bar + right zoom column)
	canvas = np.zeros((video_height + bottom_bar_height, video_width + right_frame_width + line_thickness, 3), dtype=original_frame.dtype)
	canvas[:video_height,:video_width] = (original_frame if show_mode == 1 else fr)
	disp = canvas
	if disp_scale_factor != 1.0:
		disp = cv2.resize(disp, None, fx=disp_scale_factor, fy=disp_scale_factor, interpolation=cv2.INTER_LINEAR)
	draw_boxes(disp)
	draw_zoom(disp, cursor_pos)

	# draw crosshair limited to the *main* video area (do not cross the right zoom column)
	try:
		cx = int(round(cursor_pos[0] * disp_scale_factor))
		cy = int(round(cursor_pos[1] * disp_scale_factor))
		h_disp, w_disp = disp.shape[:2]
		# main video area size in disp coordinates
		main_w = int(round(video_width * disp_scale_factor))
		main_h = int(round(video_height * disp_scale_factor))
		# only draw vertical line inside main_w and between 0..main_h
		if 0 <= cx < main_w and 0 <= cy < main_h:
			cv2.line(disp, (cx, 0), (cx, main_h), (255,255,255), max(1, line_thickness))
			cv2.line(disp, (0, cy), (main_w, cy), (255,255,255), max(1, line_thickness))
	except Exception:
		pass
	return disp


# ---------------------------------------------------------------------------
# ---------- SAVING: overwrite the *same* files we loaded --------------------
# ---------------------------------------------------------------------------
def save_annotation_and_overwrite_current():
	"""Overwrite the image(s), label(s) and mask(s) for the current item with the modified boxes/grey_boxes.
	   Respects original origin directories so we don't shuffle train/val assignments.
	"""
	global items, current_idx, fr, original_frame, boxes, grey_boxes, annot_count
	item = items[current_idx]
	base = item['basename']
	# Determine target paths (use origin dirs stored earlier)
	static_img_dir = item.get('static_origin_img_dir')
	static_lbl_dir = item.get('static_origin_lbl_dir')
	motion_img_dir = None
	motion_lbl_dir = None

	# if the item originally had motion image, use that origin; else try to guess by checking both motion dirs
	if item.get('motion_img'):
		if motion_train_images_dir in item.get('motion_img', ''):
			motion_img_dir = motion_train_images_dir; motion_lbl_dir = motion_train_labels_dir
		else:
			motion_img_dir = motion_val_images_dir; motion_lbl_dir = motion_val_labels_dir
	else:
		# if no motion origin known, keep image where static exists (fallback)
		if static_img_dir and 'annot_motion' in static_img_dir:
			motion_img_dir = static_img_dir
			motion_lbl_dir = static_lbl_dir

	# write static image & labels (if original static path is present OR there is at least one static-class box)
	h, w = original_frame.shape[:2]

	# produce static annotated frame (grey areas applied)
	static_ann_frame = fr.copy()
	motion_ann_frame = original_frame.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)
		cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	# Count static vs motion boxes (based on primary class index)
	static_count = 0; motion_count = 0
	for box in boxes:
		if hierarchical_mode:
			x1, y1, x2, y2, primary_cls, _, _, _ = box
		else:
			x1, y1, x2, y2, primary_cls, _ = box
		if primary_cls < len(primary_static_classes):
			static_count += 1
			if static_blocks_motion == 'true':
				cv2.rectangle(motion_ann_frame, (x1, y1), (x2, y2), (128,128,128), -line_thickness)
		else:
			motion_count += 1
			if motion_blocks_static == 'true':
				cv2.rectangle(static_ann_frame, (x1, y1), (x2, y2), (128,128,128), -line_thickness)

	# write images (overwrite)
	if static_img_dir and (static_count > 0 or save_empty_frames == 'true'):
		out_static_img_path = os.path.join(static_img_dir, base + '.jpg')
		cv2.imwrite(out_static_img_path, static_ann_frame)
		# write static labels
		if static_lbl_dir:
			out_static_lbl = os.path.join(static_lbl_dir, base + '.txt')
			with open(out_static_lbl, 'w') as f:
				for box in boxes:
					if hierarchical_mode:
						x1, y1, x2, y2, primary_cls, _, _, _ = box
					else:
						x1, y1, x2, y2, primary_cls, _ = box
					if primary_cls < len(primary_static_classes):
						xc = (x1 + x2) / 2 / w
						yc = (y1 + y2) / 2 / h
						bw = abs(x2 - x1) / w
						bh = abs(y2 - y1) / h
						f.write(f"{primary_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
	# write motion images & labels
	if motion_img_dir and (motion_count > 0 or save_empty_frames == 'true'):
		out_motion_img_path = os.path.join(motion_img_dir, base + '.jpg')
		cv2.imwrite(out_motion_img_path, motion_ann_frame)
		if motion_lbl_dir:
			out_motion_lbl = os.path.join(motion_lbl_dir, base + '.txt')
			with open(out_motion_lbl, 'w') as f:
				for box in boxes:
					if hierarchical_mode:
						x1, y1, x2, y2, primary_cls, _, _, _ = box
					else:
						x1, y1, x2, y2, primary_cls, _ = box
					if primary_cls >= len(primary_static_classes):
						cls_in_file = primary_cls - len(primary_static_classes)
						xc = (x1 + x2) / 2 / w
						yc = (y1 + y2) / 2 / h
						bw = abs(x2 - x1) / w
						bh = abs(y2 - y1) / h
						f.write(f"{cls_in_file} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

	# write mask files (to both static & motion mask dirs if both present; prefer existing dirs)
	mask_content = ""
	for gx1, gy1, gx2, gy2 in grey_boxes:
		mask_content += f"{gx1} {gy1} {gx2} {gy2}\n"
	# static mask
	if static_lbl_dir:
		static_mask_dir = static_lbl_dir.replace('labels', 'masks')
		os.makedirs(static_mask_dir, exist_ok=True)
		static_mask_path = os.path.join(static_mask_dir, base + '.mask.txt')
		with open(static_mask_path, 'w') as f:
			f.write(mask_content)
	# motion mask
	if motion_lbl_dir:
		motion_mask_dir = motion_lbl_dir.replace('labels', 'masks')
		os.makedirs(motion_mask_dir, exist_ok=True)
		motion_mask_path = os.path.join(motion_mask_dir, base + '.mask.txt')
		with open(motion_mask_path, 'w') as f:
			f.write(mask_content)


	# ----------------- BEGIN: create/update secondary crop files for current boxes -----------------
	# When in hierarchical_mode, write cropped images for each box that has a valid secondary index.
	try:
		if hierarchical_mode:
			base = item['basename']  # already in the "<video>_<frame>" format
			created_crops = set()
			for b in boxes:
				# unpack robustly for both hierarchical and non-hierarchical formats
				if len(b) >= 6:
					x1_b = int(round(b[0])); y1_b = int(round(b[1])); x2_b = int(round(b[2])); y2_b = int(round(b[3]))
					primary_idx = int(b[4])
					secondary_idx = int(b[5])
				else:
					continue

				# skip if secondary not assigned
				if secondary_idx is None or secondary_idx < 0:
					continue

				# safety checks for indices
				if primary_idx < 0 or primary_idx >= len(primary_classes):
					continue
				if secondary_idx < 0 or secondary_idx >= len(secondary_classes):
					continue

				primary_name = primary_classes[primary_idx]
				secondary_name = secondary_classes[secondary_idx]

				# build expected filename exactly as original annot script used
				fname = f"{base}_{x1_b}_{y1_b}.jpg"

				# motion crop (if motion_cropped_base_dir exists)
				if motion_cropped_base_dir:
					m_dir = os.path.join(motion_cropped_base_dir, primary_name, secondary_name)
					os.makedirs(m_dir, exist_ok=True)
					m_path = os.path.join(m_dir, fname)
					try:
						crop = motion_ann_frame[y1_b:y2_b, x1_b:x2_b]
						if hasattr(crop, "size") and crop.size:
							cv2.imwrite(m_path, crop)
							created_crops.add(m_path)
					except Exception as e:
						# best-effort: continue on error
						print(f"Warning writing motion crop {m_path}: {e}")

				# static crop (if static_cropped_base_dir exists)
				if static_cropped_base_dir:
					s_dir = os.path.join(static_cropped_base_dir, primary_name, secondary_name)
					os.makedirs(s_dir, exist_ok=True)
					s_path = os.path.join(s_dir, fname)
					try:
						crop = static_ann_frame[y1_b:y2_b, x1_b:x2_b]
						if hasattr(crop, "size") and crop.size:
							cv2.imwrite(s_path, crop)
							created_crops.add(s_path)
					except Exception as e:
						print(f"Warning writing static crop {s_path}: {e}")

			# ensure the item's record of original secondary crops includes newly created files
			orig = item.get('_orig_secondary_crops', set())
			# normalize to absolute paths (orig stored as absolute earlier)
			orig.update(created_crops)
			item['_orig_secondary_crops'] = orig
	except Exception as e:
		print(f"Warning during secondary-crop creation: {e}")
	# ----------------- END: create/update secondary crop files for current boxes -----------------


	# ----------------- BEGIN: remove deleted secondary crop files -----------------
	# Delete any secondary crop images that were present when we loaded this item
	# but are no longer matched to a surviving box. This removes from both
	# motion_cropped_base_dir and static_cropped_base_dir locations.
	try:
		item = items[current_idx]
		orig_crops = item.get('_orig_secondary_crops', set())
		# parse video_label and frame_number from basename
		if '_' in item['basename']:
			video_label_part, tail = item['basename'].rsplit('_', 1)
			try:
				frame_num = int(tail)
			except Exception:
				frame_num = None
		else:
			video_label_part = item['basename']
			frame_num = None

		# build current expected crop file paths from boxes (both motion & static crop dirs)
		current_crops = set()
		if hierarchical_mode:
			for b in boxes:
				# box format: (x1, y1, x2, y2, primary_cls, secondary_cls, ..., ...)
				if len(b) >= 6:
					x1_b = int(round(b[0])); y1_b = int(round(b[1]))
					primary_idx = b[4]; secondary_idx = b[5]
					# only consider boxes that have an assigned secondary
					if secondary_idx is None or secondary_idx < 0:
						continue
					# find names
					if primary_idx is None or primary_idx >= len(primary_classes):
						continue
					if secondary_idx >= len(secondary_classes):
						continue
					primary_name = primary_classes[primary_idx]
					secondary_name = secondary_classes[secondary_idx]
					# expected filename
					if frame_num is None:
						continue
					fname = f"{video_label_part}_{frame_num}_{x1_b}_{y1_b}.jpg"
					# both crop locations (motion & static) may contain the files
					m_path = os.path.join(motion_cropped_base_dir, primary_name, secondary_name, fname) if motion_cropped_base_dir else None
					s_path = os.path.join(static_cropped_base_dir, primary_name, secondary_name, fname) if static_cropped_base_dir else None
					if m_path: current_crops.add(m_path)
					if s_path: current_crops.add(s_path)

		# files to delete = orig - current
		to_delete = orig_crops - current_crops
		if to_delete:
			for p in sorted(to_delete):
				try:
					if os.path.exists(p):
						os.remove(p)
						# try to rmdir parent if empty
						parent = os.path.dirname(p)
						try:
							if os.path.isdir(parent) and not os.listdir(parent):
								os.rmdir(parent)
						except Exception:
							# ignore dir-remove errors (concurrent files etc.)
							pass
				except Exception as e:
					# best-effort: print error but continue
					print(f"Warning: could not remove crop {p}: {e}")
		# update stored original set to match current (so subsequent saves are incremental)
		item['_orig_secondary_crops'] = (orig_crops - to_delete) | (current_crops & orig_crops)  # keep only existing ones
	except Exception as e:
		print(f"Warning during secondary-crop cleanup: {e}")
	# ----------------- END: remove deleted secondary crop files -----------------


	print(f"Saved and overwrote annotation for {base}")
	annot_count += 1


# cv2 -> PhotoImage helper
def cv2_to_photoimage(bgr_img):
	rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	pil = Image.fromarray(rgb)
	return ImageTk.PhotoImage(pil)

# ---------- Tk UI ----------
class DatasetInspectorTk:
	def __init__(self, root):
		self.root = root
		root.title("BehaveAI — Annotation Inspector")
		# default larger window so panel visible
		root.geometry("1200x900")

		self.main = tk.Frame(root)
		self.main.pack(fill='both', expand=True)

		# left area using grid so bottom controls are fixed
		self.left = tk.Frame(self.main)
		self.left.pack(side='left', fill='both', expand=True)
		self.left.columnconfigure(0, weight=1)
		self.left.rowconfigure(0, weight=1)  # canvas row stretches

		canvas_w = min(1400, video_width + right_frame_width + 60)
		canvas_h = min(1000, video_height + 180)
		self.canvas = tk.Canvas(self.left, bg='black', highlightthickness=0,
								width=canvas_w, height=canvas_h)
		self.canvas.grid(row=0, column=0, sticky='nsew')

		# bottom frame: fixed height / no expanding
		self.bottom_frame = tk.Frame(self.left)
		self.bottom_frame.grid(row=1, column=0, sticky='ew')
		self.bottom_frame.columnconfigure(1, weight=1)

		# controls row (grey + seek)
		self.controls = tk.Frame(self.bottom_frame)
		self.controls.pack(fill='x', padx=4, pady=(4,0))

		self.grey_btn = tk.Button(self.controls, text="Grey (g)", width=10, command=self.toggle_grey)
		self.grey_btn.pack(side='left', padx=(2,4))

		self.seek = ttk.Scale(self.controls, from_=0, to=max(0, len(items)-1),
							  orient='horizontal', command=self.on_seek)
		self.seek.pack(side='left', fill='x', expand=True, padx=(0,4))

		# status label with current basename
		self.status_var = tk.StringVar()
		self.status_label = tk.Label(self.bottom_frame, textvariable=self.status_var, anchor='w')
		self.status_label.pack(fill='x', padx=4, pady=(2,4))

		# buttons frame: class buttons at absolute bottom
		self.buttons_frame = tk.Frame(self.bottom_frame)
		self.buttons_frame.pack(side='bottom', fill='x', pady=(4,4))

		self.primary_buttons = []
		self.secondary_buttons = []

		col = 0
		for idx, name in enumerate(primary_classes):
			if name == '0': continue
			color_hex = None
			if idx < len(primary_colors):
				bgr = primary_colors[idx]
				color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
			btn = tk.Button(self.buttons_frame, text="{} ({})".format(name, primary_classes_info[idx][0]),
							width=12, relief='raised', command=lambda i=idx: self.select_primary(i))
			btn.grid(row=0, column=col, padx=2, pady=2)
			self.primary_buttons.append((btn, color_hex, idx))
			col += 1

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

		# bind events to canvas (we convert canvas -> video coords inside handlers)
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

		self.display_size = (video_width, video_height)
		self.tk_img = None
		self.last_mouse = None		# canvas coords
		self.drawing = False
		self.start_canvas_xy = None
		self.composite_scale = 1.0

		self.seek.set(current_idx)
		self.update_status()
		self.update_button_states()

		self.root.after(30, self.loop)

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
		self.redraw()

	def update_button_states(self):
		for btn, col, cls in self.primary_buttons:
			if cls == active_primary:
				btn.config(relief='sunken')
				if col:
					try: btn.config(bg=col)
					except Exception: pass
			else:
				btn.config(relief='raised', bg='#888888')
		for btn, col, cls in self.secondary_buttons:
			if cls == active_secondary:
				btn.config(relief='sunken')
				if col:
					try: btn.config(bg=col)
					except Exception: pass
			else:
				btn.config(relief='raised', bg='#888888')
		self.grey_btn.config(relief='sunken' if grey_mode else 'raised')

	def on_seek(self, val):
		global current_idx, frame_updated
		try:
			idx = int(float(val))
		except Exception:
			idx = 0
		if idx != current_idx:
			current_idx = idx
			load_item(current_idx)
			frame_updated = True
			self.update_status()
			self.seek.set(current_idx)

	def update_status(self):
		try:
			self.status_var.set(items[current_idx]['basename'])
		except Exception:
			self.status_var.set("")

	def canvas_to_video(self, canvas_point):
		# Map canvas coords (event.x,event.y) -> video pixel coords (vx,vy)
		cx, cy = canvas_point
		# compute composite (disp) size in pixels (video area + right column + bottom bar)
		disp_w = video_width + right_frame_width + line_thickness
		disp_h = video_height + bottom_bar_height
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1
		scale_w = float(c_w) / float(max(1, disp_w))
		scale_h = float(c_h) / float(max(1, disp_h))
		scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else 1.0
		# Top-left anchored, so canvas x,y map to scaled image coords
		display_x = min(max(0, cx), int(round(disp_w * scale)) - 1) / scale
		display_y = min(max(0, cy), int(round(disp_h * scale)) - 1) / scale
		vx = int(round(display_x * (video_width / float(max(1, video_width)))))
		vy = int(round(display_y * (video_height / float(max(1, video_height)))))
		return (vx, vy)

	def video_to_canvas(self, vx, vy):
		# map video pixels to canvas coords using current composite scaling
		disp_w = video_width + right_frame_width + line_thickness
		disp_h = video_height + bottom_bar_height
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1
		scale_w = float(c_w) / float(max(1, disp_w))
		scale_h = float(c_h) / float(max(1, disp_h))
		scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else 1.0
		cx = int(round(vx * (video_width / float(max(1, video_width))) * scale))
		cy = int(round(vy * (video_height / float(max(1, video_height))) * scale))
		return (cx, cy)

	def on_mouse_down(self, event):
		self.drawing = True
		self.start_canvas_xy = (event.x, event.y)
		self.last_mouse = (event.x, event.y)
		global ix, iy, drawing, cursor_pos
		drawing = True
		# compute video coords and store
		vx, vy = self.canvas_to_video((event.x,event.y))
		ix, iy = int(vx), int(vy)
		cursor_pos = (ix, iy)

	def on_mouse_drag(self, event):
		if not self.drawing:
			return
		self.last_mouse = (event.x, event.y)
		vx, vy = self.canvas_to_video((event.x,event.y))
		global cursor_pos
		cursor_pos = (int(vx), int(vy))
		self.redraw(temp_rect=(self.start_canvas_xy, (event.x, event.y)))

	def on_mouse_up(self, event):
		if not self.drawing:
			return
		self.drawing = False
		global ix, iy, boxes, grey_boxes, drawing, cursor_pos
		drawing = False
		start_v = self.canvas_to_video(self.start_canvas_xy)
		end_v = self.canvas_to_video((event.x, event.y))
		x1, x2 = sorted([int(round(start_v[0])), int(round(end_v[0]))])
		y1, y2 = sorted([int(round(start_v[1])), int(round(end_v[1]))])
		x1 = max(0, min(video_width-1, x1)); x2 = max(0, min(video_width-1, x2))
		y1 = max(0, min(video_height-1, y1)); y2 = max(0, min(video_height-1, y2))
		cursor_pos = (int(round(end_v[0])), int(round(end_v[1])))
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
		vx, vy = self.canvas_to_video((event.x, event.y))
		x, y = int(vx), int(vy)
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
		global last_mouse_move, cursor_pos
		self.last_mouse = (event.x, event.y)
		last_mouse_move = time.time()
		# convert canvas coords to video coords and store for zoom/crosshair
		vx, vy = self.canvas_to_video((event.x, event.y))
		cursor_pos = (int(vx), int(vy))
		self.redraw()

	def on_key_all(self, event):
		global active_primary, active_secondary, grey_mode, boxes, grey_boxes, current_idx
		ch = event.char
		ks = event.keysym

		# step larger when Shift is held (event.state & 0x1 tests Shift mask)
		if ks == 'Left':
			step = -10 if (event.state & 0x1) else -1
			self.key_step(step)
			return
		if ks == 'Right':
			step = 10 if (event.state & 0x1) else 1
			self.key_step(step)
			return

		if ch:
			try:
				c_ord = ord(ch)
			except Exception:
				c_ord = None
			if c_ord and c_ord in primary_class_dict and c_ord in secondary_class_dict:
				if ch != '0':
					active_primary = primary_class_dict[c_ord]
					active_secondary = secondary_class_dict[c_ord]
					grey_mode = False
					self.update_button_states(); self.redraw(); return
			if c_ord and c_ord in primary_class_dict:
				if ch != '0':
					active_primary = primary_class_dict[c_ord]
					grey_mode = False
					self.update_button_states(); self.redraw(); return
			if c_ord and c_ord in secondary_class_dict:
				if ch != '0':
					active_secondary = secondary_class_dict[c_ord]
					grey_mode = False
					self.update_button_states(); self.redraw(); return

		if ch == 'u':
			if grey_mode:
				if grey_boxes: grey_boxes.pop()
			elif boxes:
				boxes.pop()
			self.redraw(); return
		if ch == 'g':
			self.toggle_grey(); return
		if ks == 'Return':
			save_annotation_and_overwrite_current()
			boxes.clear(); grey_boxes.clear()
			global current_idx
			current_idx = min(current_idx + 1, len(items) - 1)
			load_item(current_idx)
			self.seek.set(current_idx)
			self.update_status()
			self.redraw()
			return

	def key_step(self, delta):
		global current_idx
		current_idx = min(max(0, current_idx + delta), len(items) - 1)
		load_item(current_idx)
		self.seek.set(current_idx); self.update_status(); self.redraw()

	def toggle_show_mode(self):
		global show_mode
		show_mode *= -1
		self.redraw()

	def key_save(self):
		save_annotation_and_overwrite_current()
		boxes.clear(); grey_boxes.clear()
		global current_idx
		current_idx = min(current_idx + 1, len(items) - 1)
		load_item(current_idx)
		self.seek.set(current_idx); self.update_status(); self.redraw()

	def redraw(self, temp_rect=None):
		disp = refresh_display()
		if disp is None:
			self.canvas.delete('all')
			self.canvas.create_rectangle(0,0,int(self.canvas.winfo_width()), int(self.canvas.winfo_height()), fill='black')
			return
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1
		h, w = disp.shape[:2]
		scale_w = float(c_w) / float(max(1, w))
		scale_h = float(c_h) / float(max(1, h))
		scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else 1.0
		self.composite_scale = scale
		scaled_w = max(1, int(round(w * scale)))
		scaled_h = max(1, int(round(h * scale)))
		scaled = cv2.resize(disp, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
					
		# draw crosshair on the *main video area only* 
		try:
			cx = int(round(cursor_pos[0] * disp_scale_factor))
			cy = int(round(cursor_pos[1] * disp_scale_factor))
		
			# full disp size
			h_disp, w_disp = disp.shape[:2]
		
			# main video area size in display pixels
			main_w = max(1, int(round(video_width * disp_scale_factor)))
			main_h = max(1, int(round(video_height * disp_scale_factor)))
		
			# only draw if cursor is inside main video area
			if 0 <= cx < main_w and 0 <= cy < main_h:
				cv2.line(disp, (cx, 0), (cx, main_h), (255,255,255), max(1, line_thickness))
				cv2.line(disp, (0, cy), (main_w, cy), (255,255,255), max(1, line_thickness))
		except Exception:
			pass


					
		self.tk_img = cv2_to_photoimage(scaled)
		try:
			self.canvas.config(scrollregion=(0, 0, scaled_w, scaled_h))
		except Exception:
			pass
		self.canvas.delete('all')
		self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')
		self.update_status()

	def loop(self):
		# lightweight loop: redraw tick (the redraw function already does cheap checks)
		self.redraw()
		self.root.after(30, self.loop)

# Launch
root = tk.Tk()
app = DatasetInspectorTk(root)
root.mainloop()
print("Done inspecting annotations.")
