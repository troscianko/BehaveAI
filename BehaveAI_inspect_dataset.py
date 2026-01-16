import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import configparser
import yaml
import random
import time
from ultralytics import YOLO
from collections import deque
import sys

# ---------- Determine settings INI path and project directory ----------
def choose_ini_path_from_dialog():
	root = tk.Tk()
	root.withdraw()
	ini_path = filedialog.askopenfilename(
		title="Select BehaveAI settings INI",
		filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
	)
	root.destroy()
	return ini_path

# If a command-line argument is provided, interpret as project dir or INI path.
# Otherwise prompt the user with a file picker.
if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	if os.path.isdir(arg):
		config_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		config_path = arg
else:
	config_path = choose_ini_path_from_dialog()
	if not config_path:
		# user cancelled
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)

if not os.path.exists(config_path):
	# show a GUI error then exit
	try:
		root = tk.Tk(); root.withdraw()
		messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
		root.destroy()
	except Exception:
		# fallback to console message if GUI isn't available
		print(f"Configuration file not found: {config_path}")
	sys.exit(1)

# Make the project directory the working directory so all relative paths resolve there.
project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
print(f"Using project directory: {project_dir}")
print(f"Using config file: {config_path}")

# Load configuration
config = configparser.ConfigParser()
config.optionxform = str  # preserve case if needed
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
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	motion_blocks_static = config['DEFAULT']['motion_blocks_static'].lower()
	static_blocks_motion = config['DEFAULT']['static_blocks_motion'].lower()
	save_empty_frames = config['DEFAULT']['save_empty_frames'].lower()
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")

# Basic validation (kept from original)
if len(primary_motion_classes) > len(primary_motion_colors) or len(primary_motion_classes) != len(primary_motion_hotkeys):
	raise ValueError("Primary motion classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if len(secondary_motion_classes) > len(secondary_motion_colors) or len(secondary_motion_classes) != len(secondary_motion_hotkeys):
	raise ValueError("Secondary motion classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if len(primary_static_classes) > len(primary_static_colors) or len(primary_static_classes) != len(primary_static_hotkeys):
	raise ValueError("Primary static classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if len(secondary_static_classes) > len(secondary_static_colors) or len(secondary_static_classes) != len(secondary_static_hotkeys):
	raise ValueError("Secondary static classes, colours and hotkeys must match in configuration. Ensure colours are seprated by semicolons")
if motion_blocks_static != 'true' and motion_blocks_static != 'false':
	raise ValueError("motion_blocks_static must be true or false")
if static_blocks_motion != 'true' and static_blocks_motion != 'false':
	raise ValueError("static_blocks_motion must be true or false")
if save_empty_frames != 'true' and save_empty_frames != 'false':
	raise ValueError("save_empty_frames must be true or false")

expA2 = 1 - expA
expB2 = 1 - expB

# Setup classes & hotkey dicts
primary_classes_info = list(zip(primary_hotkeys, primary_classes))
secondary_classes_info = list(zip(secondary_hotkeys, secondary_classes))
primary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(primary_classes_info)}
secondary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(secondary_classes_info)}
active_primary = 0
if len(primary_static_classes) <= 1:
	active_primary = 1
active_secondary = 0


# compute base frameWindow (exactly as original)
frameWindow = 4  # suitable for sequential
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

# create the raw buffer with base frameWindow length (NOT multiplied)
raw_buf = deque(maxlen=frameWindow)



# ---------------------------------------------------------------------------
# ---------- NEW: BUILD A LIST OF EXISTING ANNOTATED IMAGES ------------------
# ---------------------------------------------------------------------------
# Collect basenames across static & motion (train + val)
def list_images_labels_and_masks():
	items = {}  # basename -> dict with paths for static/motion image/label/mask and origin path type
	# helper
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
				# store whether this image is 'static' or 'motion' and which train/val dir
				if 'static_img' not in rec:
					rec['static_img'] = img_path
					rec['static_lbl'] = lbl_path if lbl_path and os.path.exists(lbl_path) else None
					rec['static_mask'] = mask_path if mask_path and os.path.exists(mask_path) else None
					# record which dataset subdir this came from so we save back to same
					rec['static_origin_img_dir'] = img_dir
					rec['static_origin_lbl_dir'] = lbl_dir
				else:
					# if already present prefer train over val? keep first found
					pass

	# check all 4 image directories
	add_dir(static_train_images_dir, static_train_labels_dir)
	add_dir(static_val_images_dir, static_val_labels_dir)
	add_dir(motion_train_images_dir, motion_train_labels_dir)
	add_dir(motion_val_images_dir, motion_val_labels_dir)

	# Convert dict to ordered list
	ordered = []
	for base, rec in sorted(items.items()):
		ordered.append({'basename': base, **rec})
	return ordered

items = list_images_labels_and_masks()
if not items:
	print("No annotated images found in the expected dataset directories.")
	print("Checked:", static_train_images_dir, static_val_images_dir, motion_train_images_dir, motion_val_images_dir)
	exit()

# We'll iterate over items by index
current_idx = 0

# ---------------------------------------------------------------------------
# ---------- VARIABLES REUSED FROM ORIGINAL UI ------------------------------
# ---------------------------------------------------------------------------
# state vars
drawing = False
cursor_pos = (0, 0)
ix = iy = -1
boxes = []  # displayed/edited boxes
grey_boxes = []
frame_updated = True
original_frame = None   # motion falseocolour or fallback
fr = None			   # static RGB or fallback
video_label = ""		# for display and saving
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

# zoom & layout
zoom_factor = 2
zoom_prop = 0.1
zoom_size = 250

# initially derive a default size from the first available image
first_item = items[0]
sample_img_path = first_item.get('static_img') or first_item.get('motion_img')
if not sample_img_path:
	# load whatever image we can from disk
	for it in items:
		if it.get('static_img'):
			sample_img_path = it['static_img']; break
		if it.get('motion_img'):
			sample_img_path = it['motion_img']; break
if sample_img_path:
	sample = cv2.imread(sample_img_path)
	if sample is None:
		video_height, video_width = 480, 640
	else:
		video_height, video_width = sample.shape[:2]
else:
	video_height, video_width = 480, 640

right_frame_width = int(video_height / 3)
button_height = int(font_size * 40)
ts = cv2.getTextSize("XyZ", cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)[0]
bottom_bar_height = int(ts[1]) + 6 * line_thickness

# window title
def build_window_title(basename):
	elements = ["BehaveAI Annotations (inspect):", basename, "ESC=quit BACKSPACE=clear u=undo ENTER=save SPACE=toggle view LEFT/RIGHT </> seek"]
	return ' '.join(elements)

video_name = build_window_title(items[current_idx]['basename'])
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

# reuse many of your drawing functions unchanged but adapted to use current fr/original_frame
def draw_buttons(frame):
	total_width = frame.shape[1]
	h = int(video_height * disp_scale_factor)
	y_start = h + bottom_bar_height + line_thickness
	scaled_h = frame.shape[0]
	scaled_bottom_bar_height = scaled_h - y_start
	ty = int(y_start + (scaled_bottom_bar_height - bottom_bar_height) / 2) + (line_thickness * 4)

	if hierarchical_mode:
		total_buttons = len(primary_classes) + len(secondary_classes) + 1
		button_width = total_width // total_buttons
		secondary_offset = len(primary_classes) * button_width
	else:
		total_buttons = len(primary_classes) + 1
		button_width = total_width // total_buttons
		secondary_offset = 0

	# primary
	if len(primary_classes) > 1:
		for idx in range(len(primary_classes)):
			if primary_classes[idx] != '0':
				is_active = (idx == active_primary)
				if hierarchical_mode:
					text = f"{primary_classes[idx].upper()} ({primary_classes_info[idx][0]})"
				else:
					text = f"{primary_classes[idx]} ({primary_classes_info[idx][0]})"
				color = primary_colors[idx] if is_active else (128, 128, 128)
				x1, y1 = idx * button_width, h
				x2, y2 = x1 + button_width, scaled_h
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, -line_thickness)
				border_color = (255, 255, 255) if is_active else (0, 0, 0)
				cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2 if is_active else 1)
				ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)[0]
				tx = x1 + (button_width - ts[0]) // 2
				cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), line_thickness, cv2.LINE_AA)

	# secondary
	if hierarchical_mode:
		for idx in range(len(secondary_classes)):
			is_active = (idx == active_secondary)
			text = f"{secondary_classes[idx]} ({secondary_classes_info[idx][0]})"
			color = secondary_colors[idx] if is_active else (128, 128, 128)
			x1, y1 = secondary_offset + idx * button_width, h
			x2, y2 = x1 + button_width, scaled_h
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, -line_thickness)
			border_color = (255, 255, 255) if is_active else (0, 0, 0)
			cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2 if is_active else 1)
			ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)[0]
			tx = x1 + (button_width - ts[0]) // 2
			cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), line_thickness, cv2.LINE_AA)

	# grey button
	text = "Grey (g)"
	color = (128, 128, 128)
	x1, y1 = total_width - button_width, h
	x2, y2 = total_width, scaled_h
	is_active = grey_mode
	cv2.rectangle(frame, (x1, y1), (x2, y2), color, -line_thickness)
	border_color = (255, 255, 255) if is_active else (0, 0, 0)
	cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2 if is_active else 1)
	ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)[0]
	tx = x1 + (button_width - ts[0]) // 2
	cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), line_thickness, cv2.LINE_AA)


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
				if conf != -1:
					label = label + f' {conf:.2f}'
				label = label + f" {secondary_classes[secondary_cls]}"
				if secondary_conf != -1:
					label = label + f' {secondary_conf:.2f}'
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

	# grey boxes
	for gx1, gy1, gx2, gy2 in grey_boxes:
		overlay = frame.copy()
		cv2.rectangle(overlay, (int(gx1*disp_scale_factor), int(gy1*disp_scale_factor)), (int(gx2*disp_scale_factor), int(gy2*disp_scale_factor)), (128, 128, 128), -line_thickness)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


def draw_zoom(disp, cursor_pos_in):
	if cursor_pos_in is None:
		return
	cx, cy = cursor_pos_in
	cx = int(cx/disp_scale_factor); cy = int(cy/disp_scale_factor)
	h = int(video_height); w = int(video_width)
	zoom_size_local = int(right_frame_width)
	if fr is None or original_frame is None:
		return

	half_size = zoom_size_local // 4
	x1 = max(0, cx - half_size); y1 = max(0, cy - half_size)
	x2 = min(w, cx + half_size); y2 = min(h, cy + half_size)
	crop_static = fr[y1:y2, x1:x2]
	crop_motion = original_frame[y1:y2, x1:x2]
	if crop_static.size == 0 or crop_motion.size == 0:
		return
	crop_w = x2 - x1; crop_h = y2 - y1
	rel_x = cx - x1; rel_y = cy - y1
	zoom_w = int(zoom_size_local * disp_scale_factor); zoom_h = int(zoom_size_local * disp_scale_factor)
	zoomed_static = cv2.resize(crop_static, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
	zoomed_motion = cv2.resize(crop_motion, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
	zoom_x = int(rel_x * zoom_w / crop_w) if crop_w>0 else 0
	zoom_y = int(rel_y * zoom_h / crop_h) if crop_h>0 else 0
	pos_x = int(video_width * disp_scale_factor) + line_thickness
	pos_y = 0
	disp[pos_y:pos_y+zoom_h, pos_x:pos_x+zoom_w] = zoomed_static
	disp[pos_y+zoom_h:pos_y+zoom_h+zoom_h, pos_x:pos_x+zoom_w] = zoomed_motion

	cv2.line(disp, (pos_x, pos_y + zoom_y), (pos_x + zoom_w, pos_y + zoom_y), (255, 255, 255), line_thickness)
	cv2.line(disp, (pos_x + zoom_x, pos_y), (pos_x + zoom_x, pos_y + zoom_h), (255, 255, 255), line_thickness)
	cv2.line(disp, (pos_x, pos_y + zoom_h + zoom_y), (pos_x + zoom_w, pos_y + zoom_h + zoom_y), (255, 255, 255), line_thickness)
	cv2.line(disp, (pos_x + zoom_x, pos_y + zoom_h), (pos_x + zoom_x, pos_y + zoom_h + zoom_h), (255, 255, 255), line_thickness)
	cv2.rectangle(disp, (pos_x-1, pos_y-1), (pos_x + zoom_w + 1, pos_y + zoom_h + 1 + zoom_h), (0, 0, 0), line_thickness)
	cv2.line(disp, (pos_x, pos_y + zoom_h), (pos_x + zoom_w, pos_y + zoom_h), (0, 0, 0), line_thickness)

	label = f"{primary_classes[active_primary]}"
	label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
	label_w, label_h = label_size
	cv2.rectangle(disp, (pos_x-line_thickness, pos_y + zoom_h - label_h - line_thickness*4), (pos_x + label_w + line_thickness*2, pos_y + zoom_h), (0, 0, 0), -1)
	cv2.putText(disp, label, (pos_x, pos_y + zoom_h - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, primary_colors[active_primary], line_thickness, cv2.LINE_AA)
	if hierarchical_mode and primary_classes[active_primary] != secondary_classes[active_secondary]:
		if primary_classes[active_primary] not in ignore_secondary:
			label = f"{secondary_classes[active_secondary]}"
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			cv2.rectangle(disp, (pos_x-line_thickness, pos_y + zoom_h), (pos_x + label_w + line_thickness*2, pos_y + zoom_h + label_h + line_thickness*4), (0, 0, 0), -1)
			cv2.putText(disp, label, (pos_x, pos_y + zoom_h + label_h + line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, font_size, secondary_colors[active_secondary], line_thickness, cv2.LINE_AA)

	# mini animation if raw buffer has multiple frames
	now = time.time()
	if (now - last_mouse_move) > ANIM_STILL_THRESHOLD and len(raw_buf) == raw_buf.maxlen:
		half_size = zoom_size_local // 2
		x1 = max(0, cx - half_size); y1 = max(0, cy - half_size)
		x2 = min(w, cx + half_size); y2 = min(h, cy + half_size)
		idx = int(((now - last_mouse_move) * ANIM_FPS) % raw_buf.maxlen)
		frame_to_draw = raw_buf[idx]
		small_crop = frame_to_draw[y1:y2, x1:x2]
		if small_crop.size:
			anim_w = zoom_w; anim_h = zoom_h
			anim_zoom = cv2.resize(small_crop, (anim_w, anim_h), interpolation=cv2.INTER_LINEAR)
			anim_x = pos_x; anim_y = pos_y + (zoom_h * 2)
			disp[anim_y:anim_y+anim_h, anim_x:anim_x+anim_w] = anim_zoom
			cv2.rectangle(disp, (anim_x-1, anim_y-1), (anim_x + anim_w+1, anim_y + anim_h+1), (0,0,0), line_thickness)


def refresh_display():
	global original_frame, fr, cursor_pos
	if original_frame is None or fr is None:
		return
	x, y = cursor_pos
	x = int(x / disp_scale_factor); y = int(y / disp_scale_factor)
	h, w = original_frame.shape[:2]
	if show_mode == 1:
		disp_src = original_frame.copy()
	else:
		disp_src = fr.copy()
	canvas = np.zeros((video_height + bottom_bar_height, video_width + right_frame_width + line_thickness, 3), dtype=disp_src.dtype)
	canvas[:video_height,:video_width] = disp_src
	disp = canvas
	if disp_scale_factor != 1.0:
		disp = cv2.resize(disp, None, fx=disp_scale_factor, fy=disp_scale_factor, interpolation=cv2.INTER_LINEAR)
	draw_buttons(disp)
	draw_boxes(disp)
	if drawing:
		color = (128, 128, 128) if grey_mode else primary_colors[active_primary]
		current_y = min(y, h)
		cv2.rectangle(disp, (int(ix*disp_scale_factor), int(iy*disp_scale_factor)), (int(x*disp_scale_factor), int(current_y*disp_scale_factor)), color, line_thickness)
	else:
		cv2.line(disp, (int(x*disp_scale_factor), 0), (int(x*disp_scale_factor), int(h*disp_scale_factor)), (255, 255, 255), line_thickness)
		cv2.line(disp, (0, int(y*disp_scale_factor)), (int(w*disp_scale_factor), int(y*disp_scale_factor)), (255, 255, 255), line_thickness)
	draw_zoom(disp, cursor_pos)
	cv2.setWindowTitle(video_name, build_window_title(items[current_idx]['basename']))
	cv2.imshow(video_name, disp)


# mouse callback adapted to operate on our loaded frames
def mouse_callback(event, x, y, flags, param):
	global ix, iy, drawing, active_primary, active_secondary, active_class, grey_mode
	global boxes, grey_boxes, cursor_pos, zoom_hide, last_mouse_move

	cursor_pos = (x, y)
	x = int(x / disp_scale_factor)
	y = int(y / disp_scale_factor)

	if original_frame is None:
		return

	h = int(video_height * disp_scale_factor)
	w = int(video_width * disp_scale_factor)

	if event == cv2.EVENT_LBUTTONDOWN:
		if x >= w and y >= h:
			if hierarchical_mode:
				total_buttons = len(primary_classes) + len(secondary_classes) + 1
			else:
				total_buttons = len(primary_classes) + 1
			button_width = w // total_buttons
			button_idx = x // button_width

			if button_idx == total_buttons - 1:
				# grey toggle
				grey_mode = not grey_mode
			elif button_idx < len(primary_classes):
				active_primary = button_idx
				grey_mode = False
			elif hierarchical_mode:
				active_secondary = button_idx - len(primary_classes)
				grey_mode = False
			refresh_display()
		else:
			drawing = True
			ix, iy = x, y
			refresh_display()

	elif event == cv2.EVENT_MOUSEMOVE:
		zoom_hide = 0
		if drawing:
			last_mouse_move = time.time()
			refresh_display()

	elif event == cv2.EVENT_LBUTTONUP and drawing:
		drawing = False
		current_y = min(y, h)
		if abs(x - ix) > 5 and abs(current_y - iy) > 5:
			if grey_mode:
				grey_boxes.append((min(ix, x), min(iy, current_y), max(ix, x), max(iy, current_y)))
			else:
				x1, y1 = min(ix, x), min(iy, current_y)
				x2, y2 = max(ix, x), max(iy, current_y)
				if hierarchical_mode:
					boxes.append((x1, y1, x2, y2, active_primary, active_secondary, -1, -1))
				else:
					boxes.append((x1, y1, x2, y2, active_primary, -1))
		refresh_display()

	elif event == cv2.EVENT_RBUTTONUP:
		if y >= h:
			return
		deleted = False
		for i in range(len(boxes)-1, -1, -1):
			box = boxes[i]
			x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
			if x1 <= x <= x2 and y1 <= y <= y2:
				del boxes[i]; deleted = True; break
		if not deleted:
			for i in range(len(grey_boxes)-1, -1, -1):
				gx1, gy1, gx2, gy2 = grey_boxes[i]
				if gx1 <= x <= gx2 and gy1 <= y <= gy2:
					del grey_boxes[i]; break
		refresh_display()

cv2.setMouseCallback(video_name, mouse_callback)

# raw buffer for mini animation (per-item)
raw_buf = deque(maxlen=4)

# ---------------------------------------------------------------------------
# ---------- HELPERS: parse labels, masks, load images & optionally video ---
# ---------------------------------------------------------------------------
def norm_to_pixels(xc, yc, bw, bh, w, h):
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p/2); y1 = int(cy - bh_p/2)
	x2 = int(cx + bw_p/2); y2 = int(cy + bh_p/2)
	# clip
	x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1)); x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
	return x1, y1, x2, y2

def load_labels_and_masks_for_item(item):
	"""Load boxes and masks for the given item dict into global boxes and grey_boxes.
	   This reconstructs global primary class indices (static classes keep indices as saved; motion label classes are offset).
	"""
	global boxes, grey_boxes
	boxes = []
	grey_boxes = []
	base = item['basename']
	# load static labels (if exist)
	if item.get('static_lbl') and os.path.exists(item['static_lbl']):
		with open(item['static_lbl'], 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) < 5:
					continue
				cls = int(parts[0])
				xc, yc, bw, bh = parts[1:5]
				h, w = fr.shape[:2]
				x1,y1,x2,y2 = norm_to_pixels(xc, yc, bw, bh, w, h)
				# static labels were saved as primary_cls directly
				if hierarchical_mode:
					boxes.append((x1,y1,x2,y2, cls, 0, -1, -1))
				else:
					boxes.append((x1,y1,x2,y2, cls, -1))
	# load motion labels (if exist)
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
				# when saved, motion label class = primary_cls - len(primary_static_classes)
				global_primary_cls = cls + len(primary_static_classes)
				if hierarchical_mode:
					boxes.append((x1,y1,x2,y2, global_primary_cls, 0, -1, -1))
				else:
					boxes.append((x1,y1,x2,y2, global_primary_cls, -1))
	# load mask file (prefer static mask if present, else motion)
	mask_path = item.get('static_mask') or item.get('motion_mask')
	if mask_path and os.path.exists(mask_path):
		with open(mask_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) >= 4:
					gx1, gy1, gx2, gy2 = map(int, parts[:4])
					grey_boxes.append((gx1, gy1, gx2, gy2))

# ~ def find_video_for_item(item):
	# ~ """Try to locate a video in ./clips that matches the saved video_label extracted from the basename.
	   # ~ The saved naming convention used earlier was: {video_label}_{frame_number}.jpg
	   # ~ We'll split basename at first underscore to try to infer video_label and frame_number.
	# ~ """
	# ~ clips_dir = os.path.join(os.getcwd(), "clips")
	# ~ if not os.path.isdir(clips_dir):
		# ~ return None, None
	# ~ base = item['basename']
	# ~ if '_' not in base:
		# ~ return None, None
	# ~ parts = base.split('_', 1)
	# ~ video_label_guess = parts[0]
	# ~ frame_number_guess = None
	# ~ # try to parse trailing number from second part if it starts with an int
	# ~ try:
		# ~ tail = parts[1].split('_')[0]
		# ~ frame_number_guess = int(tail)
	# ~ except Exception:
		# ~ frame_number_guess = None
	# ~ # search for file starting with video_label_guess
	# ~ for fname in os.listdir(clips_dir):
		# ~ if fname.lower().startswith(video_label_guess.lower()) and fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
			# ~ return os.path.join(clips_dir, fname), frame_number_guess
	# ~ return None, None

def find_video_for_item(item):
	"""
	Improved video lookup for an annotation item.

	Naming convention expected for annotation frames:
		<video_filename_without_extension>_<frameNumber>.jpg

	This function:
	  - Splits on the LAST underscore to separate video label and frame number.
	  - Matches clips in ./clips by exact stem (case-insensitive).
	  - Falls back to startswith matching only if no exact stem match exists.
	  - Returns (video_path_or_None, frame_number_or_None).
	"""
	# ~ clips_dir = os.path.join(os.getcwd(), "clips")
	if not os.path.isdir(clips_dir):
		return None, None

	base = item['basename']
	# Must split on the last underscore to allow underscores inside video names
	if '_' not in base:
		return None, None

	video_label_guess, tail = base.rsplit('_', 1)

	# Try to parse the tail as an integer frame index
	frame_number_guess = None
	try:
		frame_number_guess = int(tail)
	except Exception:
		frame_number_guess = None

	# First try exact stem match (most robust)
	for fname in os.listdir(clips_dir):
		if not fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
			continue
		stem = os.path.splitext(fname)[0]
		if stem.lower() == video_label_guess.lower():
			return os.path.join(clips_dir, fname), frame_number_guess

	# Fallback: startswith match (only if no exact match found)
	for fname in os.listdir(clips_dir):
		if not fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
			continue
		stem = os.path.splitext(fname)[0]
		if stem.lower().startswith(video_label_guess.lower()):
			return os.path.join(clips_dir, fname), frame_number_guess

	# no match
	return None, None



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
	load_labels_and_masks_for_item(item)

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



	# try to find and load video preview frames
	# try to find and load video preview frames
	# try to find and load video preview frames (replicating original sampling behaviour)
	video_path_found, guessed_frame = find_video_for_item(item)
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





# populate motion_img keys in items: check both motion dirs
for it in items:
	base = it['basename']
	p1 = os.path.join(motion_train_images_dir, base + '.jpg')
	p2 = os.path.join(motion_val_images_dir, base + '.jpg')
	if os.path.exists(p1):
		it['motion_img'] = p1; it['motion_lbl'] = os.path.join(motion_train_labels_dir, base + '.txt'); it['motion_mask'] = os.path.join(motion_train_labels_dir.replace('labels','masks'), base + '.mask.txt')
	elif os.path.exists(p2):
		it['motion_img'] = p2; it['motion_lbl'] = os.path.join(motion_val_labels_dir, base + '.txt'); it['motion_mask'] = os.path.join(motion_val_labels_dir.replace('labels','masks'), base + '.mask.txt')
	# static already filled earlier but ensure label/mask presence tracking
	if it.get('static_img'):
		base_lbl = os.path.join(it['static_origin_lbl_dir'], it['basename'] + '.txt')
		it['static_lbl'] = base_lbl if os.path.exists(base_lbl) else None
		base_mask = os.path.join(it['static_origin_lbl_dir'].replace('labels','masks'), it['basename'] + '.mask.txt')
		it['static_mask'] = base_mask if os.path.exists(base_mask) else None

# load first item
load_item(current_idx)
video_path = None

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

# ---------------------------------------------------------------------------
# ---------- MAIN UI LOOP --------------------------------------------------
# ---------------------------------------------------------------------------
# create a simple "Item" trackbar to jump between items
def on_trackbar(x):
	global current_idx, frame_updated
	current_idx = x
	load_item(current_idx)
	frame_updated = True
	cv2.setWindowTitle(video_name, build_window_title(items[current_idx]['basename']))

cv2.createTrackbar('Item', video_name, 0, max(0, len(items)-1), on_trackbar)

# set initial trackbar pos
cv2.setTrackbarPos('Item', video_name, current_idx)

print("Starting inspection of annotation dataset. Items found:", len(items))
print("Use LEFT/RIGHT to step, ENTER to save & advance, BACKSPACE to clear, u to undo, right-click to delete boxes/greys.")

while True:
	now = time.time()
	need_redraw = False

	if frame_updated:
		frame_updated = False
		# boxes/grey_boxes loaded in load_item
		refresh_display()
		need_redraw = True
		last_anim_draw = time.time()
	else:
		# animation tick: if video preview buffer is present, update
		if (now - last_mouse_move) > ANIM_STILL_THRESHOLD and (now - last_anim_draw) >= ANIM_DT and len(raw_buf) == raw_buf.maxlen:
			last_anim_draw = now
			need_redraw = True

	if cv2.getWindowProperty(video_name, cv2.WND_PROP_VISIBLE) < 1:
		break

	key = cv2.waitKey(20) & 0xFF
	if key == 27:  # ESC
		break
	if key == 8:  # BACKSPACE clear
		boxes.clear(); grey_boxes.clear(); refresh_display()
	elif key == 13:  # ENTER -> save & advance to next
		save_annotation_and_overwrite_current()
		grey_boxes.clear(); boxes.clear()
		current_idx = min(current_idx + 1, len(items) - 1)
		cv2.setTrackbarPos('Item', video_name, current_idx)
		load_item(current_idx)
		frame_updated = True
	elif key == ord('u'):
		if grey_mode:
			if grey_boxes: grey_boxes.pop()
		elif boxes:
			boxes.pop()
		refresh_display()
	elif key == ord('g'):
		grey_mode = True
	elif key in primary_class_dict and key in secondary_class_dict:
		if key != ord('0'):
			active_primary = primary_class_dict[key]; active_secondary = secondary_class_dict[key]; grey_mode = False
			refresh_display()
	elif key in primary_class_dict:
		if key != ord('0'):
			active_primary = primary_class_dict[key]
			if active_primary < len(primary_static_classes):
				show_mode = -1
			else:
				show_mode = 1
			grey_mode = False
			refresh_display()
	elif key in secondary_class_dict:
		if key != ord('0'):
			active_secondary = secondary_class_dict[key]
			if active_secondary < len(secondary_static_classes):
				show_mode = -1
			else:
				show_mode = 1
			grey_mode = False
			refresh_display()
	elif key == 83:  # right arrow
		current_idx = min(current_idx + 1, len(items) - 1)
		cv2.setTrackbarPos('Item', video_name, current_idx)
		load_item(current_idx)
		frame_updated = True
	elif key == 81:  # left arrow
		current_idx = max(current_idx - 1, 0)
		cv2.setTrackbarPos('Item', video_name, current_idx)
		load_item(current_idx)
		frame_updated = True
	elif key == 46:  # > (.)
		current_idx = min(current_idx + 10, len(items) - 1)
		cv2.setTrackbarPos('Item', video_name, current_idx)
		load_item(current_idx)
		frame_updated = True
	elif key == 44:  # < (,)
		current_idx = max(current_idx - 10, 0)
		cv2.setTrackbarPos('Item', video_name, current_idx)
		load_item(current_idx)
		frame_updated = True
	elif key == 32:  # SPACE toggle view
		show_mode *= -1; refresh_display()
	elif key == 35:  # HASH to toggle auto-annotate (if you want)
		auto_ann_switch *= -1
		if auto_ann_switch == 1:
			# call your auto_annotate() if present (not invoked in this inspector by default)
			try:
				auto_annotate()
			except Exception:
				pass
		refresh_display()
	elif key == 45:  # -
		disp_scale_factor *= 0.8; refresh_display()
	elif key == 61:  # =
		disp_scale_factor *= 1.25; refresh_display()

	if need_redraw:
		refresh_display()

cv2.destroyAllWindows()
print("Done inspecting annotations.")
