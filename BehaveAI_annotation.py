import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import configparser
import yaml
import random
import time
from ultralytics import YOLO
from collections import deque
import platform
import config_watcher



# Load configuration
config = configparser.ConfigParser()
config_path = 'BehaveAI_settings.ini'
if not os.path.exists(config_path):
	raise FileNotFoundError(f"Configuration file not found: {config_path}")
config.read(config_path)

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

# Validate configuration

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

# Setup classes based on mode
primary_classes_info = list(zip(primary_hotkeys, primary_classes))
secondary_classes_info = list(zip(secondary_hotkeys, secondary_classes))
primary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(primary_classes_info)}
secondary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(secondary_classes_info)}
active_primary = 0
if len(primary_static_classes) <= 1:
	active_primary = 1
active_secondary = 0



#-------Check whether models exist-------------

motion_model_count = 0

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
				motion_model_count += 1
				
		# ~ print(f"secondary_motion_models {secondary_motion_models}")

#-------CHECK PRIMARY MODEL EXISTS----------
if primary_static_classes[0] != '0':
	if not os.path.exists(primary_static_model_path):
		print('Primary static model not found')
		model_static = None
	else:
		print('Primary static model found')
		model_static = YOLO(primary_static_model_path)

if primary_motion_classes[0] != '0':
	if not os.path.exists(primary_motion_model_path):
		print('Primary motion model not found')
		model_motion = None
	else:
		print('Primary motion model found')
		model_motion = YOLO(primary_motion_model_path)
		motion_model_count += 1
	

if motion_model_count > 0:
	# check whether settings have been changed, and motion annotation library needs rebuilding 
	settings_changed = config_watcher.check_settings_changed(current_config_path='BehaveAI_settings.ini', saved_config_path=None, model_dirs=['model_primary_motion'])
	
	if settings_changed:
		root = tk.Tk()
		root.withdraw()
		msg = (
			"Motion-processing settings have changed since training motion model(s).\n\n"
			"Do you want to rebuild the annotation dataset now?"
		)
		user_wants_regen = messagebox.askyesno("Rebuild annotations?", msg)
		root.destroy()
	
		if user_wants_regen:
			print("User requested regeneration of annotations...")
			rc = config_watcher.run_regeneration(regen_script='Regenerate_annotations.py', regen_args=None)
			if rc == 0:
				print("Regeneration script completed successfully.")
			else:
				print(f"Warning: regeneration script returned code {rc}.")


# File dialog for video selection
clips_dir = os.path.join(os.getcwd(), "clips")
initial_dir = clips_dir if os.path.isdir(clips_dir) else os.getcwd()

root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select video file", initialdir=initial_dir)
root.destroy()
if not video_path:
	print("No video selected. Exiting.")
	exit()


# Create directories
for d in [motion_train_images_dir, motion_val_images_dir, motion_train_labels_dir, motion_val_labels_dir]:
	os.makedirs(d, exist_ok=True)

for d in [static_train_images_dir, static_val_images_dir, static_train_labels_dir, static_val_labels_dir]:
	os.makedirs(d, exist_ok=True)

# Write YAML config

# Static dataset YAML
static_yaml_output = 'static_annotations.yaml'
static_yaml_dict = {
	'train': os.path.abspath(static_train_images_dir),
	'val': os.path.abspath(static_val_images_dir),
	'nc': len(primary_static_classes),
	'names': primary_static_classes
}
with open(static_yaml_output, 'w') as yf:
	yaml.dump(static_yaml_dict, yf)
print(f"Written static YOLO dataset config to {static_yaml_output}")

# Motion dataset YAML
motion_yaml_output = 'motion_annotations.yaml'
motion_yaml_dict = {
	'train': os.path.abspath(motion_train_images_dir),
	'val': os.path.abspath(motion_val_images_dir),
	'nc': len(primary_motion_classes),
	'names': primary_motion_classes
}
with open(motion_yaml_output, 'w') as yf:
	yaml.dump(motion_yaml_dict, yf)
print(f"Written motion YOLO dataset config to {motion_yaml_output}")

capture = cv2.VideoCapture(video_path)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
video_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
right_frame_width = int(video_height/3)
# ~ right_frame_width = 120



# the frame window needs to be larger the further back in time smoothing covers
frameWindow = 4 # suitable for sequential
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
	# beyond this the numbers get very large - 0.95 would be ~90
	
	raw_buf = deque(maxlen=frameWindow) # length of animation preview
else:
	raw_buf = deque(maxlen=4) # in sequential mode restrict to 4 frames
	

# ~ print(f"expA {expA}, expB {expB} frameWindow {frameWindow}")

frameWindow = frameWindow * (frame_skip +1)

# initial last-frame to display = earliest valid last-frame (frameWindow-1), clamped to [0, total_frames-1]
frame_number = min(max(frameWindow - 1, 0), total_frames - 1)
frame_updated = True

# State variables
drawing = False
cursor_pos = (0, 0)
ix = iy = -1
boxes = []  # For all boxes: (x1, y1, x2, y2, class_idx) or (x1, y1, x2, y2, static_cls, motion_cls)
grey_boxes = []
original_frame = None
video_label = os.path.splitext(os.path.basename(video_path))[0]
button_height = int(font_size * 40)
# ~ bottom_bar_height = button_height 
ts  = cv2.getTextSize("XyZ", cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)[0]
bottom_bar_height = int(ts[1]) + 6 * line_thickness
grey_mode = False
annot_count = 1
auto_ann_switch = 1
show_mode = 1
zoom_hide = 0
disp_scale_factor = 1.0


last_mouse_move = 0.0 # timestamp of last mouse move
ANIM_STILL_THRESHOLD = 0.5   # seconds to wait before animating
ANIM_FPS = 8				 # frames per second for the mini‐animation
last_anim_draw = 0.0
ANIM_DT = 1.0 / ANIM_FPS


# zoom box variables
zoom_factor = 2
zoom_prop = 0.1 # proporiton of screen width that the zoomed boxes should be - shouldn't be >~0.3
zoom_size = 250  # Size of the zoomed region in pixels (before zoom)
cursor_pos = (0, 0)


print("Annotating video: " + video_label)
print(f"Mode: {'Hierarchical' if hierarchical_mode else 'Standard'}")

# Window title
elements = ["BehaveAI Video:", video_label, "ESC=quit BACKSPACE=clear u=undo ENTER=save SPACE=flip mode LEFT/RIGHT </> seek",]

video_name = ' '.join(elements)

cv2.namedWindow(video_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)


# UI functions
def draw_buttons(frame):
	total_width = frame.shape[1]
	# ~ h = frame.shape[0]
	h = int(video_height * disp_scale_factor)
	y_start = h + bottom_bar_height + line_thickness # Start from bottom
	scaled_h = frame.shape[0]
	scaled_bottom_bar_height = scaled_h - y_start
	ty = int(y_start + (scaled_bottom_bar_height - bottom_bar_height) / 2) + (line_thickness * 4)
	
	# Calculate button widths
	ignore_first = 0
	if hierarchical_mode:
		total_buttons = len(primary_classes) + len(secondary_classes) + 1
		button_width = total_width // total_buttons
		secondary_offset = len(primary_classes) * button_width
	else:
		total_buttons = len(primary_classes) + 1
		button_width = total_width // total_buttons
		secondary_offset = 0
	
	# Draw primary classes
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
				# ~ ty = h + (bottom_bar_height * disp_scale_factor + ts[1]) // 2
				cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), line_thickness, cv2.LINE_AA)

	# Draw secondary classes
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
			# ~ ty = h + (bottom_bar_height * disp_scale_factor + ts[1]) // 2
			cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), line_thickness, cv2.LINE_AA)
	
	# Draw grey button
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
	# ~ ty = y_start + int((bottom_bar_height * disp_scale_factor + ts[1]) // 2)
	cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), line_thickness, cv2.LINE_AA)

def draw_boxes(frame):
	# Draw all boxes
	for box in boxes:

			
		if hierarchical_mode:
			x1, y1, x2, y2, primary_cls, secondary_cls, conf, secondary_conf = box
			x1 = int(x1 * disp_scale_factor)
			y1 = int(y1 * disp_scale_factor)
			x2 = int(x2 * disp_scale_factor)
			y2 = int(y2 * disp_scale_factor)
			if primary_classes[primary_cls] in ignore_secondary:
				label = f"{primary_classes[primary_cls].upper()}"
				if conf != -1: # add confidence if auto-annotated
					label = label + f' {conf:.2f}'
				label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
				label_w, label_h = label_size
				cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
				cv2.rectangle(frame, (x1, y1), (x2, y2), primary_colors[primary_cls], line_thickness)
				cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, 
							font_size, primary_colors[primary_cls], line_thickness, cv2.LINE_AA)
			else:
				# Draw outer static box (slightly larger)
				outer_thickness = line_thickness + 2
				cv2.rectangle(frame, (x1-outer_thickness, y1-outer_thickness), 
							 (x2+outer_thickness, y2+outer_thickness), 
							 primary_colors[primary_cls], outer_thickness)
				label = f"{primary_classes[primary_cls].upper()}"
				if conf != -1: # add confidence if auto-annotated
					label = label + f' {conf:.2f}'
				label = label + f" {secondary_classes[secondary_cls]}"
				if secondary_conf != -1: # add confidence if auto-annotated
					label = label + f' {secondary_conf:.2f}'	
				label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
				label_w, label_h = label_size
				cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
				cv2.rectangle(frame, (x1, y1), (x2, y2), secondary_colors[secondary_cls], line_thickness)
				cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, 
							font_size, secondary_colors[secondary_cls], line_thickness, cv2.LINE_AA)
		else:
			x1, y1, x2, y2, primary_cls, conf = box
			x1 = int(x1 * disp_scale_factor)
			y1 = int(y1 * disp_scale_factor)
			x2 = int(x2 * disp_scale_factor)
			y2 = int(y2 * disp_scale_factor)
			label = f"{primary_classes[primary_cls]}"
			if conf != -1: # add confidence if auto-annotated
				label = label + f' {conf:.2f}'
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
			cv2.rectangle(frame, (x1, y1), (x2, y2), primary_colors[primary_cls], line_thickness)
			cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, 
						font_size, primary_colors[primary_cls], line_thickness, cv2.LINE_AA)

	# Draw grey boxes
	for gx1, gy1, gx2, gy2 in grey_boxes:
		overlay = frame.copy()
		cv2.rectangle(overlay, (int(gx1*disp_scale_factor), int(gy1*disp_scale_factor)), (int(gx2*disp_scale_factor), int(gy2*disp_scale_factor)), (128, 128, 128), -line_thickness)
		# ~ cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def select_frame(x):
	global frame_number, frame_updated
	frame_number = x
	frame_updated = True

def refresh_display():
	if original_frame is not None:
		
		x, y = cursor_pos
	
		x = int(x / disp_scale_factor)
		y = int(y / disp_scale_factor)
		
		if original_frame is None:
			return
		
		h, w = original_frame.shape[:2]
		

		
		# Show temporary box while drawing (clipped to above bottom bar)
		if show_mode == 1:
			disp = original_frame.copy()
		else:
			disp = fr.copy()
			
		canvas = np.zeros((video_height + bottom_bar_height, video_width + right_frame_width + line_thickness, 3), dtype=disp.dtype)
		canvas[:video_height,:video_width] = disp
		disp = canvas
		
		
		# Apply zoom to disp
		if disp_scale_factor != 1.0:
			disp = cv2.resize(disp, None, fx=disp_scale_factor, fy=disp_scale_factor, interpolation=cv2.INTER_LINEAR)
							
		draw_buttons(disp)
		draw_boxes(disp)
		
		if drawing: # draw annotation box
			color = (128, 128, 128) if grey_mode else primary_colors[active_primary]
			# Clip drawing to above bottom bar
			current_y = min(y, h)
			cv2.rectangle(disp, (int(ix*disp_scale_factor), int(iy*disp_scale_factor)), (int(x*disp_scale_factor), int(current_y*disp_scale_factor)), color, line_thickness)
		else: # draw crosshairs
			cv2.line(disp, (int(x*disp_scale_factor), 0), (int(x*disp_scale_factor), int(h*disp_scale_factor)), (255, 255, 255), line_thickness)
			cv2.line(disp, (0, int(y*disp_scale_factor)), (int(w*disp_scale_factor), int(y*disp_scale_factor)), (255, 255, 255), line_thickness)
			
		draw_zoom(disp, cursor_pos)  # Add zoom view to temporary drawing
		
		# draw current (last) frame number top-left so label matches trackbar
		# ~ label = f"Frame (last): {frame_number}"
		# ~ cv2.putText(disp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), line_thickness, cv2.LINE_AA)


		cv2.imshow(video_name, disp)


def draw_zoom(disp, cursor_pos):
	if cursor_pos is None:
		return
	
	cx, cy = cursor_pos
	cx = int(cx/disp_scale_factor)
	cy = int(cy/disp_scale_factor)
	h = int(video_height)
	w = int(video_width)
	zoom_size = int(right_frame_width)
	
	# Skip if cursor is in bottom bar
	# ~ if cy >= bottom_bar_top:
		# ~ return
	
	# Calculate crop area around cursor
	half_size = zoom_size // 4
	x1 = max(0, cx - half_size)
	y1 = max(0, cy - half_size)
	x2 = min(w, cx + half_size)
	y2 = min(h, cy + half_size)
	
	# Crop and enlarge
	if fr is None:
		return
	
	crop_static = fr[y1:y2, x1:x2]
	crop_motion = original_frame[y1:y2, x1:x2]
	
	if crop_static.size == 0 or crop_motion.size == 0:
		return
		
	# Get actual crop dimensions (may be smaller near edges)
	crop_w = x2 - x1
	crop_h = y2 - y1
	
	# Calculate cursor's relative position within crop
	rel_x = cx - x1
	rel_y = cy - y1
	
	# Resize both crops to the same zoomed size
	zoom_w = int(zoom_size * disp_scale_factor)
	zoom_h = int(zoom_size * disp_scale_factor)
	zoomed_static = cv2.resize(crop_static, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
	zoomed_motion = cv2.resize(crop_motion, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
	
	# Calculate crosshair position in zoomed image
	zoom_x = int(rel_x * zoom_w / crop_w)
	zoom_y = int(rel_y * zoom_h / crop_h)
	
	pos_x = int(video_width * disp_scale_factor) + line_thickness
	pos_y = 0
	
	
	# Place zoomed images
	disp[pos_y:pos_y+zoom_h, pos_x:pos_x+zoom_w] = zoomed_static
	disp[pos_y+zoom_h:pos_y+zoom_h+zoom_h, pos_x:pos_x+zoom_w] = zoomed_motion
	
	# Draw crosshairs in static zoomed box
	cv2.line(disp, 
			 (pos_x, pos_y + zoom_y), 
			 (pos_x + zoom_w, pos_y + zoom_y), 
			 (255, 255, 255), line_thickness)
	cv2.line(disp, 
			 (pos_x + zoom_x, pos_y), 
			 (pos_x + zoom_x, pos_y + zoom_h), 
			 (255, 255, 255), line_thickness)
	
	# Draw crosshairs in motion zoomed box
	cv2.line(disp, 
			 (pos_x, pos_y + zoom_h + zoom_y), 
			 (pos_x + zoom_w, pos_y + zoom_h + zoom_y), 
			 (255, 255, 255), line_thickness)
	cv2.line(disp, 
			 (pos_x + zoom_x, pos_y + zoom_h), 
			 (pos_x + zoom_x, pos_y + zoom_h + zoom_h), 
			 (255, 255, 255), line_thickness)
	
	# Draw border around both zoomed regions
	cv2.rectangle(disp, 
				 (pos_x-1, pos_y-1), 
				 (pos_x + zoom_w + 1, pos_y + zoom_h + 1 + zoom_h), 
				 (0, 0, 0), line_thickness)
	# line between zoom boxes
	cv2.line(disp, 
			 (pos_x, pos_y + zoom_h), 
			 (pos_x + zoom_w, pos_y + zoom_h), 
			 (0, 0, 0), line_thickness)


	label = f"{primary_classes[active_primary]}"
	label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
	label_w, label_h = label_size
	cv2.rectangle(disp, (pos_x-line_thickness, pos_y + zoom_h - label_h - line_thickness*4), (pos_x + label_w + line_thickness*2, pos_y + zoom_h), (0, 0, 0), -1)
	cv2.putText(disp, label, (pos_x, pos_y + zoom_h - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
							font_size, primary_colors[active_primary], line_thickness, cv2.LINE_AA)
	if hierarchical_mode and primary_classes[active_primary] != secondary_classes[active_secondary]:
		if primary_classes[active_primary] not in ignore_secondary:
			label = f"{secondary_classes[active_secondary]}"
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			cv2.rectangle(disp, (pos_x-line_thickness, pos_y + zoom_h), (pos_x + label_w + line_thickness*2, pos_y + zoom_h + label_h + line_thickness*4), (0, 0, 0), -1)
			cv2.putText(disp, label, (pos_x, pos_y + zoom_h + label_h + line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
									font_size, secondary_colors[active_secondary], line_thickness, cv2.LINE_AA)		

	# ─── Mini‐animation box ───────────────────────────────────────────
	now = time.time()
	if (now - last_mouse_move) > ANIM_STILL_THRESHOLD and len(raw_buf) == raw_buf.maxlen:
			# Calculate crop area around cursor
		half_size = zoom_size // 2 ## don't zoom animation
		x1 = max(0, cx - half_size)
		y1 = max(0, cy - half_size)
		x2 = min(w, cx + half_size)
		y2 = min(h, cy + half_size)
		# pick which buffer index to draw based on elapsed
		idx = int(((now - last_mouse_move) * ANIM_FPS) % raw_buf.maxlen)
		frame_to_draw = raw_buf[idx]
		# crop exactly same way
		small_crop = frame_to_draw[y1:y2, x1:x2]
		if small_crop.size:
			anim_w = zoom_w   # reuse same dims
			anim_h = zoom_h
			anim_zoom = cv2.resize(small_crop, (anim_w, anim_h), interpolation=cv2.INTER_LINEAR)
			# position: beneath the two existing zoom boxes
			anim_x = pos_x
			anim_y = pos_y + (zoom_h * 2)
			# draw it
			disp[anim_y:anim_y+anim_h, anim_x:anim_x+anim_w] = anim_zoom
			# border
			cv2.rectangle(disp,
						  (anim_x-1, anim_y-1),
						  (anim_x + anim_w+1, anim_y + anim_h+1),
						  (0,0,0), line_thickness)


def mouse_callback(event, x, y, flags, param):
	global ix, iy, drawing, active_primary, active_secondary, active_class, grey_mode
	global boxes, grey_boxes, cursor_pos, zoom_hide

	cursor_pos = (x, y)
	
	x = int(x / disp_scale_factor)
	y = int(y / disp_scale_factor)
	
	if original_frame is None:
		return
	
	h = int(video_height * disp_scale_factor)
	w = int(video_width * disp_scale_factor)
	
	if event == cv2.EVENT_LBUTTONDOWN:
		if x >= w and y >= h:  # click outside video region
			# Handle button clicks
			if hierarchical_mode:
				total_buttons = len(primary_classes) + len(secondary_classes) + 1
			else:
				total_buttons = len(primary_classes) + 1
			button_width = w // total_buttons
			button_idx = x // button_width
			
			# Grey button
			if button_idx == total_buttons - 1:
				grey_mode = not grey_mode
			# Primary class buttons
			elif button_idx < len(primary_classes):
				active_primary = button_idx
				grey_mode = False
			# Secondary class buttons
			elif hierarchical_mode:
				active_secondary = button_idx - len(primary_classes)
				grey_mode = False
			refresh_display()
		else:
			# Start drawing a box (ensure above bottom bar)
			drawing = True
			ix, iy = x, y
			refresh_display()
	
	elif event == cv2.EVENT_MOUSEMOVE:
		zoom_hide = 0
		if drawing:
			last_mouse_move = time.time() # record movement timestamp
			refresh_display()

	elif event == cv2.EVENT_LBUTTONUP and drawing:
		drawing = False
		# Clip box coordinates to above bottom bar
		current_y = min(y, h)
		
		if abs(x - ix) > 5 and abs(current_y - iy) > 5:
			if grey_mode:
				# Add grey area (clipped to above bottom bar)
				grey_boxes.append((min(ix, x), min(iy, current_y), max(ix, x), max(iy, current_y)))
			else:
				# Add a new box (clipped to above bottom bar)
				x1, y1 = min(ix, x), min(iy, current_y)
				x2, y2 = max(ix, x), max(iy, current_y)
				
				if hierarchical_mode:
					boxes.append((x1, y1, x2, y2, active_primary, active_secondary,  -1, -1))
				else:
					boxes.append((x1, y1, x2, y2, active_primary, -1))
		
		refresh_display()

	elif event == cv2.EVENT_RBUTTONUP:
		# Check if right-click is in bottom bar (ignore if so)
		if y >= h:
			return
			
		# Check boxes (reverse order to delete topmost first)
		deleted = False
		for i in range(len(boxes)-1, -1, -1):
			box = boxes[i]
			# Extract coordinates (first 4 elements)
			x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
			if x1 <= x <= x2 and y1 <= y <= y2:
				del boxes[i]
				deleted = True
				break
				
		# Check grey boxes if no regular box was deleted
		if not deleted:
			for i in range(len(grey_boxes)-1, -1, -1):
				gx1, gy1, gx2, gy2 = grey_boxes[i]
				if gx1 <= x <= gx2 and gy1 <= y <= gy2:
					del grey_boxes[i]
					break
					
		refresh_display()



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


	# ~ # Save grey box coordinates
	# ~ mask_content = ""
	# ~ for gx1, gy1, gx2, gy2 in grey_boxes:
		# ~ mask_content += f"{gx1} {gy1} {gx2} {gy2}\n"
	
	# ~ # Save to both static and motion directories
	# ~ static_mask_path = os.path.join(static_target_lbl_dir, f"{base_filename}.mask.txt")
	# ~ motion_mask_path = os.path.join(motion_target_lbl_dir, f"{base_filename}.mask.txt")
	
	# ~ for path in [static_mask_path, motion_mask_path]:
		# ~ with open(path, 'w') as f:
			# ~ f.write(mask_content)

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

def auto_annotate():
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




# Setup UI and loop
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
# ~ cv2.createTrackbar('Frame', video_name, 0, total_frames - frameWindow + frame_skip -1, select_frame)
cv2.createTrackbar('Frame', video_name, 0, max(0, total_frames - 1), select_frame)
cv2.setTrackbarPos('Frame', video_name, frame_number)
cv2.setMouseCallback(video_name, mouse_callback)



while True:
	
	now = time.time()

	# Load new frame
	if frame_updated:
		frame_updated = False
		boxes.clear()
		grey_boxes.clear()
	
		# Treat frame_number as the *last* frame of the window
		last_frame = frame_number
		start_frame = last_frame - frameWindow + 1  # start so that last read frame == last_frame
	
		# If the window extends before the start of video => show a blank area
		if start_frame < 0:
			# blank frames for display when not enough preceding frames
			original_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
			fr = original_frame.copy()
			motion_image = original_frame.copy()
			raw_buf.clear()
			need_redraw = True
			last_anim_draw = time.time()
			# Do not auto-annotate when there aren't enough frames
			# (user wanted blank display for numbers below window size)
			# Leave boxes empty and continue to redraw the blank.
		else:
			# Position capture at start_frame (so last read corresponds to last_frame)
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
						fr = cv2.resize(fr, (0, 0), fx=scale_factor, fy=scale_factor)
					raw_buf.append(fr.copy())
					gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
	
					if i == 0:
						prev_frames = [gray.copy()] * 3
						# first processed frame in window; continue to next raw read
						frame_count += 1
						if frame_count > frame_skip:
							frame_count = 0
						continue
	
					diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
	
					if strategy == 'exponential':
						prev_frames[0] = gray
						prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, expA2, 0)
						prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, expB2, 0)
					elif strategy == 'sequential':
						prev_frames[2] = prev_frames[1]
						prev_frames[1] = prev_frames[0]
						prev_frames[0] = gray
	
				frame_count += 1
				if frame_count > frame_skip:
					frame_count = 0
	
			# Build motion image from diffs/gray (same logic as before)
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
	
				if auto_ann_switch == 1:
					auto_annotate()
					zoom_hide = 1
	
			need_redraw = True
			last_anim_draw = time.time()
	else:
		need_redraw = False
		
	# animation tick
	if (now - last_mouse_move) > ANIM_STILL_THRESHOLD and (now - last_anim_draw) >= ANIM_DT:
		last_anim_draw = now
		# redraw in place (draw_zoom will pick the correct buffer index)
		need_redraw = True

	# listen for keys
	if cv2.getWindowProperty(video_name, cv2.WND_PROP_VISIBLE) < 1:
		break
		
	key = cv2.waitKey(5) & 0xFF
	if key == 27:  # ESC
		break
	if key == 8:  # BACKSPACE
		boxes.clear()
		grey_boxes.clear()
	elif key == 13:  # ENTER
		save_annotation()
		grey_boxes.clear()
		boxes.clear()
		
		frame_number = min(frame_number + 1, total_frames - 2)
		frame_updated = True
		cv2.setTrackbarPos('Frame', video_name, frame_number)

		
	elif key == ord('u'):  # Undo
		if grey_mode:
			if grey_boxes:
				grey_boxes.pop()
		elif boxes:
			boxes.pop()
	elif key == ord('g'):  # Grey mode
		grey_mode = True
		
	elif key in primary_class_dict and key in secondary_class_dict: # dual key for both primary and secondary
		if key != ord('0'):
			print("zero")
			active_primary = primary_class_dict[key]
			active_secondary = secondary_class_dict[key]
			grey_mode = False
		
	elif key in primary_class_dict:
		if key != ord('0'):
			active_primary = primary_class_dict[key]
			if active_primary < len(primary_static_classes):
				show_mode = -1 # static RGB
			else:
				show_mode =1 # motion false colour
			grey_mode = False
			
	elif key in secondary_class_dict:
		if key != ord('0'):
			active_secondary = secondary_class_dict[key]
			if active_secondary < len(secondary_static_classes):
				show_mode = -1 # static RGB
			else:
				show_mode =1 # motion false colour
			grey_mode = False
			
	elif key == 83:  # Right arrow
		frame_number = min(frame_number + 1, total_frames - 1)
		frame_updated = True
		cv2.setTrackbarPos('Frame', video_name, frame_number)
	elif key == 81:  # Left arrow
		frame_number = max(frame_number - 1, 0)
		frame_updated = True
		cv2.setTrackbarPos('Frame', video_name, frame_number)
	elif key == 46:  # > (.)
		frame_number = min(frame_number + 10, total_frames - 1)
		frame_updated = True
		cv2.setTrackbarPos('Frame', video_name, frame_number)
	elif key == 44:  # < (,)
		frame_number = max(frame_number - 10, 0)
		frame_updated = True
		cv2.setTrackbarPos('Frame', video_name, frame_number)
	elif key == 32:  # SPACE
		show_mode *= -1
	elif key == 35:  # HASH # KEY
		auto_ann_switch *= -1
		if auto_ann_switch == 1:
			auto_annotate()
		zoom_hide = 1
	elif key == 45:  # - minus
		disp_scale_factor *= 0.8
	elif key == 61:  # =+ equals (plus)
		disp_scale_factor *= 1.25
		
	if need_redraw:
		refresh_display()

capture.release()
cv2.destroyAllWindows()
print("Done annotating video: " + video_label)
