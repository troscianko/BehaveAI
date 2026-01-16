#!/usr/bin/env python3
"""
Regenerate motion annotation images for a BehaveAI project.

Usage:
	python regenerate_motion_dataset.py <path/to/BehaveAI_settings.ini>
or:
	python regenerate_motion_dataset.py		# will prompt for INI via file dialog

This script:
 - reads the settings INI (and resolves relative paths relative to the INI's directory)
 - rebuilds motion images (annot_motion/images/{train,val}) using the same processing
   as the annotation tool (sampling a small window of frames, computing diffs, chromatic tail, etc.)
 - applies masks and blocking boxes in the same way as your annotator
"""
import cv2
import os
import numpy as np
import configparser
import glob
import sys
import time
from collections import deque

# optional GUI prompt if INI not supplied
try:
	import tkinter as tk
	from tkinter import filedialog, messagebox
	_HAS_TK = True
except Exception:
	_HAS_TK = False

# -----------------------
# Helpers: path resolve / config loader
# -----------------------

def resolve_project_path(project_dir, value, fallback):
	"""Resolve a path specified in the INI: absolute or relative to project_dir."""
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	if os.path.isabs(value):
		return os.path.normpath(value)
	return os.path.normpath(os.path.join(project_dir, value))


def load_config(config_path):
	"""
	Read configuration from config_path and return (params_dict, clips_dir_resolved).
	params contains numeric / strategy settings used by the image generation pipeline.
	clips_dir_resolved is an absolute (or normalized) path to the clips directory resolved
	relative to the INI's project directory.
	"""
	config = configparser.ConfigParser()
	config.optionxform = str  # preserve case
	config.read(config_path)

	project_dir = os.path.dirname(os.path.abspath(config_path))

	params = {}
	try:
		# Read parameters (same names as your previous implementation)
		params['scale_factor'] = float(config['DEFAULT'].get('scale_factor', '1.0'))
		params['expA'] = float(config['DEFAULT'].get('expA', '0.5'))
		params['expB'] = float(config['DEFAULT'].get('expB', '0.8'))
		params['strategy'] = config['DEFAULT'].get('strategy', 'exponential')
		params['chromatic_tail_only'] = config['DEFAULT'].get('chromatic_tail_only', 'false').lower()
		params['lum_weight'] = float(config['DEFAULT'].get('lum_weight', '0.7'))
		params['rgb_multipliers'] = [float(x) for x in config['DEFAULT'].get('rgb_multipliers', '2,2,2').split(',')]
		params['frame_skip'] = int(config['DEFAULT'].get('frame_skip', '0'))
		params['motion_threshold'] = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))
		params['motion_blocks_static'] = config['DEFAULT'].get('motion_blocks_static', 'false').lower()
		params['static_blocks_motion'] = config['DEFAULT'].get('static_blocks_motion', 'false').lower()
		params['save_empty_frames'] = config['DEFAULT'].get('save_empty_frames', 'false').lower()

		# Compute base frame window size (number of sampled frames)
		base_window = 4
		if params['strategy'] == 'exponential':
			if params['expA'] > 0.2 or params['expB'] > 0.2:
				base_window = 5
			if params['expA'] > 0.5 or params['expB'] > 0.5:
				base_window = 10
			if params['expA'] > 0.7 or params['expB'] > 0.7:
				base_window = 15
			if params['expA'] > 0.8 or params['expB'] > 0.8:
				base_window = 20
			if params['expA'] > 0.9 or params['expB'] > 0.9:
				base_window = 45

		params['base_frame_window'] = base_window
		params['frame_window'] = base_window * (params['frame_skip'] + 1)

	except KeyError as e:
		raise KeyError(f"Missing configuration parameter: {e}")

	# Resolve clips_dir relative to project_dir (fallback 'clips')
	clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
	clips_dir = resolve_project_path(project_dir, clips_dir_ini, 'clips')

	return params, clips_dir


# -----------------------
# Image processing helpers (unchanged logic besides small improvements)
# -----------------------

def generate_base_images(video_path, frame_num, params):
	"""
	Generate static and motion images for a specific video frame.
	frame_num is interpreted as the LAST frame of the motion window to mimic the annotator.
	Returns (static_img_bgr, motion_img_bgr) or (None, None) on failure.
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error opening video: {video_path}")
		return None, None

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if total_frames <= 0:
		print(f"Video appears empty or unreadable: {video_path}")
		cap.release()
		return None, None

	step = params['frame_skip'] + 1
	base_N = params.get('base_frame_window', 4)

	# compute start so last appended index should equal frame_num
	start_frame = int(frame_num - (base_N - 1) * step)
	start_frame = max(0, start_frame)
	if start_frame > total_frames - 1:
		print(f"Start frame {start_frame} beyond video length ({total_frames}) for {video_path}")
		cap.release()
		return None, None

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	collected = []
	read_count = 0
	idx = start_frame
	# safety limit: don't try more than frame_window + some slack
	max_reads = params['frame_window'] + 10

	while len(collected) < base_N and idx <= total_frames - 1 and read_count <= max_reads:
		ret, frame = cap.read()
		if not ret:
			break
		if (read_count % step) == 0:
			if params['scale_factor'] != 1.0:
				frame = cv2.resize(frame, None, fx=params['scale_factor'], fy=params['scale_factor'])
			collected.append(frame.copy())
		read_count += 1
		idx += 1

	if not collected:
		cap.release()
		print(f"Could not collect frames for target {frame_num} (start {start_frame}) in {video_path}")
		return None, None

	# Process collected frames to produce diffs for the last frame
	prev_frames = [None] * 3
	static_img = None
	diffs = None
	gray = None

	for i, f in enumerate(collected):
		if f is None:
			continue
		frame_bgr = f
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

		if static_img is None:
			static_img = frame_bgr.copy()
			prev_frames = [gray.copy()] * 3
			continue

		current_diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

		if params['strategy'] == 'exponential':
			prev_frames[0] = gray
			prev_frames[1] = cv2.addWeighted(prev_frames[1], params['expA'], gray, 1 - params['expA'], 0)
			prev_frames[2] = cv2.addWeighted(prev_frames[2], params['expB'], gray, 1 - params['expB'], 0)
		elif params['strategy'] == 'sequential':
			prev_frames[2] = prev_frames[1]
			prev_frames[1] = prev_frames[0]
			prev_frames[0] = gray

		static_img = frame_bgr.copy()
		diffs = current_diffs

	if diffs is None or gray is None:
		cap.release()
		print(f"Insufficient frames to compute diffs for {frame_num} (collected {len(collected)} frames)")
		return None, None

	# Build motion image (chromatic tail or normal)
	if params['chromatic_tail_only'] == 'true':
		tb = cv2.subtract(diffs[0], diffs[1])
		tr = cv2.subtract(diffs[2], diffs[1])
		tg = cv2.subtract(diffs[1], diffs[0])

		blue = cv2.addWeighted(gray, params['lum_weight'], tb, params['rgb_multipliers'][2], params['motion_threshold'])
		green = cv2.addWeighted(gray, params['lum_weight'], tg, params['rgb_multipliers'][1], params['motion_threshold'])
		red = cv2.addWeighted(gray, params['lum_weight'], tr, params['rgb_multipliers'][0], params['motion_threshold'])
	else:
		blue = cv2.addWeighted(gray, params['lum_weight'], diffs[0], params['rgb_multipliers'][2], params['motion_threshold'])
		green = cv2.addWeighted(gray, params['lum_weight'], diffs[1], params['rgb_multipliers'][1], params['motion_threshold'])
		red = cv2.addWeighted(gray, params['lum_weight'], diffs[2], params['rgb_multipliers'][0], params['motion_threshold'])

	motion_img = cv2.merge([blue, green, red]).astype(np.uint8)

	cap.release()
	return static_img, motion_img


def read_mask_file(mask_path):
	boxes = []
	if os.path.exists(mask_path):
		with open(mask_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) == 4:
					try:
						boxes.append(tuple(map(int, parts)))
					except Exception:
						pass
	return boxes


def apply_grey_boxes(image, boxes):
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result


def apply_blocking_boxes(image, boxes):
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result


def get_blocking_boxes(label_path, img_w, img_h):
	boxes = []
	if os.path.exists(label_path):
		with open(label_path, 'r') as f:
			for line in f:
				parts = line.split()
				if len(parts) < 5:
					continue
				try:
					xc = float(parts[1]); yc = float(parts[2])
					w = float(parts[3]); h = float(parts[4])
				except Exception:
					continue
				x1 = int((xc - w/2) * img_w)
				y1 = int((yc - h/2) * img_h)
				x2 = int((xc + w/2) * img_w)
				y2 = int((yc + h/2) * img_h)
				boxes.append((x1, y1, x2, y2))
	return boxes


# -----------------------
# Main regeneration function
# -----------------------

def regenerate_annotations(config_path):
	"""Regenerate motion images using parameters & clips_dir from config_path."""
	params, clips_dir = load_config(config_path)

	# Ensure we operate with project_dir as cwd to keep relative paths consistent
	project_dir = os.path.dirname(os.path.abspath(config_path))
	os.chdir(project_dir)

	print(f"Regenerating using INI: {config_path}")
	print(f"Using clips directory: {clips_dir}")

	# base_dirs currently only needs motion; keep same structure in case you extend
	base_dirs = [
		('annot_motion', ['train', 'val'])
	]

	# collect unique base names from these motion label directories
	base_names = set()
	for base_dir, splits in base_dirs:
		for split in splits:
			label_dir = os.path.join(base_dir, 'labels', split)
			if not os.path.exists(label_dir):
				continue
			for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
				if label_file.endswith('.mask.txt'):
					continue
				base_name = os.path.splitext(os.path.basename(label_file))[0]
				base_names.add((base_name, split, base_dir))

	print(f"Found {len(base_names)} annotated motion frames to process.")

	# extensions to search for video files
	exts = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

	# process each unique frame
	for base_name, split, base_dir in sorted(base_names):
		parts = base_name.split('_')
		try:
			frame_num = int(parts[-1])
		except ValueError:
			print(f"Skipping {base_name}: trailing token is not an integer")
			continue
		video_name = '_'.join(parts[:-1])

		# find video in clips_dir
		video_path = None
		for ext in exts:
			test_path = os.path.join(clips_dir, video_name + ext)
			if os.path.exists(test_path):
				video_path = test_path
				break

		if not video_path:
			print(f"Video not found for {base_name}: looking in {clips_dir} for files named {video_name}.*")
			continue

		static_img, motion_img = generate_base_images(video_path, frame_num, params)
		if static_img is None:
			print(f"  Could not generate images for {base_name}")
			continue

		img_h, img_w = static_img.shape[:2]

		static_mask_path = os.path.join('annot_static', 'masks', split, f"{base_name}.mask.txt")
		motion_mask_path = os.path.join('annot_motion', 'masks', split, f"{base_name}.mask.txt")

		static_mask_boxes = read_mask_file(static_mask_path)
		motion_mask_boxes = read_mask_file(motion_mask_path)

		static_label_path = os.path.join('annot_static', 'labels', split, f"{base_name}.txt")
		motion_label_path = os.path.join('annot_motion', 'labels', split, f"{base_name}.txt")

		# Process motion images (save into annot_motion/images/<split>/)
		if base_dir == 'annot_motion' or params['save_empty_frames'] == 'true':
			if motion_img is None:
				print(f"  No motion image for {base_name}")
			else:
				motion_final = motion_img.copy()

				# Apply grey boxes
				motion_final = apply_grey_boxes(motion_final, motion_mask_boxes)

				# Apply static blocking if enabled
				if params['static_blocks_motion'] == 'true':
					static_boxes = get_blocking_boxes(static_label_path, img_w, img_h)
					motion_final = apply_blocking_boxes(motion_final, static_boxes)

				motion_img_path = os.path.join('annot_motion', 'images', split, f"{base_name}.jpg")
				os.makedirs(os.path.dirname(motion_img_path), exist_ok=True)
				cv2.imwrite(motion_img_path, motion_final)
				print(f"Regenerated motion: {motion_img_path}")

	print("Regeneration loop complete.")


# -----------------------
# CLI & prompt logic
# -----------------------

def choose_ini_path_via_dialog():
	if not _HAS_TK:
		return None
	root = tk.Tk()
	root.withdraw()
	path = filedialog.askopenfilename(title="Select BehaveAI settings INI", filetypes=[("INI files", "*.ini"), ("All files", "*.*")])
	root.destroy()
	return path

if __name__ == "__main__":
	# Determine config_path from command-line or prompt
	if len(sys.argv) > 1:
		arg = os.path.abspath(sys.argv[1])
		if os.path.isdir(arg):
			config_path = os.path.join(arg, "BehaveAI_settings.ini")
		else:
			config_path = arg
	else:
		config_path = choose_ini_path_via_dialog()
		if not config_path:
			# no selection: report and exit
			print("No settings INI selected — exiting.")
			sys.exit(0)

	config_path = os.path.abspath(config_path)
	if not os.path.exists(config_path):
		print(f"Config file not found: {config_path}")
		sys.exit(1)

	# Run regeneration
	start_t = time.time()
	regenerate_annotations(config_path)
	elapsed = time.time() - start_t
	print(f"Regeneration complete! Elapsed {elapsed:.1f} s")
