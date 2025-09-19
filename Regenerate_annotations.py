import cv2
import os
import numpy as np
import configparser
import glob
from collections import deque

def load_config(config_path):
	"""Load and parse the configuration file"""
	config = configparser.ConfigParser()
	config.read(config_path)
	
	params = {}
	try:
		# Read parameters
		params['scale_factor'] = float(config['DEFAULT'].get('scale_factor', '1.0'))
		params['expA'] = float(config['DEFAULT'].get('expA', '0.5'))
		params['expB'] = float(config['DEFAULT'].get('expB', '0.8'))
		params['strategy'] = config['DEFAULT'].get('strategy', 'exponential')
		# ~ params['chromatic_tail_only'] = config['DEFAULT'].get('chromatic_tail_only', 'false').lower() == 'false'
		params['chromatic_tail_only'] = config['DEFAULT'].get('chromatic_tail_only', 'false').lower()
		params['lum_weight'] = float(config['DEFAULT'].get('lum_weight', '0.7'))
		params['rgb_multipliers'] = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
		params['frame_skip'] = int(config['DEFAULT'].get('frame_skip', '0'))
		params['motion_threshold'] = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))
		# ~ params['motion_blocks_static'] = config['DEFAULT'].get('motion_blocks_static', 'false').lower() == 'true'
		# ~ params['static_blocks_motion'] = config['DEFAULT'].get('static_blocks_motion', 'false').lower() == 'true'
		# ~ params['save_empty_frames'] = config['DEFAULT'].get('save_empty_frames', 'false').lower() == 'true'
		params['motion_blocks_static'] = config['DEFAULT'].get('motion_blocks_static', 'false').lower()
		params['static_blocks_motion'] = config['DEFAULT'].get('static_blocks_motion', 'false').lower()
		params['save_empty_frames'] = config['DEFAULT'].get('save_empty_frames', 'false').lower()
		
		# Compute frame window size
		frame_window = 4
		if params['strategy'] == 'exponential':
			if params['expA'] > 0.2 or params['expB'] > 0.2:
				frame_window = 5
			if params['expA'] > 0.5 or params['expB'] > 0.5:
				frame_window = 10
			if params['expA'] > 0.7 or params['expB'] > 0.7:
				frame_window = 15
			if params['expA'] > 0.8 or params['expB'] > 0.8:
				frame_window = 20
			if params['expA'] > 0.9 or params['expB'] > 0.9:
				frame_window = 45
		params['frame_window'] = frame_window * (params['frame_skip'] + 1)
		
	except KeyError as e:
		raise KeyError(f"Missing configuration parameter: {e}")
	
	return params

def generate_base_images(video_path, frame_num, params):
	"""
	Generate base static and motion images for a specific video frame
	using the same processing as the annotation tool
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error opening video: {video_path}")
		return None, None
	
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if frame_num >= total_frames:
		print(f"Frame {frame_num} exceeds video length ({total_frames})")
		return None, None
	
	# Calculate start frame for the buffer
	# ~ start_frame = max(0, frame_num - params['frame_window'] + 1)
	# ~ cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
	
	prev_frames = [None] * 3
	static_img = None
	motion_img = None
	frame_count = 0
	diffs = None
	gray_static = None
	
	# Process frames leading up to the target frame
	for i in range(params['frame_window']):
		ret, frame = cap.read()
		if not ret:
			break
		
		# Apply scaling if needed
		if params['scale_factor'] != 1.0:
			frame = cv2.resize(frame, None, fx=params['scale_factor'], 
							  fy=params['scale_factor'])
		
		# Skip frames according to frame_skip
		if frame_count > 0:
			frame_count -= 1
			continue
		
		# Process this frame (every frame_skip+1 frames)
		frame_count = params['frame_skip']
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Initialize on first valid frame
		if static_img is None:
			prev_frames = [gray.copy()] * 3
			static_img = frame
			continue
		
		# Calculate differences with previous frames
		current_diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
		
		# Update previous frames based on strategy
		if params['strategy'] == 'exponential':
			prev_frames[0] = gray
			prev_frames[1] = cv2.addWeighted(
				prev_frames[1], params['expA'], 
				gray, 1 - params['expA'], 0
			)
			prev_frames[2] = cv2.addWeighted(
				prev_frames[2], params['expB'], 
				gray, 1 - params['expB'], 0
			)
		elif params['strategy'] == 'sequential':
			prev_frames[2] = prev_frames[1]
			prev_frames[1] = prev_frames[0]
			prev_frames[0] = gray
		
		# Store diffs for the target frame
		# ~ if start_frame + i == frame_num:
		static_img = frame
		gray_static = gray
		diffs = current_diffs
	
	cap.release()
	


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

	# ~ blue = cv2.addWeighted(
		# ~ gray_static, params['lum_weight'],
		# ~ diffs[0], params['rgb_multipliers'][2],
		# ~ -params['motion_threshold']
	# ~ )
	# ~ green = cv2.addWeighted(
		# ~ gray_static, params['lum_weight'],
		# ~ diffs[1], params['rgb_multipliers'][1],
		# ~ -params['motion_threshold']
	# ~ )
	# ~ red = cv2.addWeighted(
		# ~ gray_static, params['lum_weight'],
		# ~ diffs[2], params['rgb_multipliers'][0],
		# ~ -params['motion_threshold']
	# ~ )
	motion_img = cv2.merge([blue, green, red]).astype(np.uint8)
	
	return static_img, motion_img

def read_mask_file(mask_path):
	"""Read grey box coordinates from mask file"""
	boxes = []
	if os.path.exists(mask_path):
		with open(mask_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) == 4:
					boxes.append(tuple(map(int, parts)))
	return boxes

def apply_grey_boxes(image, boxes):
	"""Apply grey boxes to an image"""
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result

def apply_blocking_boxes(image, boxes):
	"""Apply blocking boxes to an image"""
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result

def get_blocking_boxes(label_path, img_w, img_h):
	"""Convert normalized label coordinates to absolute coordinates"""
	boxes = []
	if os.path.exists(label_path):
		with open(label_path, 'r') as f:
			for line in f:
				parts = line.split()
				if len(parts) < 5: 
					continue
				# Parse normalized coordinates
				xc = float(parts[1]); yc = float(parts[2])
				w = float(parts[3]); h = float(parts[4])
				# Convert to absolute coordinates
				x1 = int((xc - w/2) * img_w)
				y1 = int((yc - h/2) * img_h)
				x2 = int((xc + w/2) * img_w)
				y2 = int((yc + h/2) * img_h)
				boxes.append((x1, y1, x2, y2))
	return boxes

def regenerate_annotations(config_path):
	"""Main function to regenerate annotation images"""
	# Load configuration
	params = load_config(config_path)
	
	# Find all label files
	# ~ base_dirs = [
		# ~ ('annot_static', ['train', 'val']),
		# ~ ('annot_motion', ['train', 'val'])
	# ~ ]
	base_dirs = [
		('annot_motion', ['train', 'val'])
	]
	
	# Collect all unique base names (video_frame combinations)
	base_names = set()
	for base_dir, splits in base_dirs:
		for split in splits:
			label_dir = os.path.join(base_dir, 'labels', split)
			if not os.path.exists(label_dir):
				continue
			label_files = glob.glob(os.path.join(label_dir, '*.txt'))
			for label_file in label_files:
				if label_file.endswith('.mask.txt'):
					continue  # Skip mask files
				base_name = os.path.splitext(os.path.basename(label_file))[0]
				base_names.add((base_name, split, base_dir))
	
	# Process each unique frame
	for base_name, split, base_dir in base_names:
		# Extract video name and frame number
		parts = base_name.split('_')
		frame_num = int(parts[-1])
		video_name = '_'.join(parts[:-1])
		
		# Find video file
		video_path = None
		clips_dir = 'clips'
		for ext in ['.mp4', '.avi', '.mov', '.mkv']:
			test_path = os.path.join(clips_dir, video_name + ext)
			if os.path.exists(test_path):
				video_path = test_path
				break
		
		if not video_path:
			print(f"Video not found: {video_name}")
			print(f"Expecting video files ending with .mp4, .avi, .mov, or .mkv in /clips/ directory")
			continue
		
		# Generate base images
		static_img, motion_img = generate_base_images(video_path, frame_num, params)
		if static_img is None:
			print(f"  Could not generate images for {base_name}")
			continue
		
		# Get image dimensions
		img_h, img_w = static_img.shape[:2]
		
		# Get mask paths
		static_mask_path = os.path.join('annot_static', 'masks', split, f"{base_name}.mask.txt")
		motion_mask_path = os.path.join('annot_motion', 'masks', split, f"{base_name}.mask.txt")
		
		# Read mask files
		static_mask_boxes = read_mask_file(static_mask_path)
		motion_mask_boxes = read_mask_file(motion_mask_path)
		
		# Get label paths
		static_label_path = os.path.join('annot_static', 'labels', split, f"{base_name}.txt")
		motion_label_path = os.path.join('annot_motion', 'labels', split, f"{base_name}.txt")
		
		# ~ # Process static image
		# ~ if base_dir == 'annot_static' or params['save_empty_frames']:
			# ~ static_final = static_img.copy()
			
			# ~ # Apply grey boxes
			# ~ static_final = apply_grey_boxes(static_final, static_mask_boxes)
			
			# ~ # Apply motion blocking if enabled
			# ~ if params['motion_blocks_static']:
				# ~ motion_boxes = get_blocking_boxes(motion_label_path, img_w, img_h)
				# ~ static_final = apply_blocking_boxes(static_final, motion_boxes)
			
			# ~ # Save static image
			# ~ static_img_path = os.path.join('annot_static', 'images', split, f"{base_name}.jpg")
			# ~ os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
			# ~ cv2.imwrite(static_img_path, static_final)
			# ~ print(f"Regenerated static: {static_img_path}")
		
		# Process motion image
		if base_dir == 'annot_motion' or params['save_empty_frames']:
			if motion_img is None:
				print(f"  Could not generate motion image for {base_name}")
			else:
				motion_final = motion_img.copy()
				
				# Apply grey boxes
				motion_final = apply_grey_boxes(motion_final, motion_mask_boxes)
				
				# Apply static blocking if enabled
				if params['static_blocks_motion']:
					static_boxes = get_blocking_boxes(static_label_path, img_w, img_h)
					motion_final = apply_blocking_boxes(motion_final, static_boxes)
				
				# Save motion image
				motion_img_path = os.path.join('annot_motion', 'images', split, f"{base_name}.jpg")
				os.makedirs(os.path.dirname(motion_img_path), exist_ok=True)
				cv2.imwrite(motion_img_path, motion_final)
				print(f"Regenerated motion: {motion_img_path}")

if __name__ == "__main__":
	config_path = 'BehaveAI_settings.ini'
	if not os.path.exists(config_path):
		print(f"Config file not found: {config_path}")
	else:
		regenerate_annotations(config_path)
	print("Regeneration complete!")
