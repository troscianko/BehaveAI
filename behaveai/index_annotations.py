# index_annotations.py
# Helper class to list annotated images and load saved labels/masks / find video files.
# This duplicates the logic used in your inspector verbatim (so behaviour stays identical).

import os
import cv2
import numpy as np

class AnnotationIndex:
	def __init__(
		self,
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
		ignore_secondary=None
	):
		# directories
		self.static_train_images_dir = static_train_images_dir
		self.static_val_images_dir = static_val_images_dir
		self.static_train_labels_dir = static_train_labels_dir
		self.static_val_labels_dir = static_val_labels_dir
		self.motion_train_images_dir = motion_train_images_dir
		self.motion_val_images_dir = motion_val_images_dir
		self.motion_train_labels_dir = motion_train_labels_dir
		self.motion_val_labels_dir = motion_val_labels_dir
		self.motion_cropped_base_dir = motion_cropped_base_dir
		self.static_cropped_base_dir = static_cropped_base_dir
		self.clips_dir = clips_dir

		# class lists & mode
		self.primary_static_classes = list(primary_static_classes) if primary_static_classes is not None else []
		self.primary_classes = list(primary_classes) if primary_classes is not None else []
		self.secondary_classes = list(secondary_classes) if secondary_classes is not None else []
		self.hierarchical_mode = bool(hierarchical_mode)
		self.ignore_secondary = set(ignore_secondary or [])



	# ------------------------------------------------------------------
	# Build list of annotated images (same behaviour as your inspector)
	# ------------------------------------------------------------------
	def list_images_labels_and_masks(self):
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

		add_dir(self.static_train_images_dir, self.static_train_labels_dir)
		# ---- FIX: pair static_val_images_dir with static_val_labels_dir (was a typo previously) ----
		add_dir(self.static_val_images_dir, self.static_val_labels_dir)
		add_dir(self.motion_train_images_dir, self.motion_train_labels_dir)
		add_dir(self.motion_val_images_dir, self.motion_val_labels_dir)

		ordered = []
		for base, rec in sorted(items.items()):
			ordered.append({'basename': base, **rec})
		return ordered

	# ------------------------------------------------------------------
	# Find a video in clips_dir corresponding to an annotation item basename
	# ------------------------------------------------------------------
	def find_video_for_item(self, item):
		if not os.path.isdir(self.clips_dir):
			return None, None
		base = item.get('basename', '')
		if '_' not in base:
			return None, None
		video_label_guess, tail = base.rsplit('_', 1)
		frame_number_guess = None
		try:
			frame_number_guess = int(tail)
		except Exception:
			frame_number_guess = None

		for fname in os.listdir(self.clips_dir):
			if not fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
				continue
			stem = os.path.splitext(fname)[0]
			if stem.lower() == video_label_guess.lower():
				return os.path.join(self.clips_dir, fname), frame_number_guess

		for fname in os.listdir(self.clips_dir):
			if not fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
				continue
			stem = os.path.splitext(fname)[0]
			if stem.lower().startswith(video_label_guess.lower()):
				return os.path.join(self.clips_dir, fname), frame_number_guess

		return None, None

	# ------------------------------------------------------------------
	# Load the labels & masks for an item into boxes and grey_boxes lists.
	# Returns (boxes, grey_boxes) but does NOT yet attach secondary crops.
	# ------------------------------------------------------------------
	def load_labels_and_masks_for_item(self, item, fr, original_frame):
		boxes = []
		grey_boxes = []
		base = item.get('basename', '')

		# static labels
		static_lbl = item.get('static_lbl')
		if static_lbl and os.path.exists(static_lbl):
			try:
				with open(static_lbl, 'r') as f:
					for line in f:
						parts = line.strip().split()
						if len(parts) < 5:
							continue
						cls = int(parts[0])
						xc, yc, bw, bh = parts[1:5]
						if fr is None:
							# can't compute pixel coords; skip
							continue
						h, w = fr.shape[:2]
						x1, y1, x2, y2 = self._norm_to_pixels(xc, yc, bw, bh, w, h)
						if self.hierarchical_mode:
							# use -1 for "no secondary assigned"
							boxes.append((x1, y1, x2, y2, cls, -1, -1, -1))
						else:
							boxes.append((x1, y1, x2, y2, cls, -1))
			except Exception:
				pass

		# motion labels
		motion_lbl = item.get('motion_lbl')
		if motion_lbl and os.path.exists(motion_lbl):
			try:
				with open(motion_lbl, 'r') as f:
					for line in f:
						parts = line.strip().split()
						if len(parts) < 5:
							continue
						cls = int(parts[0])
						xc, yc, bw, bh = parts[1:5]
						if original_frame is None:
							continue
						h, w = original_frame.shape[:2]
						x1, y1, x2, y2 = self._norm_to_pixels(xc, yc, bw, bh, w, h)
						global_primary_cls = cls + len(self.primary_static_classes)
						if self.hierarchical_mode:
							boxes.append((x1, y1, x2, y2, global_primary_cls, -1, -1, -1))
						else:
							boxes.append((x1, y1, x2, y2, global_primary_cls, -1))
			except Exception:
				pass

		# masks (prefer static mask then motion mask)
		mask_path = item.get('static_mask') or item.get('motion_mask')
		if mask_path and os.path.exists(mask_path):
			try:
				with open(mask_path, 'r') as f:
					for line in f:
						parts = line.strip().split()
						if len(parts) >= 4:
							gx1, gy1, gx2, gy2 = map(int, parts[:4])
							grey_boxes.append((gx1, gy1, gx2, gy2))
			except Exception:
				pass

		# if hierarchical_mode is enabled, attach secondary crops now
		if self.hierarchical_mode and boxes:
			boxes = self._attach_secondary_crops(item, boxes)

		return boxes, grey_boxes

	# ------------------------------------------------------------------
	# Convenience: load labels by basename (used by annotation script which constructs basename)
	# It will construct a lightweight "item" dict (checking expected label & mask locations)
	# and then call load_labels_and_masks_for_item(...)
	# ------------------------------------------------------------------
	def load_labels_for_basename(self, base_fn, fr, original_frame):
		item = {'basename': base_fn, 'static_img': None, 'motion_img': None, 'static_lbl': None, 'motion_lbl': None, 'static_mask': None, 'motion_mask': None}
		# static label search
		for d in (self.static_train_labels_dir, self.static_val_labels_dir):
			if d and os.path.isdir(d):
				p = os.path.join(d, base_fn + '.txt')
				if os.path.exists(p):
					item['static_lbl'] = p
					item['static_origin_lbl_dir'] = d
					item['static_origin_img_dir'] = d.replace('labels','images')
					break
		# motion label search
		for d in (self.motion_train_labels_dir, self.motion_val_labels_dir):
			if d and os.path.isdir(d):
				p = os.path.join(d, base_fn + '.txt')
				if os.path.exists(p):
					item['motion_lbl'] = p
					item['motion_img'] = os.path.join(d.replace('labels','images'), base_fn + '.jpg') if os.path.isdir(d.replace('labels','images')) else None
					item['motion_origin_lbl_dir'] = d
					break
		# masks
		for d in (self.static_train_labels_dir, self.static_val_labels_dir, self.motion_train_labels_dir, self.motion_val_labels_dir):
			if d and os.path.isdir(d):
				mp = os.path.join(d.replace('labels','masks'), base_fn + '.mask.txt')
				if os.path.exists(mp):
					item['static_mask'] = mp
					break

		return self.load_labels_and_masks_for_item(item, fr, original_frame)

	# ------------------------------------------------------------------
	# helper: parse crop filename pattern: <video_label>_<frame>_<x1>_<y1>.<ext>
	# ------------------------------------------------------------------
	def _parse_crop_filename(self, fn):
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

	# ------------------------------------------------------------------
	# Attach secondary crop matches to boxes (in-place semantics via returning a new list)
	# Implementation follows the inspector behaviour (exact match then small neighbourhood)
	# ------------------------------------------------------------------
	def _attach_secondary_crops(self, item, boxes):
		MATCH_TOL = 2
		# build map by (x1, y1, primary_name) -> list of box indices
		box_index = {}
		for bi, b in enumerate(boxes):
			bx1 = int(round(b[0])); by1 = int(round(b[1]))
			primary_idx = b[4] if len(b) > 4 else None
			primary_name = self.primary_classes[primary_idx] if primary_idx is not None and primary_idx < len(self.primary_classes) else None
			key = (bx1, by1, primary_name)
			box_index.setdefault(key, []).append(bi)

		# prepare mapping secondary dir name -> index
		sec_name_to_idx = {name: idx for idx, name in enumerate(self.secondary_classes)}

		# parse video_label and frame from basename
		if '_' in item.get('basename', ''):
			video_label_guess, tail = item['basename'].rsplit('_', 1)
			try:
				frame_number_guess = int(tail)
			except Exception:
				frame_number_guess = None
		else:
			video_label_guess = item.get('basename', '')
			frame_number_guess = None

		# scan both cropped base dirs (motion then static)
		for base_crop_dir in (self.motion_cropped_base_dir, self.static_cropped_base_dir):
			if not base_crop_dir or not os.path.isdir(base_crop_dir):
				continue
			for primary_name in os.listdir(base_crop_dir):
				prim_dir = os.path.join(base_crop_dir, primary_name)
				if not os.path.isdir(prim_dir):
					continue
				for secondary_name in os.listdir(prim_dir):
					sec_dir = os.path.join(prim_dir, secondary_name)
					if not os.path.isdir(sec_dir):
						continue
					sec_idx = sec_name_to_idx.get(secondary_name)
					if sec_idx is None:
						continue
					for fn in os.listdir(sec_dir):
						if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
							continue
						parsed = self._parse_crop_filename(fn)
						if parsed is None:
							continue
						vlabel_part, fn_frame, x1_fn, y1_fn = parsed
						if vlabel_part != video_label_guess or fn_frame != frame_number_guess:
							continue
						# exact key match first
						key = (x1_fn, y1_fn, primary_name)
						matched = False
						if key in box_index:
							for bi in box_index[key]:
								b = boxes[bi]
								if len(b) >= 8:
									boxes[bi] = (b[0], b[1], b[2], b[3], b[4], sec_idx, b[6], b[7])
								else:
									primary_cls = b[4] if len(b) > 4 else 0
									conf = b[6] if len(b) > 6 else -1
									boxes[bi] = (b[0], b[1], b[2], b[3], primary_cls, sec_idx, conf, -1)
								matched = True
						if matched:
							continue
						# otherwise small neighbourhood search
						for dx in range(-MATCH_TOL, MATCH_TOL + 1):
							if matched: break
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
									if matched: break
							if matched: break

		return boxes

	# small helper used above (copied behaviour)
	def _norm_to_pixels(self, xc, yc, bw, bh, w, h):
		cx = float(xc) * w
		cy = float(yc) * h
		bw_p = float(bw) * w
		bh_p = float(bh) * h
		x1 = int(cx - bw_p/2); y1 = int(cy - bh_p/2)
		x2 = int(cx + bw_p/2); y2 = int(cy + bh_p/2)
		x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1)); x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
		return x1, y1, x2, y2

	# ------------------------------------------------------------------
	# Delete all saved files for a basename (labels, masks, images, original motion images,
	# and cropped secondary images when hierarchical_mode is enabled).
	# Returns list of deleted file paths (empty list if none).
	# ------------------------------------------------------------------
	def delete_frame(self, base_filename):
	# ~ def delete_frame(self, base_filename, video_label=None, frame_number=None):
		deleted = []

		# Label directories
		label_dirs = [
			self.static_train_labels_dir,
			self.static_val_labels_dir,
			self.motion_train_labels_dir,
			self.motion_val_labels_dir,
		]

		# Mask directories are labels -> masks
		mask_dirs = [d.replace('labels', 'masks') if d else None for d in label_dirs]

		# Image directories
		image_dirs = [
			self.static_train_images_dir,
			self.static_val_images_dir,
			self.motion_train_images_dir,
			self.motion_val_images_dir,
		]

		# File extensions to consider for images
		image_exts = ('.jpg', '.jpeg', '.png')

		# --- delete label files (.txt) ---
		for d in label_dirs:
			if not d:
				continue
			p = os.path.join(d, base_filename + '.txt')
			if os.path.exists(p):
				try:
					os.remove(p)
					deleted.append(p)
				except Exception:
					# intentionally continue on errors
					pass

		# --- delete mask files (.mask.txt) ---
		for d in mask_dirs:
			if not d:
				continue
			p = os.path.join(d, base_filename + '.mask.txt')
			if os.path.exists(p):
				try:
					os.remove(p)
					deleted.append(p)
				except Exception:
					pass

		# --- delete image files in expected image dirs ---
		for d in image_dirs:
			if not d:
				continue
			for ext in image_exts:
				p = os.path.join(d, base_filename + ext)
				if os.path.exists(p):
					try:
						os.remove(p)
						deleted.append(p)
					except Exception:
						pass

		# --- delete 'original' motion images if they exist (keeps parity with annot code) ---
		for parent in (self.motion_train_images_dir, self.motion_val_images_dir):
			if not parent:
				continue
			od = os.path.join(parent, 'original')
			if not os.path.isdir(od):
				continue
			for ext in image_exts:
				p = os.path.join(od, base_filename + ext)
				if os.path.exists(p):
					try:
						os.remove(p)
						deleted.append(p)
					except Exception:
						pass

		# --- delete cropped secondary images when hierarchical_mode is enabled ---
		# These use filenames like: <video_label>_<frame>_<x1>_<y1>.jpg
		# ~ if self.hierarchical_mode and video_label is not None and frame_number is not None:
		if self.hierarchical_mode and base_filename is not None:
			# ~ prefix = f"{video_label}_{frame_number}_"
			for base_cropped_dir in (self.motion_cropped_base_dir, self.static_cropped_base_dir):
				if not base_cropped_dir or not os.path.isdir(base_cropped_dir):
					continue
				for root, _, files in os.walk(base_cropped_dir):
					for fname in files:
						# quick prefix + ext check
						lf = fname.lower()
						if not any(lf.endswith(ext) for ext in image_exts):
							continue
						# ~ if fname.startswith(prefix):
						if fname.startswith(base_filename):
							full = os.path.join(root, fname)
							if os.path.exists(full):
								try:
									os.remove(full)
									deleted.append(full)
								except Exception:
									pass

		return deleted
