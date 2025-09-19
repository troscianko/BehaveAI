import tkinter as tk
from tkinter import scrolledtext, Button, Frame
import subprocess
import threading
import queue
import sys
import os
import re
import base64

def is_progress_line(s: str) -> bool:
	# YOLO bars nearly always have "NN%|" in them
	return bool(re.search(r"\d+% *\|", s))

# This will catch most CSI-style ANSI escapes (e.g. \x1b[1m, \x1b[0m, \x1b[32;1m, etc.)
ansi_escape = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')

def strip_ansi(s: str) -> str:
	return ansi_escape.sub('', s)
	

class TextRedirector:
	def __init__(self, text_widget, tag):
		self.text = text_widget
		self.tag  = tag
		# Configure a special tag to mark the “last inserted” region
		self.text.tag_configure("last_insert", background="")

	def write(self, s):
		"""Append new text (or a newline) and mark it as last_insert."""
		self.text.configure(state='normal')
		# Clear any old last_insert region
		self.text.tag_remove("last_insert", "1.0", "end")
		# Insert the text tagged both with stdout/stderr and last_insert
		self.text.insert("end", s, (self.tag, "last_insert"))
		self.text.see("end")
		self.text.configure(state='disabled')

	def overwrite(self, s):
		"""Delete the last_insert region, then write this text in its place."""
		self.text.configure(state='normal')
		ranges = self.text.tag_ranges("last_insert")
		if ranges:
			# Remove exactly that region
			self.text.delete(ranges[0], ranges[1])
		# Now append the new text (and mark it last_insert again)
		self.write(s)

class ScriptRunnerApp:
	def __init__(self, root):
		self.root = root
		root.title("BehaveAI Launcher")
		root.geometry("800x500")

		# Logo & Buttons
		base64_image = "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"
		image_data = base64.b64decode(base64_image)		
		self.logo_img = tk.PhotoImage(data=image_data)		
		# ~ self.logo_img = tk.PhotoImage(file="BehaveAI_200.png")
		self.button_frame = Frame(root); self.button_frame.pack(pady=10)
		tk.Label(self.button_frame, image=self.logo_img).pack(side=tk.LEFT, padx=10)
		Button(self.button_frame, text="Annotate", command=lambda: self.run_script("BehaveAI_annotation.py"), width=20).pack(side=tk.LEFT, padx=10)
		Button(self.button_frame, text="Train & batch classify", command=lambda: self.run_script("BehaveAI_classify_track.py"), width=20).pack(side=tk.LEFT, padx=10)

		# ScrolledText setup
		self.output_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='normal', height=15)
		self.output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		self.output_area.tag_config('stdout', foreground='black')
		self.output_area.tag_config('stderr', foreground='blue')

		# Wrap it in two redirectors
		self.stdout_rd = TextRedirector(self.output_area, 'stdout')
		self.stderr_rd = TextRedirector(self.output_area, 'stderr')

		# Queue and update loop
		self.output_queue = queue.Queue()
		self.update_output()

	def run_script(self, script_name):
		threading.Thread(target=self.execute_script, args=(script_name,), daemon=True).start()

	def execute_script(self, script_name):
		env = os.environ.copy()
		env['PYTHONUNBUFFERED'] = '1'
		proc = subprocess.Popen([sys.executable, '-u', script_name],
								 stdout=subprocess.PIPE,
								 stderr=subprocess.PIPE,
								 bufsize=0,
								 env=env)
		threading.Thread(target=self.read_stream, args=(proc.stdout, 'stdout'), daemon=True).start()
		threading.Thread(target=self.read_stream, args=(proc.stderr, 'stderr'), daemon=True).start()
		code = proc.wait()
		self.output_queue.put(('stdout', f"\nProcess exited with code: {code}\n".encode()))

	def read_stream(self, stream, tag):
		while True:
			chunk = stream.read(1)
			if not chunk:
				break
			self.output_queue.put((tag, chunk))
		stream.close()

	def update_output(self):
		if not hasattr(self, 'output_buffer'):
			self.output_buffer = {'stdout': b'', 'stderr': b''}

		while not self.output_queue.empty():
			tag, data = self.output_queue.get()
			buf = self.output_buffer[tag] + data

			# CR case
			if buf.endswith(b'\r'):
				raw = buf[:-1]
				# ~ line = raw.decode('utf-8', errors='replace')
				line = strip_ansi(raw.decode('utf-8', errors='replace'))
				rd = self.stdout_rd if tag=='stdout' else self.stderr_rd

				if is_progress_line(line):
					# true progress → overwrite last bar
					rd.overwrite(line)
				else:
					# header or any other CR line → force it as a normal line
					rd.write(line + '\n')

				buf = b''

			# NL case
			elif buf.endswith(b'\n'):
				raw = buf[:-1]
				# ~ line = raw.decode('utf-8', errors='replace') + '\n'
				line = strip_ansi(raw.decode('utf-8', errors='replace')) + '\n'
				rd = self.stdout_rd if tag=='stdout' else self.stderr_rd

				rd.write(line)
				buf = b''

			# otherwise, keep buffering
			self.output_buffer[tag] = buf

		self.root.after(50, self.update_output)


if __name__ == "__main__":
	root = tk.Tk()
	app = ScriptRunnerApp(root)
	root.mainloop()
