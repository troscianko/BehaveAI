import tkinter as tk
from tkinter import scrolledtext, Button, Frame
import subprocess
import threading
import queue
import sys
import os

class ScriptRunnerApp:
	def __init__(self, root):
		self.root = root
		root.title("BehaveAI Launcher")
		root.geometry("800x500")

		from tkinter import PhotoImage
		self.logo_img = PhotoImage(file="BehaveAI_200.png")  
	

		# Create main layout
		self.button_frame = Frame(root)
		self.button_frame.pack(pady=10)
		
		self.output_frame = Frame(root)
		self.output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

		# Add image to the left
		self.logo_label = tk.Label(self.button_frame, image=self.logo_img)
		self.logo_label.pack(side=tk.LEFT, padx=10)

		# Create buttons
		self.btn_annotate = Button(
			self.button_frame,
			text="Annotate",
			command=lambda: self.run_script("BehaveAI_annotation.py"),
			width=20
		)
		self.btn_annotate.pack(side=tk.LEFT, padx=10)
		
		self.btn_classify = Button(
			self.button_frame,
			text="Train & batch classify",
			command=lambda: self.run_script("BehaveAI_classify_track.py"),
			width=20
		)
		self.btn_classify.pack(side=tk.LEFT, padx=10)

	
		# ~ self.btn_clear = Button(
			# ~ self.button_frame,
			# ~ text="Clear Output",
			# ~ command=self.clear_output,
			# ~ width=10
		# ~ )
		# ~ self.btn_clear.pack(side=tk.LEFT, padx=10)
		
		# Create output display
		self.output_area = scrolledtext.ScrolledText(
			self.output_frame,
			wrap=tk.WORD,
			state='normal',
			height=15
		)
		self.output_area.pack(fill=tk.BOTH, expand=True)
		self.output_area.tag_config('stdout', foreground='black')
		self.output_area.tag_config('stderr', foreground='red')
		
		# Create queue for thread-safe communication
		self.output_queue = queue.Queue()
		self.update_output()
	
	def run_script(self, script_name):
		"""Start a new thread to run the selected script"""
		threading.Thread(
			target=self.execute_script,
			args=(script_name,),
			daemon=True
		).start()
	
	def execute_script(self, script_name):
		"""Execute the specified script and capture its output with real-time updates"""
		try:
			# Create environment with PYTHONUNBUFFERED
			env = os.environ.copy()
			env['PYTHONUNBUFFERED'] = '1'
			
			# Start the subprocess with unbuffered output
			process = subprocess.Popen(
				[sys.executable, '-u', script_name],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				bufsize=1,  # Line buffered
				universal_newlines=True,
				env=env
			)
			
			# Create separate readers for stdout and stderr
			threading.Thread(
				target=self.read_stream,
				args=(process.stdout, 'stdout'),
				daemon=True
			).start()
			
			threading.Thread(
				target=self.read_stream,
				args=(process.stderr, 'stderr'),
				daemon=True
			).start()
			
			# Wait for process to complete
			process.wait()
			self.output_queue.put(('stdout', f"\nProcess exited with code: {process.returncode}\n"))
			
		except Exception as e:
			self.output_queue.put(('stderr', f"Error running script: {str(e)}\n"))
	
	def read_stream(self, stream, tag):
		"""Read from a stream in a separate thread and put lines into the queue"""
		try:
			while True:
				line = stream.readline()
				if not line:
					break
				self.output_queue.put((tag, line))
		finally:
			stream.close()
		
	def update_output(self):
		"""Check for new output from the queue every 100ms"""
		while not self.output_queue.empty():
			tag, line = self.output_queue.get()
			self.output_area.configure(state='normal')
	
			if f"Process exited with code" in line:
				self.output_area.insert(tk.END, 'Finished\n', tag)
			elif line.endswith('\r') or line.startswith('Progress:') or line.startswith('Done processing'):
				# Overwrite the last line in the text box
				self.output_area.delete("end-2l", "end-1l")  # Delete previous line
				self.output_area.insert(tk.END, line.rstrip('\r'), tag)
			else:
				self.output_area.insert(tk.END, line, tag)
	
			self.output_area.see(tk.END)
			self.output_area.configure(state='disabled')
		
		self.root.after(100, self.update_output)

	
	def clear_output(self):
		"""Clear the output display"""
		self.output_area.configure(state='normal')
		self.output_area.delete(1.0, tk.END)
		self.output_area.configure(state='disabled')

if __name__ == "__main__":
	root = tk.Tk()
	app = ScriptRunnerApp(root)
	root.mainloop()
