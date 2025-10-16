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
    # YOLO bars nearly always have "NN%|" in them - required to deal with line updates
    return bool(re.search(r"\d+% *\|", s))

ansi_escape = re.compile(r'\x1B\[[0-9;?]*[ -/]*[@-~]')

def strip_ansi(s: str) -> str:
    return ansi_escape.sub('', s)


class TextRedirector:
    def __init__(self, text_widget, tag):
        self.text = text_widget
        self.tag = tag
        # Use a per-stream last_insert tag so stdout/stderr don't interfere
        self.last_tag = f"last_insert_{tag}"
        self.text.tag_configure(self.last_tag, background="")

    def _is_view_at_bottom(self) -> bool:
        # yview returns (top_fraction, bottom_fraction)
        try:
            top, bottom = self.text.yview()
            # treat as bottom if the bottom fraction is very close to 1.0
            return bottom >= 0.995
        except Exception:
            # in odd cases, assume at bottom
            return True

    def write(self, s):
        #Append new text (or a newline) and mark it as last_insert for this stream.
        at_bottom = self._is_view_at_bottom()
        self.text.configure(state='normal')
        # Clear any old last_insert region for THIS stream only
        self.text.tag_remove(self.last_tag, "1.0", "end")
        # Insert the text tagged with the stream tag and this stream's last_insert tag
        self.text.insert("end", s, (self.tag, self.last_tag))
        if at_bottom:
            self.text.see("end")
        self.text.configure(state='disabled')

    def overwrite(self, s):
        #Delete this stream's last_insert region, then write this text in its place.
        at_bottom = self._is_view_at_bottom()
        self.text.configure(state='normal')
        ranges = self.text.tag_ranges(self.last_tag)
        if ranges:
            # Remove exactly that region
            self.text.delete(ranges[0], ranges[1])
        # Now append the new text (and mark it last_insert again)
        self.text.insert("end", s, (self.tag, self.last_tag))
        if at_bottom:
            self.text.see("end")
        self.text.configure(state='disabled')


class ScriptRunnerApp:
    def __init__(self, root):
        self.root = root
        root.title("BehaveAI Launcher")
        root.geometry("900x500")

		#Logo
        base64_image = "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"
        image_data = base64.b64decode(base64_image)
        self.logo_img = tk.PhotoImage(data=image_data)
        self.button_frame = Frame(root); self.button_frame.pack(pady=10)
        tk.Label(self.button_frame, image=self.logo_img).pack(side=tk.LEFT, padx=10)
        Button(self.button_frame, text="Annotate", command=lambda: self.run_script("BehaveAI_annotation.py"), width=20).pack(side=tk.LEFT, padx=10)
        Button(self.button_frame, text="Inspect Dataset", command=lambda: self.run_script("BehaveAI_inspect_dataset.py"), width=20).pack(side=tk.LEFT, padx=10)
        Button(self.button_frame, text="Train & batch classify", command=lambda: self.run_script("BehaveAI_classify_track.py"), width=20).pack(side=tk.LEFT, padx=10)

        # ScrolledText setup
        self.output_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='normal', height=20)
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.output_area.tag_config('stdout', foreground='black')
        self.output_area.tag_config('stderr', foreground='blue')

        # Wrap it in two redirectors (they now have separate last_insert tags)
        self.stdout_rd = TextRedirector(self.output_area, 'stdout')
        self.stderr_rd = TextRedirector(self.output_area, 'stderr')

        # Queue and update loop
        self.output_queue = queue.Queue()
        self.output_buffer = {'stdout': b'', 'stderr': b''}

        # Track most-recent progress string globally to dedupe across streams
        self.last_progress_global = None

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
        # Friendly exit messages: "Done" for success, show code on failure
        if code == 0:
            self.output_queue.put(('stdout', b"\nDone\n"))
        else:
            self.output_queue.put(('stdout', f"\nProcess exited with code: {code}\n".encode()))


    def read_stream(self, stream, tag):
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            self.output_queue.put((tag, chunk))
        stream.close()

    def update_output(self):
        while not self.output_queue.empty():
            tag, data = self.output_queue.get()
            buf = self.output_buffer[tag] + data

            # CR case (progress bar carriage return)
            if buf.endswith(b'\r'):
                raw = buf[:-1]
                line_plain = strip_ansi(raw.decode('utf-8', errors='replace'))
                rd = self.stdout_rd if tag == 'stdout' else self.stderr_rd

                if is_progress_line(line_plain):
                    # overwrite last bar and remember globally
                    rd.overwrite(line_plain)
                    self.last_progress_global = line_plain.strip()
                else:
                    rd.write(line_plain + '\n')
                    self.last_progress_global = None

                buf = b''

            # CR case
            # ~ if buf.endswith(b'\r'):
                # ~ raw = buf[:-1]
                # ~ line = strip_ansi(raw.decode('utf-8', errors='replace'))
                # ~ rd = self.stdout_rd if tag == 'stdout' else self.stderr_rd

                # ~ # Always overwrite on carriage returns
                # ~ rd.overwrite(line)
                # ~ buf = b''

            # NL case
            elif buf.endswith(b'\n'):
                raw = buf[:-1]
                line_plain = strip_ansi(raw.decode('utf-8', errors='replace'))
                rd = self.stdout_rd if tag == 'stdout' else self.stderr_rd
                line_stripped = line_plain.strip()

                if is_progress_line(line_plain):
                    # If we have a previous progress left by CR, replace it with this final progress line.
                    # Remove any last_insert regions left by either stream, then write the final line.
                    removed = False
                    for last_tag in ('last_insert_stdout', 'last_insert_stderr'):
                        ranges = self.output_area.tag_ranges(last_tag)
                        if ranges:
                            self.output_area.configure(state='normal')
                            # delete the exact region
                            self.output_area.delete(ranges[0], ranges[1])
                            # remove the tag so it doesn't linger
                            self.output_area.tag_remove(last_tag, "1.0", "end")
                            self.output_area.configure(state='disabled')
                            removed = True
                    # Now write the final progress line (with newline)
                    rd.write(line_plain + '\n')
                    # Clear remembered progress (we've shown the final one)
                    self.last_progress_global = None

                else:
                    # Not a progress line: if it exactly duplicates a remembered progress, skip it.
                    if self.last_progress_global is not None and line_stripped == self.last_progress_global:
                        # skip duplicate final progress line
                        self.last_progress_global = None
                    else:
                        rd.write(line_plain + '\n')
                        self.last_progress_global = None

                buf = b''

            # otherwise, keep buffering
            self.output_buffer[tag] = buf

        self.root.after(50, self.update_output)


if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptRunnerApp(root)
    root.mainloop()
