import os
import tkinter as tk
from tkinter import ttk, messagebox
from rtlsdr import RtlSdr
import pyaudio
import numpy as np
import threading

class VHFListenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marine VHF Listener")
        self.root.geometry("400x300")
        
        self.sdr = None
        self.audio_stream = None
        self.running = False
        
        # GUI Elements
        self.setup_gui()

    def setup_gui(self):
        ttk.Label(self.root, text="Center Frequency (MHz):").grid(row=0, column=0, pady=10, padx=10)
        self.freq_entry = ttk.Entry(self.root)
        self.freq_entry.grid(row=0, column=1, pady=10, padx=10)

        ttk.Label(self.root, text="Sample Rate (MS/s):").grid(row=1, column=0, pady=10, padx=10)
        self.sr_entry = ttk.Entry(self.root)
        self.sr_entry.insert(0, "2.048")
        self.sr_entry.grid(row=1, column=1, pady=10, padx=10)

        self.start_btn = ttk.Button(self.root, text="Start Listening", command=self.start_listening)
        self.start_btn.grid(row=2, column=0, columnspan=2, pady=10)

        self.stop_btn = ttk.Button(self.root, text="Stop Listening", command=self.stop_listening, state=tk.DISABLED)
        self.stop_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def start_listening(self):
        try:
            frequency = float(self.freq_entry.get()) * 1e6
            sample_rate = float(self.sr_entry.get()) * 1e6
            self.sdr = RtlSdr()
            self.sdr.sample_rate = sample_rate
            self.sdr.center_freq = frequency
            self.sdr.gain = 'auto'

            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            threading.Thread(target=self.stream_audio, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start SDR: {e}")

    def stop_listening(self):
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.sdr:
            self.sdr.close()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def stream_audio(self):
        try:
            p = pyaudio.PyAudio()
            self.audio_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=int(self.sdr.sample_rate),
                output=True
            )

            while self.running:
                samples = self.sdr.read_samples(256 * 1024)
                audio_data = np.real(samples).astype(np.int16).tobytes()
                self.audio_stream.write(audio_data)
        except Exception as e:
            messagebox.showerror("Error", f"Audio streaming error: {e}")
        finally:
            self.stop_listening()

def main():
    root = tk.Tk()
    app = VHFListenerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_listening)
    root.mainloop()

if __name__ == "__main__":
    main()
