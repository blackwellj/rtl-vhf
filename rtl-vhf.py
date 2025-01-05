import tkinter as tk
from tkinter import ttk, messagebox
from rtlsdr import RtlSdr
import pyaudio
import numpy as np
import threading
from scipy.signal import decimate

# Marine VHF Channel Frequencies (MHz)
VHF_CHANNELS = {
    "Channel 16 (156.8 MHz)": 156.8,
    "Channel 06 (156.3 MHz)": 156.3,
    "Channel 09 (156.45 MHz)": 156.45,
    "Channel 10 (156.5 MHz)": 156.5,
    "Channel 12 (156.6 MHz)": 156.6,
    "Channel 14 (156.7 MHz)": 156.7,
    "Channel 67 (156.375 MHz)": 156.375,
    # Add more channels as needed
}

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
        ttk.Label(self.root, text="Select VHF Channel:").grid(row=0, column=0, pady=10, padx=10)
        self.channel_var = tk.StringVar()
        self.channel_var.set("Channel 16 (156.8 MHz)")
        self.channel_menu = ttk.Combobox(self.root, textvariable=self.channel_var, values=list(VHF_CHANNELS.keys()))
        self.channel_menu.grid(row=0, column=1, pady=10, padx=10)

        self.start_btn = ttk.Button(self.root, text="Start Listening", command=self.start_listening)
        self.start_btn.grid(row=1, column=0, columnspan=2, pady=10)

        self.stop_btn = ttk.Button(self.root, text="Stop Listening", command=self.stop_listening, state=tk.DISABLED)
        self.stop_btn.grid(row=2, column=0, columnspan=2, pady=10)

    def start_listening(self):
        try:
            selected_channel = self.channel_var.get()
            frequency = VHF_CHANNELS[selected_channel] * 1e6  # Convert MHz to Hz

            self.sdr = RtlSdr()
            self.sdr.sample_rate = 2.048e6  # RTL-SDR's supported sample rate
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
            audio_rate = 48000  # Output sample rate
            decimation_factor = int(self.sdr.sample_rate // audio_rate)

            self.audio_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=audio_rate,
                output=True
            )

            while self.running:
                samples = self.sdr.read_samples(256 * 1024)
                # Perform decimation to reduce sample rate
                filtered_samples = decimate(samples, decimation_factor, zero_phase=True)
                # Convert to int16 format for audio playback
                audio_data = np.real(filtered_samples).astype(np.int16).tobytes()
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
