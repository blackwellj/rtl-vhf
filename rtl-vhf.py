import tkinter as tk
from tkinter import ttk, messagebox
from rtlsdr import RtlSdr
import pyaudio
import numpy as np
import threading
from scipy.signal import decimate

# Marine VHF Channel Frequencies (MHz)
VHF_CHANNELS = {
    "Channel 0 (156.000 MHz)": 156.000,
    "Channel 1 (156.050 MHz)": 156.050,
    "Channel 2 (156.100 MHz)": 156.100,
    "Channel 3 (156.150 MHz)": 156.150,
    "Channel 4 (156.200 MHz)": 156.200,
    "Channel 5 (156.250 MHz)": 156.250,
    "Channel 6 (156.300 MHz)": 156.300,
    "Channel 7 (156.350 MHz)": 156.350,
    "Channel 8 (156.400 MHz)": 156.400,
    "Channel 9 (156.450 MHz)": 156.450,
    "Channel 10 (156.500 MHz)": 156.500,
    "Channel 11 (156.550 MHz)": 156.550,
    "Channel 12 (156.600 MHz)": 156.600,
    "Channel 13 (156.650 MHz)": 156.650,
    "Channel 14 (156.700 MHz)": 156.700,
    "Channel 15 (156.750 MHz)": 156.750,
    "Channel 16 (156.800 MHz)": 156.800,  # Distress channel
    "Channel 17 (156.850 MHz)": 156.850,
    "Channel 18 (156.900 MHz)": 156.900,
    "Channel 19 (156.950 MHz)": 156.950,
    "Channel 20 (157.000 MHz)": 157.000,
    "Channel 21 (157.050 MHz)": 157.050,
    "Channel 22 (157.100 MHz)": 157.100,
    "Channel 23 (157.150 MHz)": 157.150,
    "Channel 24 (157.200 MHz)": 157.200,
    "Channel 25 (157.250 MHz)": 157.250,
    "Channel 26 (157.300 MHz)": 157.300,
    "Channel 27 (157.350 MHz)": 157.350,
    "Channel 28 (157.400 MHz)": 157.400,
    "Channel 60 (156.025 MHz)": 156.025,
    "Channel 61 (156.075 MHz)": 156.075,
    "Channel 62 (156.125 MHz)": 156.125,
    "Channel 63 (156.175 MHz)": 156.175,
    "Channel 64 (156.225 MHz)": 156.225,
    "Channel 65 (156.275 MHz)": 156.275,
    "Channel 66 (156.325 MHz)": 156.325,
    "Channel 67 (156.375 MHz)": 156.375,
    "Channel 68 (156.425 MHz)": 156.425,
    "Channel 69 (156.475 MHz)": 156.475,
    "Channel 70 (156.525 MHz)": 156.525,  # Digital Selective Calling (DSC)
    "Channel 71 (156.575 MHz)": 156.575,
    "Channel 72 (156.625 MHz)": 156.625,
    "Channel 73 (156.675 MHz)": 156.675,
    "Channel 74 (156.725 MHz)": 156.725,
    "Channel 75 (156.775 MHz)": 156.775,
    "Channel 76 (156.825 MHz)": 156.825,
    "Channel 77 (156.875 MHz)": 156.875,
    "Channel 78 (156.925 MHz)": 156.925,
    "Channel 79 (156.975 MHz)": 156.975,
    "Channel 80 (157.025 MHz)": 157.025,
    "Channel 81 (157.075 MHz)": 157.075,
    "Channel 82 (157.125 MHz)": 157.125,
    "Channel 83 (157.175 MHz)": 157.175,
    "Channel 84 (157.225 MHz)": 157.225,
    "Channel 85 (157.275 MHz)": 157.275,
    "Channel 86 (157.325 MHz)": 157.325,
    "Channel 87 (157.375 MHz)": 157.375,
    "Channel 88 (157.425 MHz)": 157.425,
    "Channel 99 (157.550 MHz)": 157.550,  # Search and Rescue
    "Lifeguard L1 (161.425 MHz)": 161.425,  # RNLI Lifeguards
}


class VHFListenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marine VHF Listener")
        self.root.geometry("400x300")

        self.sdr = None
        self.audio_stream = None
        self.running = False
        self.audio_device_index = None  # Default to None (use default device)

        # PyAudio Instance
        self.p = pyaudio.PyAudio()

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

        self.settings_btn = ttk.Button(self.root, text="Audio Settings", command=self.audio_settings)
        self.settings_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def audio_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Audio Settings")
        settings_window.geometry("400x300")

        ttk.Label(settings_window, text="Select Audio Output Device:").pack(pady=10)
        device_list = [self.p.get_device_info_by_index(i)['name'] for i in range(self.p.get_device_count())]
        self.device_var = tk.StringVar(value=device_list[0] if device_list else "Default")
        device_menu = ttk.Combobox(settings_window, textvariable=self.device_var, values=device_list, width=50)
        device_menu.pack(pady=10)

        def save_settings():
            selected_device_name = self.device_var.get()
            for i in range(self.p.get_device_count()):
                if self.p.get_device_info_by_index(i)['name'] == selected_device_name:
                    self.audio_device_index = i
                    messagebox.showinfo("Settings", f"Audio device set to: {selected_device_name}")
                    settings_window.destroy()
                    return
            messagebox.showerror("Settings", "Selected audio device not found.")

        save_btn = ttk.Button(settings_window, text="Save", command=save_settings)
        save_btn.pack(pady=20)

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
            audio_rate = 48000  # Output sample rate
            decimation_factor = int(self.sdr.sample_rate // audio_rate)

            self.audio_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=audio_rate,
                output=True,
                output_device_index=self.audio_device_index  # Use selected or default device
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
