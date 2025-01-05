import tkinter as tk
from tkinter import ttk, messagebox
from rtlsdr import RtlSdr
import pyaudio
import numpy as np
import threading
from scipy.signal import decimate

# Full UK Marine VHF Channel Frequencies (MHz)
VHF_CHANNELS = {
    "Channel 0 (156.000 MHz)": 156.000,
    "Channel 6 (156.300 MHz)": 156.300,
    "Channel 16 (156.800 MHz)": 156.800,  # Distress channel
    "Channel 67 (156.375 MHz)": 156.375,
    "Channel 99 (157.550 MHz)": 157.550,  # Search and Rescue
    "Lifeguard L1 (161.425 MHz)": 161.425,  # RNLI Lifeguards
}

class VHFListenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marine VHF Listener")
        self.root.geometry("600x400")

        self.sdrs = {}
        self.running = {}
        self.audio_streams = {}
        self.signal_levels = {}

        self.audio_device_index = None  # Default to None (use system's default device)
        self.p = pyaudio.PyAudio()

        self.setup_gui()

    def setup_gui(self):
        ttk.Label(self.root, text="Marine VHF Channels").grid(row=0, column=0, pady=10, padx=10, columnspan=2)
        
        # Create buttons for each channel
        for i, (channel_name, freq) in enumerate(VHF_CHANNELS.items()):
            button = ttk.Button(self.root, text=channel_name, command=lambda cn=channel_name, f=freq: self.toggle_channel(cn, f))
            button.grid(row=1 + i, column=0, pady=5, padx=10, sticky="w")
            
            # Signal strength indicator
            self.signal_levels[channel_name] = tk.StringVar(value="Signal: ---")
            signal_label = ttk.Label(self.root, textvariable=self.signal_levels[channel_name])
            signal_label.grid(row=1 + i, column=1, pady=5, padx=10, sticky="w")

        ttk.Button(self.root, text="Audio Settings", command=self.audio_settings).grid(row=len(VHF_CHANNELS) + 2, column=0, pady=10, padx=10, columnspan=2)

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

    def toggle_channel(self, channel_name, frequency):
        if channel_name in self.running and self.running[channel_name]:
            self.stop_channel(channel_name)
        else:
            self.start_channel(channel_name, frequency)

    def start_channel(self, channel_name, frequency):
        try:
            sdr = RtlSdr()
            sdr.sample_rate = 2.048e6
            sdr.center_freq = frequency * 1e6  # Convert MHz to Hz
            sdr.gain = 'auto'

            self.sdrs[channel_name] = sdr
            self.running[channel_name] = True

            threading.Thread(target=self.stream_audio, args=(channel_name,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start SDR for {channel_name}: {e}")

    def stop_channel(self, channel_name):
        self.running[channel_name] = False
        if channel_name in self.audio_streams:
            self.audio_streams[channel_name].stop_stream()
            self.audio_streams[channel_name].close()
            del self.audio_streams[channel_name]
        if channel_name in self.sdrs:
            self.sdrs[channel_name].close()
            del self.sdrs[channel_name]
        self.signal_levels[channel_name].set("Signal: ---")

    def stream_audio(self, channel_name):
        try:
            audio_rate = 48000
            sdr = self.sdrs[channel_name]
            decimation_factor = int(sdr.sample_rate // audio_rate)

            audio_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=audio_rate,
                output=True,
                output_device_index=self.audio_device_index
            )
            self.audio_streams[channel_name] = audio_stream

            while self.running[channel_name]:
                samples = sdr.read_samples(256 * 1024)
                # Calculate signal level
                signal_level = 10 * np.log10(np.mean(np.abs(samples)**2))
                self.signal_levels[channel_name].set(f"Signal: {signal_level:.1f} dB")
                # Downsample for playback (use decimate or slicing as fallback)
                try:
                    filtered_samples = decimate(samples, decimation_factor, zero_phase=True)
                except Exception:
                    filtered_samples = samples[::decimation_factor]  # Simple downsampling
                audio_data = np.real(filtered_samples).astype(np.int16).tobytes()
                audio_stream.write(audio_data)
        except Exception as e:
            messagebox.showerror("Error", f"Audio streaming error for {channel_name}: {e}")
        finally:
            self.stop_channel(channel_name)

def main():
    root = tk.Tk()
    app = VHFListenerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.stop_channel(ch) for ch in app.running if app.running[ch]])
    root.mainloop()

if __name__ == "__main__":
    main()
