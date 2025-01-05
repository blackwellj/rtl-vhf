import tkinter as tk
from tkinter import ttk, messagebox
from rtlsdr import RtlSdr
import numpy as np
import threading
from scipy.signal import butter, sosfilt

# Full UK Marine VHF Channel Frequencies (MHz)
VHF_CHANNELS = {
    "Channel 0 (156.000 MHz)": 156.000,
    "Channel 6 (156.300 MHz)": 156.300,
    "Channel 16 (156.800 MHz)": 156.800,  # Distress channel
    "Channel 67 (156.375 MHz)": 156.375,
    "Channel 99 (157.550 MHz)": 157.550,  # Search and Rescue
    "Lifeguard L1 (161.425 MHz)": 161.425,  # RNLI Lifeguards
}

CHANNEL_BANDWIDTH = 12.5e3  # 12.5 kHz for NFM
SAMPLE_RATE = 2.048e6  # RTL-SDR sample rate

class VHFListenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marine VHF Listener")
        self.root.geometry("600x400")

        self.sdr = None
        self.running = False
        self.channels_active = {}

        self.setup_gui()

    def setup_gui(self):
        ttk.Label(self.root, text="Marine VHF Channels").grid(row=0, column=0, pady=10, padx=10, columnspan=2)
        
        for i, (channel_name, freq) in enumerate(VHF_CHANNELS.items()):
            button = ttk.Button(self.root, text=channel_name, command=lambda cn=channel_name: self.toggle_channel(cn))
            button.grid(row=1 + i, column=0, pady=5, padx=10, sticky="w")
            
            self.channels_active[channel_name] = tk.StringVar(value="Inactive")
            status_label = ttk.Label(self.root, textvariable=self.channels_active[channel_name])
            status_label.grid(row=1 + i, column=1, pady=5, padx=10, sticky="w")

        ttk.Button(self.root, text="Start Listening", command=self.start_listening).grid(row=len(VHF_CHANNELS) + 2, column=0, pady=10, padx=10, columnspan=2)

    def toggle_channel(self, channel_name):
        if self.channels_active[channel_name].get() == "Active":
            self.channels_active[channel_name].set("Inactive")
        else:
            self.channels_active[channel_name].set("Active")

    def start_listening(self):
        if self.sdr is not None:
            messagebox.showinfo("Info", "Already monitoring. Stop the current session before starting a new one.")
            return

        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = SAMPLE_RATE
            self.sdr.center_freq = self.calculate_center_frequency() * 1e6
            self.sdr.gain = 'auto'

            self.running = True
            threading.Thread(target=self.monitor_channels, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start SDR: {e}")

    def calculate_center_frequency(self):
        active_frequencies = [VHF_CHANNELS[ch] for ch, state in self.channels_active.items() if state.get() == "Active"]
        if not active_frequencies:
            messagebox.showerror("Error", "No channels selected!")
            return 156.8  # Default to Channel 16
        return (min(active_frequencies) + max(active_frequencies)) / 2

    def bandpass_filter(self, data, center_freq, fs):
        lowcut = center_freq - (CHANNEL_BANDWIDTH / 2)
        highcut = center_freq + (CHANNEL_BANDWIDTH / 2)
        if lowcut < 0 or highcut >= fs / 2 or highcut <= lowcut:
            raise ValueError(f"Invalid filter critical frequencies: lowcut={lowcut}, highcut={highcut}, fs={fs}")
        sos = butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band', output='sos')
        return sosfilt(sos, data)

    def demodulate_channel(self, samples, channel_freq, center_freq):
        shift = np.exp(-2j * np.pi * (channel_freq - center_freq) * np.arange(len(samples)) / SAMPLE_RATE)
        shifted = samples * shift
        filtered = self.bandpass_filter(shifted, center_freq=0, fs=SAMPLE_RATE)
        demodulated = np.diff(np.unwrap(np.angle(filtered)))
        return demodulated

    def monitor_channels(self):
        try:
            while self.running:
                samples = self.sdr.read_samples(256 * 1024)
                for channel_name, state in self.channels_active.items():
                    if state.get() == "Active":
                        channel_freq = VHF_CHANNELS[channel_name] * 1e6
                        center_freq = self.sdr.center_freq
                        demodulated_audio = self.demodulate_channel(samples, channel_freq, center_freq)
                        print(f"Channel {channel_name}: Processed {len(demodulated_audio)} samples")
        except Exception as e:
            messagebox.showerror("Error", f"Error while monitoring channels: {e}")
        finally:
            self.stop_listening()

    def stop_listening(self):
        self.running = False
        if self.sdr is not None:
            self.sdr.close()
            self.sdr = None

def main():
    root = tk.Tk()
    app = VHFListenerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_listening)
    root.mainloop()

if __name__ == "__main__":
    main()
