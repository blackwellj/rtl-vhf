import tkinter as tk
from tkinter import ttk, messagebox
from rtlsdr import RtlSdr
import numpy as np
import threading
from scipy.signal import butter, sosfilt, resample
import pyaudio

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
AUDIO_SAMPLE_RATE = 48000  # Standard audio sample rate for playback


class VHFListenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marine VHF Listener")
        self.root.geometry("1100x600")

        self.sdr = None
        self.running = False
        self.channels_active = {}
        self.squelch_levels = {}
        self.signal_levels = {}  # Store current signal levels (in dBm) for each channel
        self.audio_streams = {}
        self.stop_flags = {}

        self.setup_gui()

    def setup_gui(self):
        ttk.Label(self.root, text="Marine VHF Channels").grid(row=0, column=0, pady=10, padx=10, columnspan=6)

        for i, (channel_name, freq) in enumerate(VHF_CHANNELS.items()):
            button = ttk.Button(self.root, text=channel_name, command=lambda cn=channel_name: self.toggle_channel(cn))
            button.grid(row=1 + i, column=0, pady=5, padx=10, sticky="w")

            self.channels_active[channel_name] = tk.StringVar(value="Inactive")
            status_label = ttk.Label(self.root, textvariable=self.channels_active[channel_name])
            status_label.grid(row=1 + i, column=1, pady=5, padx=10, sticky="w")

            self.signal_levels[channel_name] = tk.StringVar(value="Signal: -∞ dBm")
            signal_label = ttk.Label(self.root, textvariable=self.signal_levels[channel_name])
            signal_label.grid(row=1 + i, column=2, pady=5, padx=10)

            self.squelch_levels[channel_name] = tk.DoubleVar(value=None)
            squelch_level_label = ttk.Label(self.root, textvariable=self.squelch_levels[channel_name])
            squelch_level_label.grid(row=1 + i, column=3, pady=5, padx=10)

            squelch_set_button = ttk.Button(self.root, text="Set Squelch",
                                             command=lambda ch=channel_name: self.set_squelch(ch))
            squelch_set_button.grid(row=1 + i, column=4, pady=5, padx=10)

            squelch_reset_button = ttk.Button(self.root, text="Reset Squelch",
                                               command=lambda ch=channel_name: self.reset_squelch(ch))
            squelch_reset_button.grid(row=1 + i, column=5, pady=5, padx=10)

        self.listen_button_text = tk.StringVar(value="Start Listening")
        ttk.Button(self.root, textvariable=self.listen_button_text, command=self.toggle_listening).grid(
            row=len(VHF_CHANNELS) + 2, column=0, pady=10, padx=10, columnspan=6
        )

    def toggle_channel(self, channel_name):
        if self.channels_active[channel_name].get() == "Active":
            self.channels_active[channel_name].set("Inactive")
            self.stop_flags[channel_name] = True
        else:
            self.channels_active[channel_name].set("Active")
            self.stop_flags[channel_name] = False

    def toggle_listening(self):
        if self.running:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        if self.sdr is not None:
            self.report_error("Already monitoring. Stop the current session before starting a new one.")
            return

        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = SAMPLE_RATE
            self.sdr.center_freq = self.calculate_center_frequency() * 1e6
            self.sdr.gain = 'auto'

            self.running = True
            self.listen_button_text.set("Stop Listening")
            threading.Thread(target=self.monitor_channels, daemon=True).start()
        except Exception as e:
            self.report_error(f"Failed to start SDR: {e}")

    def calculate_center_frequency(self):
        active_frequencies = [VHF_CHANNELS[ch] for ch, state in self.channels_active.items() if state.get() == "Active"]
        if not active_frequencies:
            self.report_error("No channels selected!")
            return 156.8
        return (min(active_frequencies) + max(active_frequencies)) / 2

    def set_squelch(self, channel_name):
        current_signal = self.signal_levels[channel_name].get().replace("Signal: ", "").replace(" dBm", "")
        if current_signal != "-∞":
            if self.squelch_levels[channel_name].get() is not None:
                self.squelch_levels[channel_name].set(self.squelch_levels[channel_name].get() + 5)
            else:
                self.squelch_levels[channel_name].set(float(current_signal))
        else:
            self.report_error(f"No valid signal level to set squelch for {channel_name}.")

    def reset_squelch(self, channel_name):
        self.squelch_levels[channel_name].set(None)

    def bandpass_filter(self, data, center_freq, fs):
        lowcut = max(center_freq - (CHANNEL_BANDWIDTH / 2), 1)
        highcut = center_freq + (CHANNEL_BANDWIDTH / 2)

        if highcut >= fs / 2 or highcut <= lowcut:
            raise ValueError(f"Invalid filter critical frequencies: lowcut={lowcut}, highcut={highcut}, fs={fs}")

        sos = butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band', output='sos')
        return sosfilt(sos, data)

    def demodulate_channel(self, samples, channel_freq, sdr_center_freq):
        relative_freq = channel_freq - sdr_center_freq

        if abs(relative_freq) < (CHANNEL_BANDWIDTH / 2):
            relative_freq = CHANNEL_BANDWIDTH / 2

        shift = np.exp(-2j * np.pi * relative_freq * np.arange(len(samples)) / SAMPLE_RATE)
        shifted = samples * shift
        filtered = self.bandpass_filter(shifted, center_freq=abs(relative_freq), fs=SAMPLE_RATE)
        demodulated = np.diff(np.unwrap(np.angle(filtered)))
        return demodulated

    def monitor_channels(self):
        pyaudio_instance = pyaudio.PyAudio()

        for channel_name in VHF_CHANNELS:
            self.audio_streams[channel_name] = pyaudio_instance.open(format=pyaudio.paFloat32,
                                                                     channels=1,
                                                                     rate=AUDIO_SAMPLE_RATE,
                                                                     output=True)
            self.stop_flags[channel_name] = False

        try:
            while self.running:
                samples = self.sdr.read_samples(256 * 1024)

                for channel_name, state in self.channels_active.items():
                    if state.get() == "Active" and not self.stop_flags[channel_name]:
                        channel_freq = VHF_CHANNELS[channel_name] * 1e6
                        center_freq = self.sdr.center_freq
                        demodulated_audio = self.demodulate_channel(samples, channel_freq, center_freq)

                        signal_power = 10 * np.log10(np.mean(np.abs(demodulated_audio) ** 2) + 1e-12)
                        self.signal_levels[channel_name].set(f"Signal: {signal_power:.2f} dBm")

                        squelch_level = self.squelch_levels[channel_name].get()
                        if squelch_level is None or signal_power > squelch_level:
                            downsampled_audio = resample(
                                demodulated_audio,
                                int(len(demodulated_audio) * AUDIO_SAMPLE_RATE / SAMPLE_RATE)
                            )
                            self.audio_streams[channel_name].write(downsampled_audio.astype(np.float32).tobytes())
        except Exception as e:
            self.report_error(f"Error while monitoring channels: {e}")
        finally:
            self.stop_listening()
            for stream in self.audio_streams.values():
                stream.close()
            pyaudio_instance.terminate()

    def stop_listening(self):
        self.running = False
        self.listen_button_text.set("Start Listening")
        if self.sdr is not None:
            self.sdr.close()
            self.sdr = None

    def report_error(self, message):
        print(f"Error: {message}")
        messagebox.showerror("Error", message)


def main():
    root = tk.Tk()
    app = VHFListenerApp(root)
    root.protocol("WMDELETE_WINDOW", app.stop_listening)
    root.mainloop()


if __name__ == "__main__":
    main()
