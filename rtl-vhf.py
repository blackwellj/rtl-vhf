import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QTextEdit
)
from pyrtlsdr import RtlSdr
import pyaudio

# Marine VHF frequencies (MHz)
MARINE_VHF_CHANNELS = {
    "Ch 16 (Distress)": 156.8,
    "Ch 06": 156.3,
    "Ch 08": 156.4,
    "Ch 09": 156.45,
    "Ch 10": 156.5,
    "Ch 11": 156.55,
}

class MarineVHFApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marine VHF Listener")
        self.setGeometry(100, 100, 600, 400)
        
        # SDR and Audio Setup
        self.sdr = None
        self.audio_stream = None
        
        # GUI Layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        self.channel_selector = QComboBox()
        for channel_name in MARINE_VHF_CHANNELS.keys():
            self.channel_selector.addItem(channel_name)
        self.layout.addWidget(QLabel("Select Channel:"))
        self.layout.addWidget(self.channel_selector)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(QLabel("Logs:"))
        self.layout.addWidget(self.log_area)
        
        self.start_button = QPushButton("Start Listening")
        self.start_button.clicked.connect(self.start_listening)
        self.layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Listening")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_listening)
        self.layout.addWidget(self.stop_button)
    
    def start_listening(self):
        selected_channel = self.channel_selector.currentText()
        frequency = MARINE_VHF_CHANNELS[selected_channel] * 1e6  # Convert MHz to Hz
        self.log_area.append(f"Starting SDR on {selected_channel} ({frequency/1e6} MHz)...")
        
        try:
            # Setup SDR
            self.sdr = RtlSdr()
            self.sdr.sample_rate = 2.048e6  # Hz
            self.sdr.center_freq = frequency  # Hz
            self.sdr.gain = 40  # Adjust as needed
            
            # Setup Audio
            self.audio_stream = pyaudio.PyAudio().open(
                format=pyaudio.paInt16,
                channels=1,
                rate=int(self.sdr.sample_rate),
                output=True,
            )
            
            self.stop_button.setEnabled(True)
            self.start_button.setEnabled(False)
            
            # Start Receiving
            self.receive_samples()
        except Exception as e:
            self.log_area.append(f"Error: {e}")
    
    def receive_samples(self):
        try:
            for samples in self.sdr.stream(num_samples=2048):
                # Process samples (e.g., demodulate FM, apply filters)
                processed_audio = np.int16(samples * 32767)  # Example processing
                self.audio_stream.write(processed_audio.tobytes())
        except Exception as e:
            self.log_area.append(f"Stream stopped: {e}")
    
    def stop_listening(self):
        self.log_area.append("Stopping SDR...")
        try:
            if self.sdr:
                self.sdr.close()
            if self.audio_stream:
                self.audio_stream.close()
        except Exception as e:
            self.log_area.append(f"Error while stopping: {e}")
        finally:
            self.sdr = None
            self.audio_stream = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.log_area.append("SDR stopped.")
    
    def closeEvent(self, event):
        self.stop_listening()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MarineVHFApp()
    window.show()
    sys.exit(app.exec_())
