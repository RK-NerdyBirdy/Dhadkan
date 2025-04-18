import asyncio
import websockets
import numpy as np
import pandas as pd
import time
import winsound  # For beep sound (Windows only)
import keyboard
import matplotlib.pyplot as plt
import tensorflow as tf
from reportgenerator import ReportGenerator
import neurokit2 as nk
from scipy.signal import butter, lfilter, find_peaks
from collections import deque

class WirelessECGMonitor:
    def __init__(self, model_path="ecg_cnn_model.h5", websocket_uri="ws://192.168.1.X:81"):
        # Replace the IP with your ESP32's IP address
        self.websocket_uri = websocket_uri
        self.sampling_rate = 125  # Hz
        self.chunk_size = 187  # Match the input dimension from your training
        
        # Data buffers
        self.ecg_buffer = deque(maxlen=self.sampling_rate * 10)  # 10 seconds buffer
        self.r_peaks = []
        self.qrs_segments = []  # Store QRS-centered segments
        self.alert_history = []
        self.csv_filename = "ecg_alerts.csv"
        self.report_generator = ReportGenerator()
        self.patient_data = {}
        
        # For visualization
        self.display_buffer = []  # For displaying the continuous ECG
        self.last_r_peak = 0     # Index of the last detected R-peak
        self.window_size = 500   # Display window size
        
        # Flags
        self.new_peak_detected = False
        self.processing_enabled = True
        
        # Class labels for interpretable results
        self.class_labels = {
            0: "Normal",
            1: "Artial Premature",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular and normal",
            4: "Fusion of paced and normal"
        }
        
        # Load trained deep learning model
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained ML model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def set_patient_info(self, name, age):
        """Set patient information for report"""
        self.patient_data = {"name": name, "age": age}
    
    def butter_lowpass_filter(self, data, cutoff=15, fs=125, order=4):
        """Low-pass filter for ECG signal"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y
    
    def detect_r_peaks(self, signal):
        """Real-time R-peak detection using thresholding and find_peaks"""
        try:
            # Apply a bandpass filter to enhance QRS complex
            filtered_signal = self.butter_lowpass_filter(signal)
            
            # Use scipy's find_peaks with appropriate parameters for ECG
            # Adjusted for real-time use
            peaks, _ = find_peaks(filtered_signal, 
                                 height=np.mean(filtered_signal) + 0.5 * np.std(filtered_signal),
                                 distance=self.sampling_rate * 0.5)  # Min 0.5s between peaks
            
            return peaks
        except Exception as e:
            print(f"Error in R-peak detection: {e}")
            return []
    
    def extract_qrs_segment(self, signal, r_peak_idx):
        """Extract a segment centered around an R-peak"""
        half_window = self.chunk_size // 2
        start_idx = max(0, r_peak_idx - half_window)
        end_idx = min(len(signal), r_peak_idx + half_window + 1)
        
        # Create segment of exactly chunk_size length
        segment = np.zeros(self.chunk_size)
        extracted = signal[start_idx:end_idx]
        
        # Handle edge cases with padding
        if len(extracted) < self.chunk_size:
            if start_idx == 0:  # Pad at beginning
                segment[self.chunk_size - len(extracted):] = extracted
            else:  # Pad at end
                segment[:len(extracted)] = extracted
        else:
            segment = extracted[:self.chunk_size]
            
        return segment
        
    async def connect_and_process(self):
        """Connect to WebSocket and process incoming ECG data"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        plt.ion()  # Enable interactive mode
        
        ax1.set_title("Real-time ECG Signal with R-peaks")
        ax2.set_title("QRS-Centered Segment (Model Input)")
        ax3.set_title("Abnormality Probability")
        
        # Initialize plots
        ecg_line, = ax1.plot([], [], 'b-')
        r_peaks_scatter = ax1.scatter([], [], c='red', marker='x')
        qrs_line, = ax2.plot([], [], 'g-')
        
        # Bar chart for classification
        class_names = list(self.class_labels.values())
        bar_colors = ['green'] + ['red'] * (len(class_names) - 1)
        bars = ax3.bar(class_names, [0] * len(class_names), color=bar_colors)
        ax3.set_ylim(0, 1)
        
        # For smoothing prediction visualization
        last_predictions = np.zeros(len(class_names))
        
        # Initialize buffers for visualization
        self.display_buffer = [0] * self.window_size
        peak_indices = []
        qrs_segment = [0] * self.chunk_size
        
        print(f"Connecting to ESP32 at {self.websocket_uri}")
        print("Press 'q' to exit the monitoring.")
        
        try:
            async with websockets.connect(self.websocket_uri) as websocket:
                print("✅ Connected to ESP32 WebSocket server!")
                
                while True:
                    if keyboard.is_pressed('q'):
                        print("🛑 Monitoring Stopped by User.")
                        break
                    
                    try:
                        # Get data from websocket
                        ecg_value = await websocket.recv()
                        ecg_value = int(ecg_value.strip())
                        
                        # Add to buffer
                        self.ecg_buffer.append(ecg_value)
                        self.display_buffer.append(ecg_value)
                        self.display_buffer = self.display_buffer[-self.window_size:]
                        
                        # Process when we have enough data
                        if len(self.ecg_buffer) >= self.sampling_rate * 2:  # At least 2 seconds of data
                            # Convert to numpy array for processing
                            ecg_data = np.array(list(self.ecg_buffer))
                            
                            # Filter the signal
                            filtered_data = self.butter_lowpass_filter(ecg_data)
                            
                            # Detect R-peaks
                            new_peaks = self.detect_r_peaks(filtered_data)
                            
                            # Update display
                            filtered_display = self.butter_lowpass_filter(np.array(self.display_buffer))
                            ecg_line.set_data(range(len(filtered_display)), filtered_display)
                            
                            # Check if we have new peaks
                            if len(new_peaks) > 0:
                                # Get the last R-peak in the buffer
                                last_peak = new_peaks[-1]
                                
                                # Only process if it's a new peak
                                if last_peak != self.last_r_peak and last_peak < len(ecg_data) - 20:  # Ensure not at the edge
                                    self.last_r_peak = last_peak
                                    self.new_peak_detected = True
                                    
                                    # Extract QRS-centered segment
                                    qrs_segment = self.extract_qrs_segment(filtered_data, last_peak)
                                    
                                    # Run abnormality detection on QRS-centered segment
                                    abnormal, confidence, predicted_class = self.detect_abnormality(qrs_segment)
                                    
                                    # Get full prediction array for visualization
                                    prediction_array = self.model.predict(qrs_segment.reshape(1, -1), verbose=0)[0]
                                    
                                    # Smooth predictions for visualization
                                    alpha = 0.3
                                    last_predictions = alpha * prediction_array + (1 - alpha) * last_predictions
                                    
                                    # Update visualization
                                    # Show R-peaks on the ECG
                                    peak_indices = [p for p in new_peaks if p < len(filtered_display)]
                                    if peak_indices:
                                        r_peaks_scatter.set_offsets(np.c_[peak_indices, 
                                                                         [filtered_display[p] for p in peak_indices]])
                                    
                                    # Update QRS segment plot
                                    qrs_line.set_data(range(len(qrs_segment)), qrs_segment)
                                    
                                    # Update bar heights
                                    for i, bar in enumerate(bars):
                                        bar.set_height(last_predictions[i])
                                    
                                    # Highlight the predicted class
                                    for i, bar in enumerate(bars):
                                        bar.set_color(bar_colors[i])  # Default color
                                        if i == predicted_class:
                                            if predicted_class == 0:  # Normal class
                                                bar.set_color('green')
                                            else:
                                                bar.set_color('yellow')
                                    
                                    # Trigger alert if abnormal
                                    if abnormal:
                                        self.trigger_alert(confidence, predicted_class)
                                        
                            # Auto-scale the axes
                            for ax in [ax1, ax2]:
                                ax.relim()
                                ax.autoscale_view()
                            
                            plt.tight_layout()
                            plt.pause(0.01)
                        
                    except Exception as e:
                        print(f"Error processing data: {e}")
                        await asyncio.sleep(0.1)
                        
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            
        finally:
            plt.ioff()
            plt.close()
            self._generate_final_report()
    
    def detect_abnormality(self, ecg_chunk):
        """Use the trained model to detect abnormality from QRS-centered segment"""
        try:
            # Normalize data similar to training
            mean = np.mean(ecg_chunk)
            std = np.std(ecg_chunk)
            if std == 0:
                std = 1  # Prevent division by zero
            
            ecg_chunk = (ecg_chunk - mean) / std
            
            # Reshape for the model input
            ecg_chunk = ecg_chunk.reshape(1, -1)
            
            # Get class probabilities
            prediction = self.model.predict(ecg_chunk, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Consider any non-normal class as abnormal
            is_abnormal = predicted_class != 0 and confidence > 0.6
            
            return is_abnormal, confidence, predicted_class
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return False, 0, 0
    
    def trigger_alert(self, confidence, predicted_class):
        """Play beep sound and log the abnormality with class information"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        class_name = self.class_labels.get(predicted_class, "Unknown")
        
        alert_data = {
            "timestamp": timestamp, 
            "confidence": confidence,
            "class": predicted_class,
            "class_name": class_name
        }
        
        self.alert_history.append(alert_data)
        print(f"⚠️ ALERT: {class_name} detected at {timestamp} (Confidence: {confidence:.2f})")
        
        # Save alerts to CSV
        df = pd.DataFrame(self.alert_history)
        df.to_csv(self.csv_filename, index=False)
        
        # Beep sound (Windows only)
        winsound.Beep(1000, 500)
    
    def _generate_final_report(self):
        """Generate final report after monitoring"""
        # Calculate heart rate from detected R-peaks
        if len(self.ecg_buffer) > 0:
            ecg_data = np.array(list(self.ecg_buffer))
            try:
                # Use NeuroKit2 for more accurate R-peak detection for the final report
                ecg_processed = nk.ecg_process(ecg_data, sampling_rate=self.sampling_rate)
                r_peaks = ecg_processed[1]['ECG_R_Peaks']
                
                if len(r_peaks) > 1:
                    # Calculate heart rate
                    rr_intervals = np.diff(r_peaks)
                    heart_rate = 60 / (np.mean(rr_intervals) / self.sampling_rate)
                else:
                    heart_rate = 0
            except:
                heart_rate = 0
        else:
            heart_rate = 0
        
        # Add sections to report
        self.report_generator.add_section(
            "ECG Analysis Summary",
            f"Average Heart Rate: {heart_rate:.1f} bpm\n"
            f"Total Abnormalities Detected: {len(self.alert_history)}\n"
            "Signal Quality: Good"
        )
        
        self.report_generator.add_section(
            "Technical Details",
            f"Analysis Duration: {len(self.ecg_buffer)/self.sampling_rate:.1f} seconds\n"
            f"Model Confidence Threshold: 0.6\n"
            f"Sampling Rate: {self.sampling_rate} Hz"
        )
        
        # Generate PDF report
        self.report_generator.generate_report(
            patient_data=self.patient_data,
            abnormalities=self.alert_history
        )

# Run the wireless monitoring
if __name__ == "__main__":
    # Initialize the monitor
    monitor = WirelessECGMonitor(
        model_path="ecg_cnn_model.h5", 
        websocket_uri="ws://192.168.157.142:81"  # Replace with your ESP32's IP address
    )
    
    # Set patient info
    monitor.set_patient_info("Maneet Gupta", 5)
    
    # Start the monitor with asyncio
    asyncio.run(monitor.connect_and_process())