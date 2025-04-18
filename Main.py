import asyncio
import websockets
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter, filtfilt, sosfilt
import os
import datetime
import json
import winsound  # For beep sound on Windows
import sys  # For quitting with 'q'
import keyboard  # For keyboard input detection
from reportgenerator import ReportGenerator  # Import your report generator

class ECGMonitor:
    def __init__(self, model_path="ecg_cnn_model.h5", websocket_uri="ws://192.168.1.100:81"):
        # Configuration
        self.websocket_uri = websocket_uri
        self.sampling_rate = 125  # Hz
        self.chunk_size = 187  # Input dimension for CNN model
        
        # Data buffers
        self.raw_buffer = []
        self.filtered_buffer = []
        self.visualization_buffer_size = 1000  # ~8 seconds at 125Hz
        
        # For visualization
        self.fig = None
        self.axes = None
        self.line_raw = None
        self.line_filtered = None
        self.bars = None
        self.r_peak_scatter = None
        self.last_plot_time = time.time()
        self.plot_update_interval = 0.1  # seconds
        
        # Peak detection
        self.last_r_peaks = []
        self.r_peak_timestamps = []
        self.r_peak_buffer = []  # Store indices for visualization
        self.last_peak_sound_time = 0  # To prevent too frequent beeps
        self.sound_cooldown = 0.3  # Minimum seconds between beeps
        
        # Results tracking
        self.predictions_history = []
        self.class_labels = {
            0: "Normal Sinus Rhythm",
            1: "Atrial Premature",
            2: "Premature Ventricular Contraction",
            3: "Fusion Beat",
            4: "Unknown Abnormality"
        }
        self.last_prediction = np.zeros(len(self.class_labels))
        
        # Signal quality metrics
        self.signal_quality = "Unknown"
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Condition tracking for persistent logging
        self.last_condition = None
        self.condition_start_time = None
        self.condition_logging_delay = 3.0  # seconds - only log if condition persists > 3 seconds
        
        # BPM tracking and debug
        self.last_valid_hr = 0
        self.hr_timeout = 10  # seconds - how long a heart rate value remains valid
        self.last_hr_time = 0
        
        # Report generator
        self.report_generator = ReportGenerator()
        self.abnormalities = []  # List to store detected abnormalities for reporting
        
        # Try to load the trained model
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Creating a placeholder model instead")
            self.model = self.create_dummy_model()
        
        # For logging
        self.log_dir = "ecg_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"ecg_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.prepare_log_file()
        
        # For debug logs
        self.debug_log_file = os.path.join(self.log_dir, f"ecg_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create directory for reports
        os.makedirs("Report Generated", exist_ok=True)
        
        # Setup keyboard listener thread for quitting
        keyboard.on_press_key("q", self.on_quit_key_pressed)
        self.is_running = True
        
    def on_quit_key_pressed(self, e):
        """Handle the 'q' key press to quit the program"""
        print("\nQuitting application...")
        self.is_running = False
        self.generate_final_report()
        sys.exit(0)
        
    def log_debug(self, message):
        """Log debug messages to file"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.debug_log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        
    def create_dummy_model(self):
        """Create a dummy model for testing if the real model can't be loaded"""
        class DummyModel:
            def predict(self, x, verbose=0):
                # Return random predictions for testing
                batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 1 else 1
                return np.random.rand(batch_size, 5)
        return DummyModel()
    
    def prepare_log_file(self):
        """Prepare the CSV log file with headers"""
        headers = ["timestamp", "prediction_class", "class_name", "confidence", "heart_rate", "signal_quality"]
        with open(self.log_file, 'w') as f:
            f.write(','.join(headers) + '\n')
    
    def play_beep(self):
        """Play a beep sound for R peak detection"""
        current_time = time.time()
        # Only beep if enough time has passed since the last beep
        if current_time - self.last_peak_sound_time > self.sound_cooldown:
            try:
                winsound.Beep(1000, 100)  # Frequency: 1000Hz, Duration: 100ms
                self.last_peak_sound_time = current_time
            except:
                # Fall back to print if beep fails
                print("*BEEP*")
    
    def log_prediction(self, prediction_class, confidence, heart_rate):
        """Log prediction to CSV file"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        class_name = self.class_labels.get(prediction_class, "Unknown")
        
        log_entry = {
            "timestamp": timestamp,
            "prediction_class": prediction_class,
            "class_name": class_name,
            "confidence": confidence,
            "heart_rate": heart_rate,
            "signal_quality": self.signal_quality
        }
        
        self.predictions_history.append(log_entry)
        
        # Write to CSV
        with open(self.log_file, 'a') as f:
            values = [str(log_entry[key]) for key in ["timestamp", "prediction_class", "class_name", "confidence", "heart_rate", "signal_quality"]]
            f.write(','.join(values) + '\n')
        
        # Also log to debug log
        self.log_debug(f"PREDICTION: Class={prediction_class} ({class_name}), HR={heart_rate}, Quality={self.signal_quality}")
        
        # If it's an abnormality, add to reporting list
        if prediction_class != 0:  # Not normal rhythm
            self.abnormalities.append({
                "timestamp": timestamp,
                "class_name": class_name,
                "confidence": confidence
            })
    
    def design_butter_bandpass(self, lowcut=0.5, highcut=40.0, order=2):
        """Design a Butterworth bandpass filter for ECG signal using SOS format"""
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos
    
    def design_notch_filter(self, notch_freq=50.0, quality_factor=30.0):
        """Design a notch filter to remove power line interference"""
        nyq = 0.5 * self.sampling_rate
        notch_normalized = notch_freq / nyq
        
        # Using iirnotch from scipy would be better, but let's approximate with bandstop
        width = notch_normalized / quality_factor
        low = notch_normalized - width/2
        high = notch_normalized + width/2
        
        sos = butter(2, [max(0.001, low), min(0.999, high)], btype='bandstop', output='sos')
        return sos
    
    def filter_ecg(self, data):
        """Apply a combination of filters to clean the ECG signal"""
        if len(data) == 0:
            return np.array([])
            
        # Convert input to numpy array if it's not already
        data_array = np.array(data, dtype=float)
        
        # 1. Apply bandpass filter
        bandpass_sos = self.design_butter_bandpass(lowcut=0.5, highcut=40.0, order=3)
        filtered = sosfilt(bandpass_sos, data_array)
        
        # 2. Apply 50/60Hz notch filter for power line interference
        notch50_sos = self.design_notch_filter(notch_freq=50.0)  # For countries with 50Hz power
        notch60_sos = self.design_notch_filter(notch_freq=60.0)  # For countries with 60Hz power
        filtered = sosfilt(notch50_sos, filtered)
        filtered = sosfilt(notch60_sos, filtered)
        
        # 3. Apply baseline wander removal (high-pass)
        baseline_sos = butter(3, 0.5/self.sampling_rate*2, btype='highpass', output='sos')
        filtered = sosfilt(baseline_sos, filtered)
        
        return filtered
    
    def is_extreme_value(self, value):
        """Check if value is likely an error"""
        # AD8232 typically outputs in this range (adjust for your specific setup)
        # This needs to be a simple comparison, not array operation
        return value < 100 or value > 4000  # Adjust thresholds as needed
    
    def has_extreme_values(self, data_array):
        """Check if array contains extreme values"""
        # This version works with arrays
        if len(data_array) == 0:
            return False
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        return min_val < 100 or max_val > 4000
    
    def detect_r_peaks(self, filtered_data, distance=30):
        """Detect R peaks in filtered ECG data
        Returns indices of R peaks in the provided data segment"""
        if len(filtered_data) < distance:
            return []
            
        # Normalize for peak detection
        normalized = filtered_data.copy()
        std_val = np.std(normalized)
        if std_val > 0:  # Avoid division by zero
            normalized = (normalized - np.mean(normalized)) / std_val
        
        # Find peaks with prominence to avoid noise
        peaks, _ = find_peaks(normalized, distance=distance, prominence=1.5)
        
        return peaks
    
    def extract_centered_chunk(self, data, r_peak_index):
        """Extract a chunk of data with R peak at center"""
        if len(data) == 0:
            return None
            
        center = self.chunk_size // 2
        start = max(0, r_peak_index - center)
        end = start + self.chunk_size
        
        # If we don't have enough data after the R peak
        if end > len(data):
            start = max(0, len(data) - self.chunk_size)
            end = len(data)
        
        # If we still don't have a full chunk, return None
        if end - start < self.chunk_size:
            return None
            
        chunk = data[start:end]
        
        # Ensure we have exactly chunk_size samples
        if len(chunk) < self.chunk_size:
            # Pad with zeros if necessary (shouldn't happen with above checks)
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
            
        return chunk
    
    def calculate_heart_rate(self, r_peak_timestamps):
        """Calculate heart rate from R peak timestamps"""
        if len(r_peak_timestamps) < 2:
            self.log_debug(f"Not enough R-peaks to calculate HR: {len(r_peak_timestamps)}")
            return 0
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(r_peak_timestamps)
        self.log_debug(f"RR intervals: {rr_intervals}")
        
        # Filter out unreasonable intervals (physiologically impossible)
        valid_intervals = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
        self.log_debug(f"Valid intervals: {valid_intervals}, count: {len(valid_intervals)}")
        
        if len(valid_intervals) == 0:
            self.log_debug("No valid intervals found for HR calculation")
            # Use previous value but gradually reduce confidence
            current_time = time.time()
            time_since_last_valid = current_time - self.last_hr_time
            
            if time_since_last_valid < self.hr_timeout:
                # Return the last valid heart rate
                return self.last_valid_hr
            else:
                # Too much time passed, return 0
                return 0
        
        # Convert to heart rate in BPM
        heart_rate = 70 / np.mean(valid_intervals)*1.1   #############################################
        self.log_debug(f"Calculated HR: {heart_rate} BPM")
        
        # Store as last valid heart rate
        self.last_valid_hr = heart_rate
        self.last_hr_time = time.time()
        
        return heart_rate
    
    def check_signal_quality(self, raw_values):
        """Assess signal quality based on various metrics"""
        if len(raw_values) < self.sampling_rate:
            return "Insufficient data"
            
        # Convert to numpy array if it's not already
        raw_array = np.array(raw_values)
        
        # Check for flat line
        if np.std(raw_array) < 10:
            self.consecutive_errors += 1
            return "Poor - Flat signal"
            
        # Check for extreme values
        if self.has_extreme_values(raw_array):
            extreme_count = np.sum((raw_array < 100) | (raw_array > 4000))
            extreme_ratio = extreme_count / len(raw_array)
            
            if extreme_ratio > 0.1:  # More than 10% are extreme values
                self.consecutive_errors += 1
                return "Poor - Noisy signal"
        
        # Reset consecutive errors if signal is good
        self.consecutive_errors = 0
        return "Good"
    
    def predict_heartbeat_class(self, ecg_chunk):
        """Run inference on ECG chunk"""
        # Save the raw chunk for debugging (first time only)
        if not hasattr(self, 'saved_first_chunk'):
            self.log_debug(f"First inference chunk raw data: {ecg_chunk.tolist()}")
            self.saved_first_chunk = True
        
        # Normalize the chunk
        mean = np.mean(ecg_chunk)
        std = np.std(ecg_chunk)
        if std == 0:
            std = 1  # Prevent division by zero
        normalized_chunk = (ecg_chunk - mean) / std
        
        # Log normalized data occasionally
        if np.random.random() < 0.05:  # 5% chance to log
            self.log_debug(f"Normalized chunk stats: mean={mean}, std={std}")
            self.log_debug(f"Normalized chunk sample: {normalized_chunk[:10]}...")
        
        # Make prediction
        prediction = self.model.predict(normalized_chunk.reshape(1, -1), verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        return predicted_class, confidence, prediction
    
    def setup_visualization(self):
        """Setup matplotlib figure for visualization"""
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.axes = [ax1, ax2]
        
        # Setup ECG plots (raw and filtered)
        self.line_raw, = ax1.plot([], [], 'r-', lw=1, alpha=0.5, label='Raw')
        self.line_filtered, = ax1.plot([], [], 'b-', lw=2, label='Filtered')
        self.r_peak_scatter = ax1.scatter([], [], color='green', marker='o', s=50, label='R-Peaks')
        
        ax1.set_title('Real-time ECG Signal')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        
        # Prediction bars
        labels = list(self.class_labels.values())
        x = np.arange(len(labels))
        self.bars = ax2.bar(x, [0] * len(labels), color='gray')
        ax2.set_title('Heartbeat Classification')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Probability')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha='right')
        ax2.set_ylim(0, 1)
        
        # Status text at the top
        self.status_text = self.fig.text(0.5, 0.95, 'Initializing... Press Q to quit', 
                           ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the status text
        
        return self.fig
    
    def update_visualization(self, predictions=None, heart_rate=0):
        """Update visualization with new data"""
        # Only update plot at certain intervals to avoid performance issues
        current_time = time.time()
        if current_time - self.last_plot_time < self.plot_update_interval:
            return
            
        self.last_plot_time = current_time
        
        if self.fig is None:
            return
            
        # Get data for display (last N samples)
        if len(self.raw_buffer) > 0:
            display_raw = np.array(self.raw_buffer[-min(len(self.raw_buffer), self.visualization_buffer_size):])
            
            if len(self.filtered_buffer) > 0:
                # Make sure filtered buffer matches raw buffer length
                display_filtered = self.filtered_buffer[-min(len(self.filtered_buffer), len(display_raw)):]
                
                # Update lines with data
                x_values = np.arange(len(display_raw))
                self.line_raw.set_data(x_values, display_raw)
                
                if len(display_filtered) == len(display_raw):
                    self.line_filtered.set_data(x_values, display_filtered)
                
                # Update r-peaks for visualization
                valid_peaks = []
                for peak in self.r_peak_buffer:
                    # Check if peak is within the visible range
                    adjusted_idx = peak - (len(self.raw_buffer) - len(display_raw))
                    if 0 <= adjusted_idx < len(display_raw):
                        valid_peaks.append(adjusted_idx)
                
                if valid_peaks:
                    peak_y = [display_filtered[idx] if idx < len(display_filtered) else 0 for idx in valid_peaks]
                    self.r_peak_scatter.set_offsets(np.column_stack((valid_peaks, peak_y)))
                else:
                    self.r_peak_scatter.set_offsets(np.empty((0, 2)))
                
                # Adjust axes limits
                self.axes[0].set_xlim(0, len(display_raw))
                
                max_val = max(np.max(display_raw), np.max(display_filtered)) if len(display_filtered) > 0 else np.max(display_raw)
                min_val = min(np.min(display_raw), np.min(display_filtered)) if len(display_filtered) > 0 else np.min(display_raw)
                padding = (max_val - min_val) * 0.1 if max_val > min_val else 100
                self.axes[0].set_ylim(min_val - padding, max_val + padding)
        
        # Update prediction bars with smoothing
        if predictions is not None:
            # Apply smoothing to predictions for visualization stability
            alpha = 0.3  # Smoothing factor
            self.last_prediction = alpha * predictions + (1 - alpha) * self.last_prediction
            
            for i, bar in enumerate(self.bars):
                bar.set_height(self.last_prediction[i])
                
                # Color coding
                if i == 0:  # Normal class
                    bar.set_color('green')
                else:
                    bar.set_color('red')
                
                # Highlight the predicted class
                # if i == np.argmax(self.last_prediction):
                #     bar.set_color('green')
        
        # Update status text
        status = f"Signal Quality: {self.signal_quality} | Heart Rate: {heart_rate:.1f} BPM | Press Q to quit"
        self.status_text.set_text(status)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # Small pause to allow GUI to update
    
    def generate_final_report(self):
        """Generate a comprehensive report from the ECG session"""
        # Add summary information
        self.report_generator.add_section("ECG Analysis Summary", 
                                         f"Session Duration: {time.time() - self.last_plot_time:.1f} seconds\n"
                                         f"Average Heart Rate: {self.last_valid_hr:.1f} BPM\n"
                                         f"Signal Quality: {self.signal_quality}\n"
                                         f"Total Abnormalities Detected: {len(self.abnormalities)}")
        
        # Add heart rate analysis
        if hasattr(self, 'predictions_history') and self.predictions_history:
            # Extract heart rates from the history
            heart_rates = [entry.get('heart_rate', 0) for entry in self.predictions_history if entry.get('heart_rate', 0) > 0]
            
            if heart_rates:
                avg_hr = np.mean(heart_rates)
                min_hr = np.min(heart_rates)
                max_hr = np.max(heart_rates)
                
                hr_analysis = (f"Average: {avg_hr:.1f} BPM\n"
                              f"Minimum: {min_hr:.1f} BPM\n"
                              f"Maximum: {max_hr:.1f} BPM\n"
                              f"Variability: {np.std(heart_rates):.2f} BPM")
                
                self.report_generator.add_section("Heart Rate Analysis", hr_analysis)
        
        # Add rhythm analysis
        if hasattr(self, 'predictions_history') and self.predictions_history:
            rhythm_counts = {}
            for entry in self.predictions_history:
                class_name = entry.get('class_name', 'Unknown')
                if class_name in rhythm_counts:
                    rhythm_counts[class_name] += 1
                else:
                    rhythm_counts[class_name] = 1
            
            rhythm_analysis = "Rhythm Distribution:\n"
            for rhythm, count in rhythm_counts.items():
                rhythm_analysis += f"- {rhythm}: {count} instances\n"
            
            self.report_generator.add_section("Rhythm Analysis", rhythm_analysis)
        
        # Generate the final PDF
        patient_data = {
            "name": "Deepak",  # You would replace this with actual data
            "age": "20"  # You would replace this with actual data
        }
        
        self.report_generator.generate_report(patient_data, self.abnormalities)
    
    async def process_ecg_data(self):
        """Process incoming ECG data and display results"""
        print(f"Connecting to WebSocket at {self.websocket_uri}")
        self.log_debug(f"Starting ECG monitoring, connecting to {self.websocket_uri}")
        
        # Setup visualization
        self.setup_visualization()
        plt.ion()  # Turn on interactive mode
        
        try:
            async with websockets.connect(self.websocket_uri) as websocket:
                print("✅ Connected to ESP32 WebSocket server!")
                self.log_debug("Successfully connected to WebSocket server")
                
                while self.is_running:
                    try:
                        # Receive data from WebSocket
                        data = await websocket.recv()
                        try:
                            ecg_value = int(data.strip())
                        except ValueError:
                            print(f"Invalid data received: {data}")
                            self.log_debug(f"Invalid data received: {data}")
                            continue
                        
                        # Check if it's an extreme value (single value check)
                        if self.is_extreme_value(ecg_value):
                            if np.random.random() < 0.01:  # Only log occasionally
                                self.log_debug(f"Extreme value detected: {ecg_value}")
                            self.consecutive_errors += 1
                            if self.consecutive_errors > self.max_consecutive_errors:
                                self.signal_quality = "Poor - Device Error"
                            # Still add to buffer but mark it
                            self.raw_buffer.append(ecg_value)
                            continue
                        
                        # Add to raw buffer
                        self.raw_buffer.append(ecg_value)
                        
                        # Keep buffer size manageable
                        if len(self.raw_buffer) > self.visualization_buffer_size * 2:
                            self.raw_buffer = self.raw_buffer[-self.visualization_buffer_size:]
                        
                        # Process data when we have enough samples
                        if len(self.raw_buffer) >= self.chunk_size:
                            # Apply filter to entire buffer
                            self.filtered_buffer = self.filter_ecg(np.array(self.raw_buffer))
                            
                            # Assess signal quality
                            self.signal_quality = self.check_signal_quality(self.raw_buffer[-self.sampling_rate:])
                            
                            # Only process if we have good signal quality
                            if "Good" in self.signal_quality:
                                # Get latest data for R-peak detection
                                latest_data = self.filtered_buffer[-int(self.sampling_rate * 2):]  # Last 2 seconds
                                
                                # Find new R peaks
                                new_peaks = self.detect_r_peaks(latest_data)
                                
                                # Adjust peak indices to match the full buffer
                                buffer_offset = len(self.filtered_buffer) - len(latest_data)
                                adjusted_peaks = [int(peak + buffer_offset) for peak in new_peaks]
                                
                                # Record timestamps of R peaks and play sound
                                current_time = time.time()
                                for peak in adjusted_peaks:
                                    # Only add new peaks (not seen before)
                                    if not self.last_r_peaks or peak > max(self.last_r_peaks):
                                        self.last_r_peaks.append(peak)
                                        self.r_peak_timestamps.append(current_time)
                                        self.r_peak_buffer.append(peak)
                                        self.log_debug(f"New R-peak detected at index {peak}, time {current_time}")
                                        # Play beep sound for new R peak
                                        self.play_beep()
                                
                                # Keep only recent peaks
                                if len(self.last_r_peaks) > 10:
                                    self.last_r_peaks = self.last_r_peaks[-10:]
                                    self.r_peak_timestamps = self.r_peak_timestamps[-10:]
                                
                                if len(self.r_peak_buffer) > 20:
                                    self.r_peak_buffer = self.r_peak_buffer[-20:]
                                
                                # Process the latest R peak for classification
                                if self.last_r_peaks and len(self.filtered_buffer) >= self.chunk_size:
                                    # Get last R peak index in filtered buffer
                                    latest_r_peak = self.last_r_peaks[-1]
                                    
                                    # Extract centered chunk around R peak
                                    if 0 <= latest_r_peak < len(self.filtered_buffer):
                                        chunk = self.extract_centered_chunk(self.filtered_buffer, latest_r_peak)
                                        
                                        if chunk is not None and len(chunk) == self.chunk_size:
                                            # Make prediction
                                            predicted_class, confidence, predictions = self.predict_heartbeat_class(chunk)
                                            
                                            # Calculate heart rate
                                            heart_rate = self.calculate_heart_rate(self.r_peak_timestamps)
                                            
                                            # Track condition for persistent logging
                                            current_time = time.time()
                                            condition = {
                                                "class": predicted_class,
                                                "heart_rate": heart_rate,
                                                "signal_quality": self.signal_quality
                                            }
                                            
                                            # Check if condition changed
                                            if (self.last_condition is None or 
                                                self.last_condition["class"] != condition["class"] or
                                                abs(self.last_condition["heart_rate"] - condition["heart_rate"]) > 5):
                                                # New condition detected
                                                self.last_condition = condition
                                                self.condition_start_time = current_time
                                                self.log_debug(f"New condition detected: {condition}")
                                            else:
                                                # Same condition still active
                                                condition_duration = current_time - self.condition_start_time
                                                if condition_duration > self.condition_logging_delay:
                                                    # Condition has persisted for more than 3 seconds, log it
                                                    self.log_debug(f"Condition persisted for {condition_duration:.1f}s, logging it")
                                                    self.log_prediction(predicted_class, confidence, heart_rate)
                                                    # Reset timer after logging
                                                    self.condition_start_time = current_time
                                            
                                            # Update visualization with prediction results
                                            self.update_visualization(predictions, heart_rate)
                                        else:
                                            # Update visualization without new predictions
                                            heart_rate = self.calculate_heart_rate(self.r_peak_timestamps)
                                            self.update_visualization(heart_rate=heart_rate)
                                    else:
                                        # R-peak outside buffer, just update visualization
                                        heart_rate = self.calculate_heart_rate(self.r_peak_timestamps)
                                        self.update_visualization(heart_rate=heart_rate)
                                else:
                                    # No R-peaks detected yet
                                    self.update_visualization()
                            else:
                                # Poor signal quality, just update visualization
                                self.update_visualization()
                        
                    except websockets.exceptions.ConnectionClosed:
                        print("❌ Connection closed, attempting to reconnect...")
                        self.log_debug("WebSocket connection closed, attempting to reconnect")
                        break
                    except Exception as e:
                        print(f"❌ Error in processing: {e}")
                        self.log_debug(f"Error in processing: {e}")
                        # Continue trying to process data
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.log_debug(f"Connection error: {e}")
            # Try to generate report even if connection failed
            self.generate_final_report()
    
    async def run(self):
        """Main entry point for the ECG monitor"""
        print("Starting ECG Monitor...")
        print("Press 'q' at any time to quit and generate report.")
        
        try:
            await self.process_ecg_data()
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            # Generate report when exiting
            print("Generating final report...")
            self.generate_final_report()
            print("Monitor shutting down")
            # Cleanup
            if self.fig is not None:
                plt.close(self.fig)

# Main execution
async def main():
    # Configuration - update these values according to your ESP32 setup
    websocket_uri = "ws://192.168.171.142:81/"  # Update this to your ESP32's IP address and port
    
    # Create and run the ECG monitor
    monitor = ECGMonitor(websocket_uri=websocket_uri)
    await monitor.run()

if __name__ == "__main__":
    # Create the output directory for reports if it doesn't exist
    os.makedirs("Report Generated", exist_ok=True)
    
    # Run the asyncio event loop
    asyncio.run(main())