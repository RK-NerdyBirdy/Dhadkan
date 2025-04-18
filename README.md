# Dhadkan <3: An ECG Monitoring System with Real-time Analysis for Tele-medecine facilites

## Overview
This project provides a real-time ECG monitoring system that connects to an ESP32-based ECG sensor via WebSocket. The system processes incoming ECG data, detects heartbeats, classifies heart rhythms, and provides visualization and reporting capabilities. It is an helping hand to doctors who are working day and night for patient service and helps them to easily read the ecg reports with abnormalities at specefic timestamps.

## Key Features

- **Real-time ECG Monitoring**: Connects to an ESP32 device via WebSocket to receive ECG data
- **Signal Processing**:
  - Bandpass filtering (0.5-40 Hz)
  - Notch filtering for 50/60Hz power line interference removal
  - Baseline wander removal
- **Heartbeat Detection**: R-peak detection using prominence-based algorithm
- **Heartbeat Classification**: Uses a pre-trained CNN model to classify heartbeats into:
  - Normal Sinus Rhythm
  - Atrial Premature
  - Premature Ventricular Contraction
  - Fusion Beat
  - Unknown Abnormality
- **Visualization**: Real-time display of raw and filtered ECG signals with R-peak markers
- **Reporting**: Generates PDF reports summarizing ECG findings and detected abnormalities
- **Signal Quality Monitoring**: Continuously assesses signal quality

## Requirements

### Hardware
- ESP32 microcontroller with ECG sensor (e.g., AD8232 ECG module)
- Computer with Python 3.7+

### Software Dependencies
- Python packages:
  - `websockets`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `tensorflow` (for CNN model)
  - `keyboard` (for keyboard input)
  - `winsound` (Windows only, for beep sounds)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ecg-monitor.git
   cd ecg-monitor
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your trained CNN model file (`ecg_cnn_model.h5`) in the project directory

## Usage

1. Ensure your ESP32 is running the ECG WebSocket server and is connected to the same network as your computer.

2. Update the WebSocket URI in `main.py` to match your ESP32's IP address:
   ```python
   websocket_uri = "ws://192.168.1.100:81"  # Update with your ESP32's IP
   ```

3. Run the ECG monitor:
   ```bash
   python main.py
   ```

4. The system will:
   - Connect to the ESP32 WebSocket server
   - Display real-time ECG visualization
   - Play a beep sound for each detected heartbeat
   - Show classification probabilities

5. Press `q` at any time to quit and generate a report.

## Output Files

The system generates several output files in the `ecg_logs` directory:
- `ecg_session_YYYYMMDD_HHMMSS.csv`: CSV log of all ECG data and classifications
- `ecg_debug_YYYYMMDD_HHMMSS.log`: Debug log file

Reports are generated in the `Report Generated` directory as PDF files.

## Configuration

You can modify these parameters in the `ECGMonitor` class initialization:
- `sampling_rate`: ECG sampling rate (default: 125 Hz)
- `chunk_size`: Number of samples for CNN input (default: 187)
- `visualization_buffer_size`: Number of samples to display (default: 1000)
- `sound_cooldown`: Minimum time between beep sounds (default: 0.3s)
- `condition_logging_delay`: Delay before logging persistent conditions (default: 3s)

## Troubleshooting

1. **Connection Issues**:
   - Verify ESP32 is connected to the network
   - Check firewall settings to allow WebSocket connections
   - Ensure correct IP address in `websocket_uri`

2. **Model Loading Errors**:
   - Verify `ecg_cnn_model.h5` exists in the project directory
   - Check TensorFlow version compatibility

3. **Performance Issues**:
   - Reduce `visualization_buffer_size`
   - Increase `plot_update_interval`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AD8232 ECG sensor
- ESP32 microcontroller
- TensorFlow for deep learning capabilities
- Various Python libraries used for signal processing and visualization