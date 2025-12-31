import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import os
import sys

# Add src folder to path
sys.path.append('src')

# Import your existing modules
try:
    from fire_detection import FireDetector  # Changed from FireDetection to FireDetector
    from fire_detection_alarm import FireAlarmSystem
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure your src folder contains the required modules")

class FireDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 Fire Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        # Variables
        self.is_detecting = False
        self.cap = None
        self.detector = None
        self.alarm_system = None
        self.last_alert_time = 0

        # GUI Setup
        self.setup_gui()

        # Initialize systems
        self.initialize_systems()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#34495e', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        title_label = tk.Label(title_frame, text="🔥 REAL-TIME FIRE DETECTION SYSTEM",
                               font=('Arial', 24, 'bold'), fg='white', bg='#34495e')
        title_label.pack(pady=20)

        # Main Content Frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Left Frame - Video Display
        self.video_frame = tk.Frame(main_frame, bg='black', width=640, height=480)
        self.video_frame.pack(side='left', padx=(0, 20))
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.pack(fill='both', expand=True)

        # Right Frame - Controls and Info
        right_frame = tk.Frame(main_frame, bg='#2c3e50', width=300)
        right_frame.pack(side='right', fill='y')
        right_frame.pack_propagate(False)

        # Status Panel
        status_panel = tk.Frame(right_frame, bg='#34495e', relief='ridge', bd=2)
        status_panel.pack(fill='x', pady=(0, 20))

        tk.Label(status_panel, text="SYSTEM STATUS", font=('Arial', 14, 'bold'),
                 fg='white', bg='#34495e').pack(pady=(10, 5))

        self.status_label = tk.Label(status_panel, text="READY", font=('Arial', 12),
                                     fg='green', bg='#34495e')
        self.status_label.pack(pady=5)

        self.fire_status_label = tk.Label(status_panel, text="No Fire Detected",
                                          font=('Arial', 16, 'bold'), fg='white', bg='#34495e')
        self.fire_status_label.pack(pady=10)

        self.confidence_label = tk.Label(status_panel, text="Confidence: 0%",
                                         font=('Arial', 12), fg='yellow', bg='#34495e')
        self.confidence_label.pack(pady=5)

        # Control Buttons
        control_frame = tk.Frame(right_frame, bg='#2c3e50')
        control_frame.pack(fill='x', pady=20)

        self.start_btn = tk.Button(control_frame, text="▶ START DETECTION",
                                   font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                   command=self.start_detection, height=2, width=20)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(control_frame, text="⏹ STOP DETECTION",
                                  font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                  command=self.stop_detection, height=2, width=20, state='disabled')
        self.stop_btn.pack(pady=5)

        # Settings Frame
        settings_frame = tk.Frame(right_frame, bg='#34495e', relief='ridge', bd=2)
        settings_frame.pack(fill='x', pady=10)

        tk.Label(settings_frame, text="SETTINGS", font=('Arial', 12, 'bold'),
                 fg='white', bg='#34495e').pack(pady=(10, 5))

        # Confidence Threshold
        tk.Label(settings_frame, text="Confidence Threshold:",
                 font=('Arial', 10), fg='white', bg='#34495e').pack(anchor='w', padx=10)

        self.threshold_slider = tk.Scale(settings_frame, from_=0.5, to=1.0,
                                         resolution=0.05, orient='horizontal',
                                         bg='#34495e', fg='white', length=250)
        self.threshold_slider.set(0.85)
        self.threshold_slider.pack(padx=10, pady=5)

        # Alarm Toggle
        self.alarm_var = tk.BooleanVar(value=True)
        alarm_check = tk.Checkbutton(settings_frame, text="Enable Alarm",
                                     variable=self.alarm_var, font=('Arial', 10),
                                     bg='#34495e', fg='white', selectcolor='#34495e')
        alarm_check.pack(pady=10)

        # Log Frame
        log_frame = tk.Frame(right_frame, bg='#34495e', relief='ridge', bd=2)
        log_frame.pack(fill='both', expand=True, pady=(10, 0))

        tk.Label(log_frame, text="DETECTION LOG", font=('Arial', 12, 'bold'),
                 fg='white', bg='#34495e').pack(pady=(10, 5))

        self.log_text = tk.Text(log_frame, height=8, width=30, bg='#1c2833',
                                fg='white', font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True, padx=10, pady=5)

    def initialize_systems(self):
        """Initialize detection and alarm systems"""
        try:
            # Load your existing fire detection system
            model_path = "models/keras_Model.h5"
            if not os.path.exists(model_path):
                model_path = "../models/keras_Model.h5"

            self.detector = FireDetector(model_path)
            self.alarm_system = FireAlarmSystem()
            self.log_message("✅ Systems initialized successfully")
            self.log_message(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            self.log_message(f"❌ Error initializing systems: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize systems: {str(e)}")

    def start_detection(self):
        """Start the fire detection"""
        if not self.is_detecting:
            self.is_detecting = True
            self.status_label.config(text="DETECTING", fg='yellow')
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')

            # Start detection in separate thread
            self.detection_thread = threading.Thread(target=self.detect_fire)
            self.detection_thread.daemon = True
            self.detection_thread.start()

            self.log_message("🚀 Fire detection started")

    def stop_detection(self):
        """Stop the fire detection"""
        self.is_detecting = False
        self.status_label.config(text="STOPPED", fg='red')
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

        if self.cap:
            self.cap.release()
            self.cap = None

        self.log_message("⏹ Fire detection stopped")

    def detect_fire(self):
        """Main detection loop"""
        # Try different camera indexes
        for i in range(3):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                self.log_message(f"✅ Camera found at index {i}")
                break
        else:
            self.log_message("❌ Cannot open any camera")
            self.is_detecting = False
            return

        while self.is_detecting:
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("❌ Cannot read from camera")
                break

            # Flip frame horizontally
            frame = cv2.flip(frame, 1)

            # Process frame using your detector
            try:
                prediction, confidence = self.detector.predict(frame)
                is_fire = (prediction == "Fire")

                # Update GUI
                self.update_display(frame, is_fire, confidence)

                # Trigger alarm if fire detected
                if is_fire and confidence > self.threshold_slider.get() and self.alarm_var.get():
                    self.trigger_alarm(confidence)

            except Exception as e:
                self.log_message(f"Error: {str(e)}")

            time.sleep(0.03)  # ~30 FPS

        if self.cap:
            self.cap.release()

    def update_display(self, frame, is_fire, confidence):
        """Update the video display and status"""
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw detection results on frame
        text = f"{'Fire' if is_fire else 'Safe'}: {confidence:.2%}"
        color = (255, 0, 0) if is_fire else (0, 255, 0)  # BGR format
        cv2.putText(frame_rgb, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert to ImageTk
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update video label
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Update status labels
        status_text = "🔥 FIRE DETECTED!" if is_fire else "✅ Safe"
        self.fire_status_label.config(text=status_text,
                                      fg='red' if is_fire else 'green')
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")

        # Log detection
        if is_fire:
            self.log_message(f"🚨 Fire detected: {confidence:.2%}")

    def trigger_alarm(self, confidence):
        """Trigger alarm system with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time > 5:  # 5 second cooldown
            self.log_message(f"🚨 ALARM TRIGGERED! Confidence: {confidence:.2%}")

            # Visual alert (flash red)
            self.root.configure(bg='red')
            self.root.after(500, lambda: self.root.configure(bg='#2c3e50'))

            # Sound alarm
            if self.alarm_system:
                self.alarm_system.trigger()

            self.last_alert_time = current_time

    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # Insert at the beginning (top of log)
        self.log_text.insert('1.0', log_entry)

        # Keep last 50 lines
        lines = self.log_text.get(1.0, tk.END).split('\n')
        if len(lines) > 50:
            self.log_text.delete(f"{len(lines) - 50}.0", tk.END)

    def on_closing(self):
        """Handle window closing"""
        self.is_detecting = False
        if self.cap:
            self.cap.release()
        if self.alarm_system:
            self.alarm_system.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FireDetectionGUI(root)

    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()







