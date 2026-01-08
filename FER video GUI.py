"""
==============================================================================
Handshake
==============================================================================

The "Handshake" (Interaction Protocol) occurs in 4 phases:

1. INITIALIZATION (Setup Phase)
   - Trigger: User loads model weights via GUI.
   - Action: Frontend instantiates `CoatNetBackend(path)`.
   - Result: Backend initializes PyTorch, detects GPU/CPU, and loads CoatNet 
     weights + Haar Cascade into memory.

2. REAL-TIME PROCESSING LOOP (Frame-by-Frame Handshake)
   - Occurs synchronously within the video thread (~30 times/sec).
   - Request: Frontend sends raw image -> `backend.process_frame(frame)`.
   - Inference: Backend performs Face Detection -> Cropping -> Normalization -> AI Inference.
   - Response: Backend returns a tuple:
     (Emotion Label, Confidence Score, Emotion Index, Bounding Box Coords).

3. LOGIC MAPPING (Shared State)
   - The Frontend accesses `backend.MAPPING_RULE` directly.
   - Purpose: Maps specific emotion indices (e.g., 0 for Anger) to broader 
     categories (e.g., Negative) for graph coloring without duplicating logic.

4. SESSION AGGREGATION (Termination Phase)
   - Trigger: Video ends or User clicks Stop.
   - Action: Frontend calls `backend.get_gap_result()`.
   - Result: Backend calculates Global Average Pooling (GAP) of all frame 
     probabilities to determine the "Dominant Emotion" for the entire session.
==============================================================================
"""


#importing dependencies 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2 
import PIL.Image, PIL.ImageTk 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading  
import time
from datetime import timedelta
import torch
from torchvision import transforms
import timm  

# TkAgg for matplotlib 
matplotlib.use("TkAgg")


# ==============================================================================
# BACKEND LOGIC
# ==============================================================================

class CoatNetBackend:
    def __init__(self, weights_path=None):
        # CheckGPU 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # emotion labels 
        self.EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # MAPPING RULE
        self.MAPPING_RULE = {
            3: 0, # Happy -> Positive
            4: 1, # Neutral -> Ambiguous
            6: 1, # Surprise -> Ambiguous
            0: 2, # Anger -> Negative
            1: 2, # Disgust -> Negative
            2: 2, # Fear -> Negative
            5: 2  # Sad -> Negative
        }
        
        # Variables to track GAP results
        self.gap_totals = {e: 0.0 for e in self.EMOTION_LABELS}
        self.frame_count = 0
        
        # Load Haar Cascade 
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load CoatNet model
        self.model = self._load_model(weights_path)
        
        # image preprocessing
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),# imagenet normalization
        ])
    #Loads the CoatNet architecture and applies the weights.
    def _load_model(self, path):

        if not path: return None
        print(f"Loading coatnet_0_rw_224 from {path}...")
        try:
            # model archietecture
            model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=7)
            
            # Load the actual weights  
            checkpoint = torch.load(path, map_location=self.device)
        
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()  
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def process_frame(self, frame):
 
        if self.model is None: return None, 0.0
        
        # Convert to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0: return "No Face", 0.0

        # If multiple faces are found, pick the largest one 
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # Preprocess face for the model
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb_face)
            input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device) 

            with torch.no_grad(): 
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0) 
                prob, idx = torch.max(probs, 0) # Get the highest probability and its index
                
                emotion_idx = idx.item()
                emotion = self.EMOTION_LABELS[emotion_idx]
                confidence = prob.item()
                
                # Accumulate probabilities for GAP (Global Average) calculation
                for i, label in enumerate(self.EMOTION_LABELS):
                    self.gap_totals[label] += probs[i].item()
                self.frame_count += 1
                
                return emotion, confidence, emotion_idx, (x, y, w, h)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", 0.0
    #Calculates the average emotion across the entire video duration.
    def gap(self):
        
        if self.frame_count == 0: return "N/A", 0.0

        avgs = {k: v / self.frame_count for k, v in self.gap_totals.items()}

        winner = max(avgs, key=avgs.get)
        return winner, avgs[winner] * 100

# ==============================================================================
# GUI IMPLEMENTATION 
# ==============================================================================

class CoatNetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Analyzer (Duration Tracking)")
        self.root.geometry("1100x750")
        
        # Color palette definition
        self.colors = {"bg": "#1e1e1e", "panel": "#2d2d2d", "accent": "#04A8E9", "text": "white", "alert": "#ff5555"}
        self.root.configure(bg=self.colors["bg"])
        
        self.backend = None
        self.is_running = False
        self.history_data = [] # Stores (timestamp, category) 
        
        # State variable to track the currently ongoing negative emotion 
        self.active_log = None 

        self._build_ui()

    def _build_ui(self):
        """Constructs the visual layout of the application."""
        # --- HEADER ---
        header = tk.Frame(self.root, bg=self.colors["panel"], height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="CoatNet Emotion Analyzer", bg=self.colors["panel"], fg="white", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=20, pady=15)
        
        # Buttons for loading files
        btn_fr = tk.Frame(header, bg=self.colors["panel"])
        btn_fr.pack(side=tk.RIGHT, padx=20)
        self.btn_load_w = tk.Button(btn_fr, text="1. Load Model (.pt)", command=self.load_weights, bg="#444", fg="white", relief=tk.FLAT)
        self.btn_load_w.pack(side=tk.LEFT, padx=5)
        self.btn_load_v = tk.Button(btn_fr, text="2. Load Video", command=self.load_video, bg="#444", fg="white", relief=tk.FLAT)
        self.btn_load_v.pack(side=tk.LEFT, padx=5)

        # --- MAIN CONTENT AREA ---
        main = tk.Frame(self.root, bg=self.colors["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Side: Video Player
        self.video_lbl = tk.Label(main, bg="black", text="Load Model & Video to Start", fg="#555")
        self.video_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Side: Dashboard/Stats
        dash = tk.Frame(main, bg=self.colors["bg"], width=300)
        dash.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        dash.pack_propagate(False) # Prevent frame from shrinking to fit content

        self._card(dash, "CURRENT FRAME", "lbl_emo", "lbl_conf")
        self._card(dash, "VIDEO DOMINANT EMOTION (GAP)", "lbl_gap_emo", "lbl_gap_conf", highlight=True)
        
        # Listbox for logging negative emotions
        tk.Label(dash, text="NEGATIVEs LOG (Duration)", bg=self.colors["bg"], fg="#888", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(20, 5))
        self.log = tk.Listbox(dash, bg=self.colors["panel"], fg="white", bd=0, height=8)
        self.log.pack(fill=tk.X)

        # --- BOTTOM GRAPH AREA ---
        graph_fr = tk.Frame(self.root, bg=self.colors["bg"], height=200)
        graph_fr.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 20))
        
        # Configure Matplotlib Graph
        self.fig = Figure(figsize=(10, 2), dpi=100, facecolor=self.colors["bg"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#252526")
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(['Positive', 'Ambiguous', 'Negative']) 
        self.ax.set_ylim(-0.5, 2.5)
        # Style the graph to look modern/dark
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color("#888")
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(colors="#888")
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        self.line, = self.ax.plot([], [], 'o-', color=self.colors["accent"], markersize=4, linewidth=1, alpha=0.8)
        
        # Embed Graph in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_fr)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Footer controls 
        footer = tk.Frame(graph_fr, bg=self.colors["bg"])
        footer.pack(fill=tk.X, pady=(10,0))
        self.btn_start = tk.Button(footer, text="▶ START", command=self.toggle, bg=self.colors["accent"], fg="white", font=("Segoe UI", 10, "bold"), state=tk.DISABLED, relief=tk.FLAT)
        self.btn_start.pack(side=tk.LEFT)
        self.prog_var = tk.DoubleVar()
        ttk.Progressbar(footer, variable=self.prog_var, maximum=100).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
    #Helper function to create styled statistic cards.
    def _card(self, p, title, e_attr, c_attr, highlight=False):
     
        f = tk.Frame(p, bg=self.colors["panel"], padx=15, pady=15)
        f.pack(fill=tk.X, pady=(0, 10))
        tk.Label(f, text=title, bg=self.colors["panel"], fg="#888", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        l_e = tk.Label(f, text="---", bg=self.colors["panel"], fg=self.colors["accent"] if highlight else "white", font=("Segoe UI", 20, "bold"))
        l_e.pack(anchor="w")
        setattr(self, e_attr, l_e) 
        l_c = tk.Label(f, text="Confidence: --%", bg=self.colors["panel"], fg="#888", font=("Segoe UI", 9))
        l_c.pack(anchor="w")
        setattr(self, c_attr, l_c)
    #Open file dialog to select the .pt model weights file.
    def load_weights(self):
        
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt *.pth")])
        if path:
            self.backend = CoatNetBackend(path)
            if self.backend.model:
                messagebox.showinfo("Success", "CoatNet Model Loaded!")
                self.btn_load_w.config(bg="green")
    #Open file dialog to select the video file.
    def load_video(self):
        
        if not self.backend: return messagebox.showerror("Error", "Load Model Weights first!")
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if path:
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.btn_load_v.config(bg="green")
            self.btn_start.config(state=tk.NORMAL)
            self.video_lbl.config(text=f"Ready: {path.split('/')[-1]}")
    #Start or Stop the video processing loop.
    def toggle(self):
        
        if not self.is_running:
            self.is_running = True
            self.btn_start.config(text="■ STOP")
            # Run the loop in a separate THREAD so the GUI doesn't freeze
            threading.Thread(target=self.loop, daemon=True).start()
        else:
            self.is_running = False
            self.btn_start.config(text="▶ START")
    #The main processing loop running in a background thread.
    def loop(self):
        
        frame_idx = 0
        self.history_data = []
        
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break # End of video
            
   
            res = self.backend.process_frame(frame)
            
            display_emo = "Scanning..."
            display_conf = 0.0
            display_cat = 1 # Default to Ambiguous if face not found
            curr_time = frame_idx / self.fps

            if isinstance(res, tuple) and len(res) == 4:
                # Face detected and processed
                emo, conf, emo_idx, (x,y,w,h) = res
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw box on face
                display_emo = emo
                display_conf = conf
                display_cat = self.backend.MAPPING_RULE.get(emo_idx, 1)

                # RECORD DATA for graph
                self.history_data.append((curr_time, display_cat))
            else:
             
                self.history_data.append((curr_time, None))
            

            self.root.after(0, self.update_ui, frame, display_emo, display_conf, curr_time, display_cat)
            
            frame_idx += 1

        self.is_running = False
        self.root.after(0, self.finish)
    def update_ui(self, frame, emo, conf, t, cat):
        
        
        # 1. Update Video Image
        f = cv2.resize(frame, (640, 360))
        img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        self.video_lbl.configure(image=img)
        self.video_lbl.image = img 
        
        # 2. Update Dashboard Text
        color = self.colors["alert"] if cat == 2 else "white" # Red text if negative emotion
        self.lbl_emo.config(text=emo, fg=color)
        self.lbl_conf.config(text=f"Confidence: {conf*100:.1f}%")
        
        if self.total_frames > 0:
            self.prog_var.set((t / (self.total_frames/self.fps)) * 100)
        
        # 3. LOGGING LOGIC
        is_negative = (cat == 2 and emo != "Scanning...")

        if is_negative:
            # Check if we are already tracking the same emotion
            if self.active_log and self.active_log['emotion'] == emo:

                duration = t - self.active_log['start']
                if conf > self.active_log['max_conf']:
                    self.active_log['max_conf'] = conf 
                

                new_text = f"[{self.active_log['ts_str']}] {emo} ({self.active_log['max_conf']*100:.0f}%) ⏱ {duration:.2f}s"
                self.log.delete(0) 
                self.log.insert(0, new_text)
            else:
                ts_str = str(timedelta(seconds=int(t)))
                self.active_log = {
                    'emotion': emo,
                    'start': t,
                    'max_conf': conf,
                    'ts_str': ts_str
                }
                text = f"[{ts_str}] {emo} ({conf*100:.0f}%) ⏱ 0.00s"
                self.log.insert(0, text)
        else:
            self.active_log = None

        # 4. Update Graph
        if len(self.history_data) > 1:
            xs, ys = zip(*self.history_data)
            self.line.set_data(xs, ys)
            self.ax.set_xlim(0, max(10, t))
            self.canvas.draw_idle() 
#Called when video ends or stop is pressed.
    def finish(self):
        
        self.btn_start.config(text="DONE", state=tk.DISABLED)
        
        # Get final GAP result
        e, c = self.backend.gap()
        self.lbl_gap_emo.config(text=e)
        self.lbl_gap_conf.config(text=f"Global Avg: {c:.1f}%")
        
        messagebox.showinfo("Result", f"Dominant Emotion: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CoatNetApp(root)
    root.mainloop()