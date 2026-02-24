import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import socket
import time
import math
import numpy as np
import os
import urllib.request
import threading
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

import sys

if getattr(sys, 'frozen', False):
    # รันจากไฟล์ .exe ให้เอาตำแหน่งของไฟล์ exe
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    # รันจากสคริปต์ .py ปกติ
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "face_landmarker.task")
REF_MAP_FILE = os.path.join(SCRIPT_DIR, "face_mesh.png")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
REF_MAP_URL = "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png"

# --- Theme Colors ---
ACCENT = "#3B8ED0"
ACCENT_HOVER = "#36719F"
SUCCESS = "#2FA572"
DANGER = "#E04646"
SURFACE = "#2B2B2B"
SURFACE_LIGHT = "#333333"
TEXT_DIM = "#888888"

# Canonical face mesh UV coordinates (468 points) - normalized 0..1
# We'll load these from MediaPipe's canonical mesh at runtime
CANONICAL_FACE_MESH = None

# MediaPipe landmark indices for iris tracking
# Right eye (from viewer's perspective, mirrored in camera)
EYE_R = {"iris": 468, "inner": 133, "outer": 33, "top": 159, "bottom": 145}
# Left eye
EYE_L = {"iris": 473, "inner": 362, "outer": 263, "top": 386, "bottom": 374}

def get_canonical_mesh():
    """Get canonical face mesh coordinates from MediaPipe for the point picker."""
    global CANONICAL_FACE_MESH
    if CANONICAL_FACE_MESH is not None:
        return CANONICAL_FACE_MESH

    try:
        from mediapipe.python.solutions.face_mesh_connections import (
            FACEMESH_TESSELATION, FACEMESH_CONTOURS,
            FACEMESH_IRISES, FACEMESH_LIPS
        )
    except ImportError:
        FACEMESH_TESSELATION = set()
        FACEMESH_CONTOURS = set()
        FACEMESH_IRISES = set()
        FACEMESH_LIPS = set()

    # Use a static image to extract canonical landmarks
    # We'll create an approximate canonical mesh from known UV positions
    # MediaPipe canonical face model coordinates
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        # Create a simple face image for detection
        # Instead, use a pre-defined set of canonical coordinates
        landmarker.close()
    except:
        pass

    # Predefined canonical face mesh 2D positions (normalized)
    # Based on MediaPipe's canonical face model UV mapping
    # We use approximate positions for the 468 landmarks
    CANONICAL_FACE_MESH = generate_canonical_positions()
    return CANONICAL_FACE_MESH

def generate_canonical_positions():
    """Generate approximate canonical 2D face mesh positions."""
    # This uses the well-known MediaPipe face mesh UV layout
    # Key landmark positions (approximate normalized coords)
    positions = {}

    # We'll generate positions by running face detection on a frontal face
    # For now, use a mathematical approximation of face mesh layout
    num_points = 478  # MediaPipe has 478 landmarks (468 face + 10 iris)

    # Use a predefined layout based on the canonical face model
    # These are approximate 2D positions normalized to 0-1
    center_x, center_y = 0.5, 0.45
    face_w, face_h = 0.35, 0.45

    for i in range(num_points):
        # Default: distribute in a rough oval shape
        angle = (i / num_points) * 2 * math.pi * 3.7  # spiral-ish
        r = 0.15 + 0.2 * (i % 47) / 47.0
        x = center_x + r * math.cos(angle) * (face_w / 0.35)
        y = center_y + r * math.sin(angle) * (face_h / 0.45)
        positions[i] = (max(0.05, min(0.95, x)), max(0.05, min(0.95, y)))

    # Override with known key landmarks
    key_points = {
        # Face oval
        10: (0.5, 0.12), 338: (0.38, 0.13), 297: (0.32, 0.17), 332: (0.28, 0.22),
        284: (0.26, 0.30), 251: (0.25, 0.38), 389: (0.26, 0.46), 356: (0.28, 0.54),
        454: (0.30, 0.48), 323: (0.28, 0.56), 361: (0.30, 0.64), 288: (0.34, 0.70),
        397: (0.38, 0.74), 365: (0.42, 0.77), 379: (0.46, 0.79), 378: (0.48, 0.80),
        152: (0.50, 0.81), 148: (0.52, 0.80), 176: (0.54, 0.79), 149: (0.58, 0.77),
        150: (0.62, 0.74), 136: (0.66, 0.70), 172: (0.70, 0.64), 58: (0.72, 0.56),
        132: (0.72, 0.48), 93: (0.75, 0.46), 234: (0.74, 0.38), 127: (0.74, 0.30),
        162: (0.72, 0.22), 21: (0.68, 0.17), 54: (0.62, 0.13), 103: (0.56, 0.12),

        # Nose
        1: (0.50, 0.45), 2: (0.50, 0.42), 4: (0.50, 0.50), 5: (0.50, 0.40),
        6: (0.50, 0.38), 168: (0.50, 0.30), 195: (0.50, 0.33), 197: (0.50, 0.36),
        19: (0.50, 0.53), 94: (0.50, 0.55),
        # Nose wings
        48: (0.56, 0.52), 278: (0.44, 0.52),
        64: (0.55, 0.50), 294: (0.45, 0.50),
        98: (0.54, 0.54), 327: (0.46, 0.54),

        # Right eye (from viewer's perspective, left in image)
        33: (0.62, 0.32), 246: (0.64, 0.30), 161: (0.66, 0.29), 160: (0.68, 0.29),
        159: (0.69, 0.30), 158: (0.70, 0.31), 157: (0.69, 0.33), 173: (0.67, 0.34),
        133: (0.65, 0.34), 155: (0.63, 0.34), 154: (0.62, 0.33),
        # Right eye iris
        468: (0.66, 0.32), 469: (0.67, 0.31), 470: (0.68, 0.31),
        471: (0.67, 0.33), 472: (0.66, 0.33),

        # Left eye
        263: (0.38, 0.32), 466: (0.36, 0.30), 388: (0.34, 0.29), 387: (0.32, 0.29),
        386: (0.31, 0.30), 385: (0.30, 0.31), 384: (0.31, 0.33), 398: (0.33, 0.34),
        362: (0.35, 0.34), 382: (0.37, 0.34), 381: (0.38, 0.33),
        # Left eye iris
        473: (0.34, 0.32), 474: (0.33, 0.31), 475: (0.32, 0.31),
        476: (0.33, 0.33), 477: (0.34, 0.33),

        # Right eyebrow
        70: (0.60, 0.24), 63: (0.63, 0.22), 105: (0.66, 0.21), 66: (0.69, 0.22),
        107: (0.71, 0.24),

        # Left eyebrow
        300: (0.40, 0.24), 293: (0.37, 0.22), 334: (0.34, 0.21), 296: (0.31, 0.22),
        336: (0.29, 0.24),

        # Outer lips
        61: (0.56, 0.62), 185: (0.55, 0.60), 40: (0.54, 0.59), 39: (0.53, 0.58),
        37: (0.52, 0.58), 0: (0.50, 0.58), 267: (0.48, 0.58), 269: (0.47, 0.58),
        270: (0.46, 0.59), 409: (0.45, 0.60), 291: (0.44, 0.62),
        375: (0.45, 0.65), 321: (0.46, 0.67), 405: (0.47, 0.68), 314: (0.48, 0.69),
        17: (0.50, 0.69), 84: (0.52, 0.69), 181: (0.53, 0.68), 91: (0.54, 0.67),
        146: (0.55, 0.65),

        # Inner lips
        78: (0.55, 0.62), 191: (0.54, 0.61), 80: (0.53, 0.60), 81: (0.52, 0.60),
        82: (0.51, 0.60), 13: (0.50, 0.60), 312: (0.49, 0.60), 311: (0.48, 0.60),
        310: (0.47, 0.60), 415: (0.46, 0.61), 308: (0.45, 0.62),
        324: (0.46, 0.64), 318: (0.47, 0.65), 402: (0.48, 0.65), 317: (0.49, 0.66),
        14: (0.50, 0.66), 87: (0.51, 0.66), 178: (0.52, 0.65), 88: (0.53, 0.65),
        95: (0.54, 0.64),

        # Forehead
        151: (0.50, 0.15), 9: (0.50, 0.18), 8: (0.50, 0.20),

        # Chin
        16: (0.50, 0.76), 15: (0.50, 0.74),
    }

    positions.update(key_points)
    return positions


def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def normalize_value(val, radius_min, radius_max, out_min, out_max):
    if radius_max <= radius_min:
        return out_min
    if val <= radius_min:
        return out_min
    elif val >= radius_max:
        return out_max
    normalized = (val - radius_min) / (radius_max - radius_min)
    return out_min + (out_max - out_min) * normalized

def download_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading MediaPipe Face Landmarker Model to {MODEL_FILE}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Download complete.")


from PIL import Image, ImageTk

# ======================================================================
# Point Picker Window
# ======================================================================
class PointPickerWindow:
    """Interactive window showing captured face mesh for clicking point IDs."""

    def __init__(self, parent_app):
        self.parent = parent_app
        self.win = ctk.CTkToplevel(parent_app.root)
        self.win.title("Face Mesh Point Picker (Live Capture)")
        self.win.geometry("700x750")
        self.win.resizable(True, True)

        self.selected_id = None
        self.hover_id = None
        self.point_radius = 4
        
        self.source_image = None
        if self.parent.latest_image is not None and self.parent.latest_landmarks is not None:
            self.source_image = self.parent.latest_image.copy()
            self.landmarks = self.parent.latest_landmarks
        else:
            self.landmarks = None

        self.canvas_image = None
        self.photo_image = None
        self._screen_points = {}  # {id: (x, y)}
        self.draw_w = 640
        self.draw_h = 480

        self.build_ui()
        if self.source_image is not None:
            self.win.after(100, self.update_canvas)
        else:
            self.show_error_message()

    def build_ui(self):
        # Header
        header = ctk.CTkFrame(self.win, fg_color=SURFACE, corner_radius=0, height=40)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="🔍 Click a point on your face to select its ID",
                      font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=14)

        # Canvas
        canvas_frame = ctk.CTkFrame(self.win, corner_radius=12)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#1a1a2e", highlightthickness=0,
                                 cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        
        if self.source_image is not None:
            self.canvas.bind("<Motion>", self.on_hover)
            self.canvas.bind("<Button-1>", self.on_click)
            self.canvas.bind("<Configure>", lambda e: self.update_canvas())

        # Info bar
        info_frame = ctk.CTkFrame(self.win, corner_radius=12)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        info_inner = ctk.CTkFrame(info_frame, fg_color="transparent")
        info_inner.pack(fill="x", padx=14, pady=10)

        self.id_label = ctk.CTkLabel(info_inner, text="Selected: None",
                                      font=ctk.CTkFont(size=16, weight="bold"),
                                      text_color=ACCENT)
        self.id_label.pack(side="left")

        # Buttons to assign
        btn_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkButton(btn_frame, text="→ X Point A", width=100, height=28,
                       fg_color="#4A4A6A", hover_color="#5A5A8A",
                       font=ctk.CTkFont(size=11),
                       command=lambda: self.assign("x", "pt_a")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_frame, text="→ X Point B", width=100, height=28,
                       fg_color="#4A4A6A", hover_color="#5A5A8A",
                       font=ctk.CTkFont(size=11),
                       command=lambda: self.assign("x", "pt_b")).pack(side="left", padx=(0, 12))
        ctk.CTkButton(btn_frame, text="→ Y Point A", width=100, height=28,
                       fg_color="#4A6A4A", hover_color="#5A8A5A",
                       font=ctk.CTkFont(size=11),
                       command=lambda: self.assign("y", "pt_a")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_frame, text="→ Y Point B", width=100, height=28,
                       fg_color="#4A6A4A", hover_color="#5A8A5A",
                       font=ctk.CTkFont(size=11),
                       command=lambda: self.assign("y", "pt_b")).pack(side="left")

    def show_error_message(self):
        self.canvas.create_text(
            self.canvas.winfo_reqwidth()//2, self.canvas.winfo_reqheight()//2,
            text="⚠️ Camera not active or no face detected.\nPlease make sure the tracker is running\nand looking at your face before opening this picker.",
            fill="#FF5555", font=("Arial", 14, "bold"), justify="center"
        )

    def update_canvas(self):
        if self.source_image is None or not self.landmarks:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        img_h, img_w = self.source_image.shape[:2]
        
        # Calculate aspect ratio preserving scale
        scale = min(w / img_w, h / img_h)
        self.draw_w = int(img_w * scale)
        self.draw_h = int(img_h * scale)
        
        # Center offsets
        self.offset_x = (w - self.draw_w) // 2
        self.offset_y = (h - self.draw_h) // 2

        # Resize image
        resized_img = cv2.resize(self.source_image, (self.draw_w, self.draw_h))
        
        # Optional: darken image slightly to make points pop more
        resized_img = cv2.convertScaleAbs(resized_img, alpha=0.5, beta=0)

        # Convert to PhotoImage
        img_pil = Image.fromarray(resized_img)
        self.photo_image = ImageTk.PhotoImage(image=img_pil)
        
        self.draw_points()

    def draw_points(self):
        self.canvas.delete("all")
        
        if self.photo_image:
            self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_image, anchor="nw")

        self._screen_points = {}

        for idx, lm in enumerate(self.landmarks):
            # Calculate screen coordinates based on scaled image
            sx = int(self.offset_x + (lm.x * self.draw_w))
            sy = int(self.offset_y + (lm.y * self.draw_h))
            self._screen_points[idx] = (sx, sy)

            if idx == self.selected_id:
                color = "#FF3333"
                r = self.point_radius + 2
                self.canvas.create_text(sx + 10, sy - 8, text=str(idx),
                                         fill="#FF6666", font=("Arial", 11, "bold"), anchor="w")
            elif idx == self.hover_id:
                color = "#FFAA00"
                r = self.point_radius + 1
                self.canvas.create_text(sx + 8, sy - 6, text=str(idx),
                                         fill="#FFCC44", font=("Arial", 9, "bold"), anchor="w")
            else:
                color = "#00FF66"
                r = self.point_radius - 2

            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=color, outline="")

    def _find_nearest(self, mx, my):
        closest_id = None
        closest_dist = float('inf')
        # Only attach to points within roughly 15 pixels
        max_dist = 15
        
        for idx, (sx, sy) in self._screen_points.items():
            d = math.sqrt((mx - sx)**2 + (my - sy)**2)
            if d < closest_dist and d < max_dist:
                closest_dist = d
                closest_id = idx
        return closest_id

    def on_hover(self, event):
        if not self._screen_points:
            return
        nearest = self._find_nearest(event.x, event.y)
        if nearest != self.hover_id:
            self.hover_id = nearest
            self.draw_points()

    def on_click(self, event):
        if not self._screen_points:
            return
        nearest = self._find_nearest(event.x, event.y)
        if nearest is not None:
            self.selected_id = nearest
            self.id_label.configure(text=f"Selected: {nearest}")
            self.draw_points()

    def assign(self, axis, field):
        if self.selected_id is None:
            return
        widgets = getattr(self.parent, f"{axis}_widgets", None)
        if widgets:
            widgets[field].set(str(self.selected_id))
            self.parent.show_toast(f"Assigned ID {self.selected_id} to {axis.upper()} {field.replace('pt_', 'Point ')}")


# ======================================================================
# Main Application
# ======================================================================
class FaceTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ShapeKey Face Tracker")
        self.root.geometry("500x850")
        self.root.minsize(460, 750)

        self._populating = False  # Flag to prevent trace callbacks during populate

        self.config = self.load_config()
        if not self.config:
            self.config = {
                "blender_ip": "127.0.0.1",
                "blender_port": 5000,
                "camera_index": 0,
                "draw_mesh": True,
                "groups": {}
            }

        self.running = True
        self.camera_visible = False
        self.target_address = (self.config.get("blender_ip", "127.0.0.1"), self.config.get("blender_port", 5000))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)

        self.groups_data = self.config.get("groups", {})
        self.current_group = tk.StringVar()
        self._last_selected_group = ""

        self.current_x_raw = 0.0
        self.current_y_raw = 0.0
        
        # Live state for picker
        self.latest_image = None
        self.latest_landmarks = None

        # Lerp state
        self.lerp_values = {}  # {group_name: {"x": float, "y": float}}
        
        # Iris EMA smoothing state: {"R": {"x": float, "y": float}, "L": {...}}
        # Separate from per-group lerp — this runs at capture-frame rate on raw iris position.
        self._iris_ema = {"R": {"x": 0.0, "y": 0.0}, "L": {"x": 0.0, "y": 0.0}}

        self.build_ui()

        download_model()

        self.tracker_thread = threading.Thread(target=self.run_tracker_loop, daemon=True)
        self.tracker_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return None
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def save_config(self):
        self.save_current_group_ui()
        self.config["groups"] = self.groups_data
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            self.show_toast("✓ Settings saved!")
        except Exception as e:
            self.show_toast(f"✗ Save failed: {e}", error=True)

    def export_config(self):
        """Export current config to a user-chosen JSON file."""
        self.save_current_group_ui()
        self.config["groups"] = self.groups_data
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Export Config",
            initialdir=SCRIPT_DIR
        )
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.config, f, indent=4)
                self.show_toast(f"✓ Exported to {os.path.basename(path)}")
            except Exception as e:
                self.show_toast(f"✗ Export failed: {e}", error=True)

    def import_config(self):
        """Import config from a JSON file."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Import Config",
            initialdir=SCRIPT_DIR
        )
        if path:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.config = data
                self.groups_data = data.get("groups", {})
                all_groups = list(self.groups_data.keys())
                self.group_combo.configure(values=all_groups if all_groups else [""])
                if all_groups:
                    self.group_combo.set(all_groups[0])
                    self.populate_ui_from_current_group()
                else:
                    self.current_group.set("")
                    self.clear_ui_inputs()
                self.show_toast(f"✓ Imported {len(all_groups)} groups")
            except Exception as e:
                self.show_toast(f"✗ Import failed: {e}", error=True)

    def show_toast(self, message, error=False):
        color = DANGER if error else SUCCESS
        toast = ctk.CTkLabel(
            self.root, text=message,
            fg_color=color, corner_radius=8,
            text_color="white", font=ctk.CTkFont(size=13, weight="bold"),
            height=36
        )
        toast.place(relx=0.5, rely=0.96, anchor="center")
        self.root.after(2000, toast.destroy)

    def build_ui(self):
        # --- Header ---
        header = ctk.CTkFrame(self.root, fg_color=SURFACE, corner_radius=0, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header, text="⬡ ShapeKey Face Tracker",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=16)

        self.show_cam_var = ctk.BooleanVar(value=True)
        self.enable_send_var = ctk.BooleanVar(value=True)
        self.lerp_enabled_var = ctk.BooleanVar(value=False)
        self.lerp_factor_var = tk.DoubleVar(value=0.15)

        # --- Main Scrollable Area ---
        main_scroll = ctk.CTkScrollableFrame(self.root, fg_color="transparent")
        main_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Controls Section ---
        ctrl_frame = ctk.CTkFrame(main_scroll, corner_radius=12)
        ctrl_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(ctrl_frame, text="Controls", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(12, 4))

        ctrl_inner = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        ctrl_inner.pack(fill="x", padx=14, pady=(0, 12))

        # Checkboxes row
        chk_row = ctk.CTkFrame(ctrl_inner, fg_color="transparent")
        chk_row.pack(fill="x", pady=4)
        ctk.CTkCheckBox(chk_row, text="Show Camera", variable=self.show_cam_var,
                         font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 14))
        ctk.CTkCheckBox(chk_row, text="Enable Send", variable=self.enable_send_var,
                         font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 14))

        # Buttons row
        btn_row = ctk.CTkFrame(ctrl_inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=6)
        ctk.CTkButton(btn_row, text="🔍 Point Picker", width=120, height=32,
                       fg_color="#4A4A6A", hover_color="#5A5A8A",
                       command=self.open_point_picker).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="🔄 Fetch", width=80, height=32,
                       fg_color=ACCENT, hover_color=ACCENT_HOVER,
                       command=self.fetch_groups).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="💾 Save", width=70, height=32,
                       fg_color=SUCCESS, hover_color="#248A5E",
                       command=self.save_config).pack(side="left", padx=(0, 4))

        btn_row2 = ctk.CTkFrame(ctrl_inner, fg_color="transparent")
        btn_row2.pack(fill="x", pady=(0, 2))
        ctk.CTkButton(btn_row2, text="📤 Export", width=90, height=30,
                       fg_color=SURFACE_LIGHT, hover_color="#444444",
                       command=self.export_config).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row2, text="📥 Import", width=90, height=30,
                       fg_color=SURFACE_LIGHT, hover_color="#444444",
                       command=self.import_config).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row2, text="📷 Mesh Map", width=100, height=30,
                       fg_color=SURFACE_LIGHT, hover_color="#444444",
                       command=self.open_mesh_map).pack(side="left")

        # --- Group Selection ---
        grp_frame = ctk.CTkFrame(main_scroll, corner_radius=12)
        grp_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(grp_frame, text="Target Group", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(12, 4))

        grp_inner = ctk.CTkFrame(grp_frame, fg_color="transparent")
        grp_inner.pack(fill="x", padx=14, pady=(0, 12))

        grp_row = ctk.CTkFrame(grp_inner, fg_color="transparent")
        grp_row.pack(fill="x")

        existing_groups = list(self.groups_data.keys())
        self.group_combo = ctk.CTkComboBox(
            grp_row, variable=self.current_group,
            values=existing_groups if existing_groups else [""],
            command=self.on_group_selected,
            width=260, height=32,
            font=ctk.CTkFont(size=13)
        )
        self.group_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        if existing_groups:
            self.group_combo.set(existing_groups[0])
            self._last_selected_group = existing_groups[0]

        ctk.CTkButton(grp_row, text="+", width=36, height=32,
                       fg_color=SUCCESS, hover_color="#248A5E",
                       font=ctk.CTkFont(size=16, weight="bold"),
                       command=self.add_manual_group).pack(side="left", padx=(0, 4))
        ctk.CTkButton(grp_row, text="−", width=36, height=32,
                       fg_color=DANGER, hover_color="#C03030",
                       font=ctk.CTkFont(size=16, weight="bold"),
                       command=self.remove_current_group).pack(side="left")

        # --- X Axis Mapping ---
        self.x_frame = self.build_axis_section(main_scroll, "X Axis", "x")
        # --- Y Axis Mapping ---
        self.y_frame = self.build_axis_section(main_scroll, "Y Axis", "y")

        self.populate_ui_from_current_group()

    def build_axis_section(self, parent, title, axis):
        frame = ctk.CTkFrame(parent, corner_radius=12)
        frame.pack(fill="x", pady=(0, 10))

        header_row = ctk.CTkFrame(frame, fg_color="transparent")
        header_row.pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(header_row, text=title, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        out_label = ctk.CTkLabel(header_row, text="0.000",
                                  font=ctk.CTkFont(size=14, weight="bold"),
                                  text_color=ACCENT)
        out_label.pack(side="right")

        raw_label = ctk.CTkLabel(header_row, text="raw: 0.000",
                                  font=ctk.CTkFont(size=11),
                                  text_color=TEXT_DIM)
        raw_label.pack(side="right", padx=(0, 10))
        ctk.CTkLabel(header_row, text="Output:", font=ctk.CTkFont(size=12),
                      text_color=TEXT_DIM).pack(side="right", padx=(0, 6))

        # Mode Selection
        mode_row = ctk.CTkFrame(frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=(0, 6))
        
        mode_var = ctk.StringVar(value="2pt")
        def trace_mode_var(*args):
            m = mode_var.get()
            if m == "iris":
                # Hide manual point entries — preset fills them automatically
                pts_row.pack_forget()
                iris_row.pack(fill="x", pady=4)
            elif m == "1pt":
                iris_row.pack_forget()
                pts_row.pack(fill="x", pady=2)
                pt_a_label.configure(text="Target Pt")
                pt_b_label.configure(text="Origin Pt")
            else:
                iris_row.pack_forget()
                pts_row.pack(fill="x", pady=2)
                pt_a_label.configure(text="Point A")
                pt_b_label.configure(text="Point B")
            if not getattr(self, '_populating', False):
                self.save_current_group_ui()
                
        mode_var.trace_add("write", trace_mode_var)
        
        ctk.CTkSegmentedButton(mode_row, values=["2pt", "1pt", "iris"],
                               variable=mode_var, width=180, height=28,
                               selected_color=ACCENT, selected_hover_color=ACCENT_HOVER).pack(side="left")

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=(0, 6))

        bar = ctk.CTkProgressBar(inner, height=8, corner_radius=4,
                                  progress_color=ACCENT)
        bar.pack(fill="x", pady=(0, 8))
        bar.set(0)

        pts_row = ctk.CTkFrame(inner, fg_color="transparent")
        pts_row.pack(fill="x", pady=2)
        pt_a, pt_a_label = self.create_entry_field(pts_row, "Point A", side="left", width=70, return_label=True)
        pt_b, pt_b_label = self.create_entry_field(pts_row, "Point B", side="left", width=70, return_label=True)

        # Iris mode controls (hidden by default)
        iris_row = ctk.CTkFrame(inner, fg_color="transparent")
        # iris_row starts hidden; trace_mode_var shows it when mode=="iris"

        iris_preset_col = ctk.CTkFrame(iris_row, fg_color="transparent")
        iris_preset_col.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(iris_preset_col, text="Eye Preset:",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(anchor="w")
        preset_btn_row = ctk.CTkFrame(iris_preset_col, fg_color="transparent")
        preset_btn_row.pack(fill="x")

        eye_info_label = ctk.CTkLabel(iris_preset_col,
                                      text="— none —",
                                      font=ctk.CTkFont(size=10),
                                      text_color=TEXT_DIM)
        eye_info_label.pack(anchor="w", pady=(2, 0))

        def fill_eye_preset(eye_dict, label):
            """Auto-fill iris/eye points + ready-to-use defaults for ±1 range.

            rad_min=0.02  → tiny dead zone at center for stability
            rad_max=0.30  → realistic max gaze distance (calibrate with Set Min/Max)
            out_min=0.0   → neutral gaze = 0.0
            out_max=1.0   → max gaze = ±1.0 (sign from direction)
            """
            pt_a.set(str(eye_dict["iris"]))
            pt_b.set(str(eye_dict["outer"]))
            rad_min.set("0.02")
            rad_max.set("0.30")
            out_min.set("0.0")
            out_max.set("1.0")
            sens.set("1.0")
            eye_info_label.configure(text=f"{label} ({eye_dict['iris']}, {eye_dict['outer']})",
                                     text_color=ACCENT)
            if not getattr(self, '_populating', False):
                self.save_current_group_ui()

        ctk.CTkButton(preset_btn_row, text="👁 Right", width=72, height=26,
                      fg_color="#4A4A6A", hover_color="#5A5A8A",
                      font=ctk.CTkFont(size=11),
                      command=lambda: fill_eye_preset(EYE_R, "Right Eye")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(preset_btn_row, text="👁 Left", width=72, height=26,
                      fg_color="#4A4A6A", hover_color="#5A5A8A",
                      font=ctk.CTkFont(size=11),
                      command=lambda: fill_eye_preset(EYE_L, "Left Eye")).pack(side="left")

        exp_col = ctk.CTkFrame(iris_row, fg_color="transparent")
        exp_col.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(exp_col, text="Exp Power:",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(anchor="w")
        exp_row = ctk.CTkFrame(exp_col, fg_color="transparent")
        exp_row.pack(fill="x")
        exp_power_var = tk.DoubleVar(value=1.2)
        def trace_exp(*args):
            exp_val_label.configure(text=f"{exp_power_var.get():.2f}")
            if not getattr(self, '_populating', False):
                self.save_current_group_ui()
        exp_power_var.trace_add("write", trace_exp)
        ctk.CTkSlider(exp_row, from_=0.5, to=3.0, variable=exp_power_var,
                      width=100, height=14).pack(side="left", padx=(0, 6))
        exp_val_label = ctk.CTkLabel(exp_row, text="1.20",
                                     font=ctk.CTkFont(size=11), text_color=ACCENT)
        exp_val_label.pack(side="left")

        rad_row = ctk.CTkFrame(inner, fg_color="transparent")
        rad_row.pack(fill="x", pady=2)

        rad_min = self.create_entry_field(rad_row, "Radius Min", side="left", width=90)
        ctk.CTkButton(
            rad_row, text="Set Min", width=64, height=28,
            fg_color="#4A4A6A", hover_color="#5A5A8A",
            font=ctk.CTkFont(size=11),
            command=lambda: self.calibrate_radius(axis, "min")
        ).pack(side="left", padx=(4, 10))

        rad_max = self.create_entry_field(rad_row, "Radius Max", side="left", width=90)
        ctk.CTkButton(
            rad_row, text="Set Max", width=64, height=28,
            fg_color="#4A6A4A", hover_color="#5A8A5A",
            font=ctk.CTkFont(size=11),
            command=lambda: self.calibrate_radius(axis, "max")
        ).pack(side="left", padx=(4, 0))

        out_row = ctk.CTkFrame(inner, fg_color="transparent")
        out_row.pack(fill="x", pady=2)
        out_min = self.create_entry_field(out_row, "Out Min", side="left")
        out_max = self.create_entry_field(out_row, "Out Max", side="left")
        sens = self.create_entry_field(out_row, "Sens", side="left", width=60)

        # Lerp per axis
        lerp_row = ctk.CTkFrame(inner, fg_color="transparent")
        lerp_row.pack(fill="x", pady=6)
        
        lerp_var = ctk.BooleanVar(value=False)
        def trace_lerp_bool(*args):
            if not getattr(self, '_populating', False):
                self.save_current_group_ui()
        lerp_var.trace_add("write", trace_lerp_bool)
        
        ctk.CTkCheckBox(lerp_row, text="Lerp Smooth", variable=lerp_var,
                         font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 10))
                         
        ctk.CTkLabel(lerp_row, text="Factor:", font=ctk.CTkFont(size=11),
                      text_color=TEXT_DIM).pack(side="left", padx=(0, 4))
                      
        lerp_fac_var = tk.DoubleVar(value=0.15)
        def trace_lerp_fac(*args):
            lerp_val_label.configure(text=f"{lerp_fac_var.get():.2f}")
            if not getattr(self, '_populating', False):
                self.save_current_group_ui()
        lerp_fac_var.trace_add("write", trace_lerp_fac)
        
        lerp_slider = ctk.CTkSlider(lerp_row, from_=0.02, to=0.5,
                                    variable=lerp_fac_var,
                                    width=100, height=14)
        lerp_slider.pack(side="left", padx=(0, 6))
        
        lerp_val_label = ctk.CTkLabel(lerp_row, text="0.15",
                                      font=ctk.CTkFont(size=11), text_color=ACCENT)
        lerp_val_label.pack(side="left")

        ctk.CTkFrame(frame, fg_color="transparent", height=4).pack()

        data = {
            "mode": mode_var,
            "pt_a": pt_a, "pt_b": pt_b,
            "exp_power": exp_power_var,
            "rad_min": rad_min, "rad_max": rad_max,
            "out_min": out_min, "out_max": out_max,
            "sens": sens,
            "lerp_var": lerp_var, "lerp_fac": lerp_fac_var,
            "out_label": out_label, "raw_label": raw_label, "bar": bar
        }
        setattr(self, f"{axis}_widgets", data)
        return frame

    def create_entry_field(self, parent, label, side="left", width=80, return_label=False):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(side=side, padx=(0, 6), fill="x", expand=True)

        lbl = ctk.CTkLabel(container, text=label, font=ctk.CTkFont(size=11),
                      text_color=TEXT_DIM)
        lbl.pack(anchor="w")
        var = tk.StringVar()

        def trace_var(*args):
            if not getattr(self, '_populating', False):
                self.save_current_group_ui()
        var.trace_add("write", trace_var)

        entry = ctk.CTkEntry(container, textvariable=var, height=28, width=width,
                              font=ctk.CTkFont(size=12))
        entry.pack(fill="x")
        if return_label:
            return var, lbl
        return var

    def calibrate_radius(self, axis, which):
        if axis == "x":
            raw = self.current_x_raw
            widgets = self.x_widgets
        else:
            raw = self.current_y_raw
            widgets = self.y_widgets

        if raw > 0:
            if which == "min":
                widgets["rad_min"].set(f"{raw:.4f}")
            else:
                widgets["rad_max"].set(f"{raw:.4f}")
            self.save_current_group_ui()
            self.show_toast(f"✓ {axis.upper()} Radius {'Min' if which == 'min' else 'Max'} = {raw:.4f}")

    def open_point_picker(self):
        PointPickerWindow(self)

    def fetch_groups(self):
        try:
            req = json.dumps({"type": "GET_GROUPS"})
            self.sock.sendto(req.encode('utf-8'), self.target_address)
            data, addr = self.sock.recvfrom(2048)
            resp = json.loads(data.decode('utf-8'))
            if resp.get("type") == "GROUPS":
                groups = resp.get("groups", [])
                self.merge_groups(groups)
                self.show_toast(f"✓ Fetched {len(groups)} groups")
        except socket.timeout:
            self.show_toast("✗ No response from Blender", error=True)
        except Exception as e:
            self.show_toast(f"✗ Error: {e}", error=True)

    def merge_groups(self, groups_list):
        # Save current group UI first
        self.save_current_group_ui()

        missing = [g for g in list(self.groups_data.keys()) if g not in groups_list]
        for m in missing:
            del self.groups_data[m]
        for g in groups_list:
            if g not in self.groups_data:
                self.groups_data[g] = {}

        all_groups = list(self.groups_data.keys())
        self.group_combo.configure(values=all_groups if all_groups else [""])
        if all_groups:
            if self.current_group.get() not in all_groups:
                self.group_combo.set(all_groups[0])
            self.populate_ui_from_current_group()
        else:
            self.current_group.set("")
            self.clear_ui_inputs()

    def save_current_group_ui(self, group_override=None):
        if self._populating:
            return

        group = group_override if group_override is not None else self.current_group.get()
        if not group or group not in self.groups_data:
            return

        def safe_int(v):
            return int(v) if str(v).strip().isdigit() else None
        def safe_float(v):
            try: return float(v)
            except: return 0.0

        for axis in ["x", "y"]:
            widgets = getattr(self, f"{axis}_widgets", None)
            if widgets is None:
                continue

            if axis not in self.groups_data[group]:
                self.groups_data[group][axis] = {}

            axis_data = self.groups_data[group][axis]

            axis_data["mode"] = widgets["mode"].get()

            a_val = safe_int(widgets["pt_a"].get())
            b_val = safe_int(widgets["pt_b"].get())
            
            if a_val is not None:
                axis_data["point_a"] = a_val
            else:
                axis_data.pop("point_a", None)
                
            if b_val is not None:
                axis_data["point_b"] = b_val
            else:
                axis_data.pop("point_b", None)

            axis_data["exp_power"] = float(widgets["exp_power"].get())
            axis_data["radius_min"] = safe_float(widgets["rad_min"].get())
            axis_data["radius_max"] = safe_float(widgets["rad_max"].get())
            axis_data["out_min"] = safe_float(widgets["out_min"].get())
            axis_data["out_max"] = safe_float(widgets["out_max"].get())
            axis_data["sens"] = safe_float(widgets["sens"].get())
            if axis_data["sens"] == 0.0:
                axis_data["sens"] = 1.0
                
            axis_data["lerp_en"] = widgets["lerp_var"].get()
            axis_data["lerp_fac"] = widgets["lerp_fac"].get()

    def on_group_selected(self, choice):
        # Save the PREVIOUS group before switching
        if self._last_selected_group:
            self.save_current_group_ui(group_override=self._last_selected_group)
        self._last_selected_group = choice
        self.populate_ui_from_current_group()

    def populate_ui_from_current_group(self):
        group = self.current_group.get()
        self._last_selected_group = group
        if not group or group not in self.groups_data:
            self.clear_ui_inputs()
            return

        self._populating = True  # Prevent trace callbacks
        try:
            data = self.groups_data[group]

            for axis in ["x", "y"]:
                widgets = getattr(self, f"{axis}_widgets", None)
                if widgets is None:
                    continue

                axis_data = data.get(axis) or {}
                # Graceful fallback: old "box" mode → "2pt"
                saved_mode = str(axis_data.get("mode", "2pt"))
                if saved_mode == "box":
                    saved_mode = "2pt"
                widgets["mode"].set(saved_mode)
                widgets["pt_a"].set(str(axis_data.get("point_a", "")) if axis_data.get("point_a") is not None else "")
                widgets["pt_b"].set(str(axis_data.get("point_b", "")) if axis_data.get("point_b") is not None else "")
                widgets["exp_power"].set(float(axis_data.get("exp_power", 1.2)))
                widgets["rad_min"].set(str(axis_data.get("radius_min", axis_data.get("min", "0.0"))))
                widgets["rad_max"].set(str(axis_data.get("radius_max", axis_data.get("max", "1.0"))))
                widgets["out_min"].set(str(axis_data.get("out_min", "0.0")))
                widgets["out_max"].set(str(axis_data.get("out_max", "1.0")))
                widgets["sens"].set(str(axis_data.get("sens", "1.0")))
                widgets["lerp_var"].set(axis_data.get("lerp_en", False))
                widgets["lerp_fac"].set(float(axis_data.get("lerp_fac", 0.15)))
        finally:
            self._populating = False

    def clear_ui_inputs(self):
        self._populating = True
        self._last_selected_group = ""
        try:
            for axis in ["x", "y"]:
                widgets = getattr(self, f"{axis}_widgets", None)
                if widgets is None:
                    continue
                widgets["mode"].set("2pt")
                for key in ["pt_a", "pt_b", "rad_min", "rad_max", "out_min", "out_max", "sens"]:
                    widgets[key].set("")
                widgets["exp_power"].set(1.2)
                widgets["lerp_var"].set(False)
                widgets["lerp_fac"].set(0.15)
                widgets["out_label"].configure(text="0.000")
                widgets["raw_label"].configure(text="raw: 0.000")
                widgets["bar"].set(0)
        finally:
            self._populating = False

    def add_manual_group(self):
        dialog = ctk.CTkInputDialog(text="Enter Group Name (e.g. Object_GroupName):", title="Add Group")
        name = dialog.get_input()
        if name and name.strip():
            name = name.strip()
            if name not in self.groups_data:
                self.save_current_group_ui()
                self.groups_data[name] = {}
                all_groups = list(self.groups_data.keys())
                self.group_combo.configure(values=all_groups)
                self.group_combo.set(name)
                self.populate_ui_from_current_group()

    def remove_current_group(self):
        grp = self.current_group.get()
        if grp and grp in self.groups_data:
            confirm = ctk.CTkToplevel(self.root)
            confirm.title("Confirm Delete")
            confirm.geometry("320x130")
            confirm.resizable(False, False)
            confirm.grab_set()

            ctk.CTkLabel(confirm, text=f"Delete '{grp}'?",
                          font=ctk.CTkFont(size=14)).pack(pady=(20, 10))
            btn_row = ctk.CTkFrame(confirm, fg_color="transparent")
            btn_row.pack(pady=10)

            def do_delete():
                del self.groups_data[grp]
                all_groups = list(self.groups_data.keys())
                self.group_combo.configure(values=all_groups if all_groups else [""])
                if all_groups:
                    self.group_combo.set(all_groups[0])
                    self.populate_ui_from_current_group()
                else:
                    self.current_group.set("")
                    self.clear_ui_inputs()
                confirm.destroy()

            ctk.CTkButton(btn_row, text="Delete", fg_color=DANGER, hover_color="#C03030",
                           width=100, command=do_delete).pack(side="left", padx=10)
            ctk.CTkButton(btn_row, text="Cancel", fg_color=SURFACE_LIGHT,
                           width=100, command=confirm.destroy).pack(side="left", padx=10)

    def open_mesh_map(self):
        if not os.path.exists(REF_MAP_FILE):
            print("Downloading Reference Mesh Map...")
            try:
                import urllib.request
                req = urllib.request.Request(REF_MAP_URL, headers={'User-Agent': 'Mozilla/5.0'})
                with open(REF_MAP_FILE, 'wb') as f:
                    f.write(urllib.request.urlopen(req).read())
            except Exception as e:
                self.show_toast(f"✗ Failed to download map: {e}", error=True)
                return
        try:
            if os.name == 'nt':
                os.startfile(REF_MAP_FILE)
            else:
                import subprocess, sys
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, REF_MAP_FILE])
        except Exception as e:
            self.show_toast(f"✗ Failed to open map: {e}", error=True)

    def run_tracker_loop(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(self.config.get("camera_index", 0))
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        window_name = 'ShapeKey Face Tracker - Preview'
        window_created = False

        while self.running and cap.isOpened():
            success, image = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            image = cv2.flip(image, 1)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Store latest for point picker
            self.latest_image = rgb_frame.copy()

            payload = {}
            if results.face_landmarks and len(results.face_landmarks) > 0:
                face_landmarks = results.face_landmarks[0]
                self.latest_landmarks = face_landmarks
                pt_left = face_landmarks[234]
                pt_left = face_landmarks[234]
                pt_right = face_landmarks[454]
                face_width = calculate_distance(pt_left, pt_right)
                if face_width <= 0: face_width = 1.0
                
                # Pre-calculate Face Local Axes for 1-point tracking projection
                # Face X Axis (Right ear to Left ear)
                fx_vec = (pt_right.x - pt_left.x, pt_right.y - pt_left.y, pt_right.z - pt_left.z)
                fx_len_sq = fx_vec[0]**2 + fx_vec[1]**2 + fx_vec[2]**2
                if fx_len_sq <= 0: fx_len_sq = 1.0
                
                # Face Y Axis (Chin to Forehead)
                pt_top = face_landmarks[10]
                pt_bottom = face_landmarks[152]
                fy_vec = (pt_bottom.x - pt_top.x, pt_bottom.y - pt_top.y, pt_bottom.z - pt_top.z)
                fy_len_sq = fy_vec[0]**2 + fy_vec[1]**2 + fy_vec[2]**2
                if fy_len_sq <= 0: fy_len_sq = 1.0

                draw_points_set = set()

                snapshot = dict(self.groups_data)

                for group_name, mappings in snapshot.items():
                    group_data = {}
                    if mappings.get("x") and mappings["x"].get("point_a") is not None and mappings["x"].get("point_b") is not None:
                        x_map = mappings["x"]
                        try:
                            mode = x_map.get("mode", "2pt")

                            if mode == "iris":
                                # Iris Local Projection Mode — X axis
                                pt_b_idx = int(x_map["point_b"])
                                eye = EYE_L if pt_b_idx == EYE_L["outer"] else EYE_R
                                eye_key = "L" if pt_b_idx == EYE_L["outer"] else "R"
                                p_iris  = face_landmarks[eye["iris"]]
                                p_inner = face_landmarks[eye["inner"]]
                                p_outer = face_landmarks[eye["outer"]]
                                p_top   = face_landmarks[eye["top"]]
                                p_bot   = face_landmarks[eye["bottom"]]
                                # Eye-local X axis vector (outer → inner)
                                ex = (p_inner.x - p_outer.x,
                                      p_inner.y - p_outer.y,
                                      p_inner.z - p_outer.z)
                                ex_len = math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)
                                if ex_len <= 0: ex_len = 1.0
                                eye_width = ex_len
                                # --- Blink guard: skip this frame if eye mostly closed ---
                                eye_h = abs(p_top.y - p_bot.y)
                                if eye_h < 0.25 * eye_width:
                                    # Eye is closing/blinking — reuse last smoothed value
                                    raw_val = self._iris_ema[eye_key]["x"]
                                else:
                                    # Eye center (3D midpoints)
                                    cx = (p_inner.x + p_outer.x) / 2.0
                                    cy = (p_top.y   + p_bot.y)   / 2.0
                                    # Project iris offset onto eye-local X axis
                                    delta = (p_iris.x - cx, p_iris.y - cy,
                                             p_iris.z - (p_inner.z + p_outer.z) / 2.0)
                                    dot = delta[0]*ex[0] + delta[1]*ex[1] + delta[2]*ex[2]
                                    inst = dot / (ex_len * eye_width)
                                    # EMA smoothing (α=0.30 — fast enough, kills jitter)
                                    ema = self._iris_ema[eye_key]
                                    ema["x"] += 0.30 * (inst - ema["x"])
                                    raw_val = ema["x"]
                                # Exponential sensitivity curve (preserves sign)
                                exp_p = float(x_map.get("exp_power", 1.2))
                                raw_val = math.copysign(abs(raw_val) ** exp_p, raw_val)
                                draw_points_set.update([eye["iris"], eye["inner"], eye["outer"], eye["top"], eye["bottom"]])
                            else:
                                pA = face_landmarks[int(x_map["point_a"])]
                                pB = face_landmarks[int(x_map["point_b"])]
                                if mode == "1pt":
                                    # 3D Vector from Origin (B) to Target (A)
                                    target_vec = (pA.x - pB.x, pA.y - pB.y, pA.z - pB.z)
                                    dot_x = (target_vec[0] * fx_vec[0]) + (target_vec[1] * fx_vec[1]) + (target_vec[2] * fx_vec[2])
                                    proj_x = dot_x / math.sqrt(fx_len_sq) / face_width
                                    raw_val = proj_x * 10.0
                                else:
                                    dist = calculate_distance(pA, pB)
                                    norm_dist = dist / face_width
                                    raw_val = norm_dist
                                draw_points_set.add(int(x_map["point_a"]))
                                draw_points_set.add(int(x_map["point_b"]))

                            if group_name == self.current_group.get():
                                self.current_x_raw = raw_val

                            sens = float(x_map.get("sens", 1.0))
                            
                            if mode in ("1pt", "iris"):
                                rad_min = float(x_map.get("radius_min", x_map.get("min", 0.0)))
                                rad_max = float(x_map.get("radius_max", x_map.get("max", 1.0)))
                                out_min = float(x_map.get("out_min", 0.0))
                                out_max = float(x_map.get("out_max", 1.0))
                                abs_dist = abs(raw_val)
                                if abs_dist <= rad_min or rad_max <= rad_min:
                                    mapped_mag = out_min
                                elif abs_dist >= rad_max:
                                    mapped_mag = out_max
                                else:
                                    normalized = (abs_dist - rad_min) / (rad_max - rad_min)
                                    mapped_mag = out_min + (out_max - out_min) * normalized
                                val = mapped_mag * sens * (1.0 if raw_val >= 0 else -1.0)
                            else:
                                val = normalize_value(
                                    raw_val,
                                    float(x_map.get("radius_min", x_map.get("min", 0.0))),
                                    float(x_map.get("radius_max", x_map.get("max", 1.0))),
                                    float(x_map.get("out_min", 0.0)),
                                    float(x_map.get("out_max", 1.0))
                                ) * sens
                                
                            group_data["x"] = max(-1.0, min(1.0, val))
                        except (IndexError, ValueError, TypeError): pass

                    if mappings.get("y") and mappings["y"].get("point_a") is not None and mappings["y"].get("point_b") is not None:
                        y_map = mappings["y"]
                        try:
                            mode = y_map.get("mode", "2pt")

                            if mode == "iris":
                                # Iris Local Projection Mode — Y axis
                                pt_b_idx = int(y_map["point_b"])
                                eye = EYE_L if pt_b_idx == EYE_L["outer"] else EYE_R
                                eye_key = "L" if pt_b_idx == EYE_L["outer"] else "R"
                                p_iris  = face_landmarks[eye["iris"]]
                                p_inner = face_landmarks[eye["inner"]]
                                p_outer = face_landmarks[eye["outer"]]
                                p_top   = face_landmarks[eye["top"]]
                                p_bot   = face_landmarks[eye["bottom"]]
                                # Eye width for scale-invariant normalization
                                ex = (p_inner.x - p_outer.x,
                                      p_inner.y - p_outer.y,
                                      p_inner.z - p_outer.z)
                                eye_width = math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)
                                if eye_width <= 0: eye_width = 1.0
                                # --- Blink guard ---
                                eye_h = abs(p_top.y - p_bot.y)
                                if eye_h < 0.25 * eye_width:
                                    raw_val = self._iris_ema[eye_key]["y"]
                                else:
                                    cy = (p_top.y + p_bot.y) / 2.0
                                    inst = (p_iris.y - cy) / eye_width
                                    # EMA smoothing
                                    ema = self._iris_ema[eye_key]
                                    ema["y"] += 0.30 * (inst - ema["y"])
                                    raw_val = ema["y"]
                                # Exponential sensitivity
                                exp_p = float(y_map.get("exp_power", 1.2))
                                raw_val = math.copysign(abs(raw_val) ** exp_p, raw_val)
                                draw_points_set.update([eye["iris"], eye["inner"], eye["outer"], eye["top"], eye["bottom"]])
                            else:
                                pA = face_landmarks[int(y_map["point_a"])]
                                pB = face_landmarks[int(y_map["point_b"])]
                                if mode == "1pt":
                                    target_vec = (pA.x - pB.x, pA.y - pB.y, pA.z - pB.z)
                                    dot_y = (target_vec[0] * fy_vec[0]) + (target_vec[1] * fy_vec[1]) + (target_vec[2] * fy_vec[2])
                                    proj_y = dot_y / math.sqrt(fy_len_sq) / face_width
                                    raw_val = proj_y * 10.0
                                else:
                                    dist = calculate_distance(pA, pB)
                                    norm_dist = dist / face_width
                                    raw_val = norm_dist
                                draw_points_set.add(int(y_map["point_a"]))
                                draw_points_set.add(int(y_map["point_b"]))

                            if group_name == self.current_group.get():
                                self.current_y_raw = raw_val

                            sens = float(y_map.get("sens", 1.0))
                            
                            if mode in ("1pt", "iris"):
                                rad_min = float(y_map.get("radius_min", y_map.get("min", 0.0)))
                                rad_max = float(y_map.get("radius_max", y_map.get("max", 1.0)))
                                out_min = float(y_map.get("out_min", 0.0))
                                out_max = float(y_map.get("out_max", 1.0))
                                abs_dist = abs(raw_val)
                                if abs_dist <= rad_min or rad_max <= rad_min:
                                    mapped_mag = out_min
                                elif abs_dist >= rad_max:
                                    mapped_mag = out_max
                                else:
                                    normalized = (abs_dist - rad_min) / (rad_max - rad_min)
                                    mapped_mag = out_min + (out_max - out_min) * normalized
                                val = mapped_mag * sens * (1.0 if raw_val >= 0 else -1.0)
                            else:
                                val = normalize_value(
                                    raw_val,
                                    float(y_map.get("radius_min", y_map.get("min", 0.0))),
                                    float(y_map.get("radius_max", y_map.get("max", 1.0))),
                                    float(y_map.get("out_min", 0.0)),
                                    float(y_map.get("out_max", 1.0))
                                ) * sens
                                
                            group_data["y"] = max(-1.0, min(1.0, val))
                        except (IndexError, ValueError, TypeError): pass

                    if group_data:
                        if "x" not in group_data: group_data["x"] = 0.0
                        if "y" not in group_data: group_data["y"] = 0.0

                        # Apply per-axis lerp smoothing
                        if group_name not in self.lerp_values:
                            self.lerp_values[group_name] = {"x": 0.0, "y": 0.0}
                        lv = self.lerp_values[group_name]
                        
                        # X axis lerp
                        x_map = mappings.get("x", {})
                        if x_map.get("lerp_en", False):
                            factor = float(x_map.get("lerp_fac", 0.15))
                            lv["x"] += (group_data["x"] - lv["x"]) * factor
                            group_data["x"] = lv["x"]
                        else:
                            lv["x"] = group_data["x"]  # Keep state updated
                            
                        # Y axis lerp
                        y_map = mappings.get("y", {})
                        if y_map.get("lerp_en", False):
                            factor = float(y_map.get("lerp_fac", 0.15))
                            lv["y"] += (group_data["y"] - lv["y"]) * factor
                            group_data["y"] = lv["y"]
                        else:
                            lv["y"] = group_data["y"]  # Keep state updated

                        payload[group_name] = group_data

                        if group_name == self.current_group.get():
                            xr = self.current_x_raw
                            yr = self.current_y_raw
                            self.root.after(0, lambda x=group_data["x"], y=group_data["y"], xr=xr, yr=yr: self.update_output_labels(x, y, xr, yr))

                if self.config.get("draw_mesh", True):
                    h, w, _ = image.shape
                    if not self.show_cam_var.get():
                        image = np.zeros((h, w, 3), dtype=np.uint8)
                    for i, lm in enumerate(face_landmarks):
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        if i in draw_points_set:
                            cv2.circle(image, (x_px, y_px), 4, (0, 0, 255), -1)
                        else:
                            cv2.circle(image, (x_px, y_px), 1, (0, 255, 0), -1)

            # Only send if enabled
            if payload and self.enable_send_var.get():
                try:
                    self.sock.sendto(json.dumps(payload).encode('utf-8'), self.target_address)
                except Exception:
                    pass

            if not window_created:
                cv2.namedWindow(window_name)
                window_created = True
            cv2.imshow(window_name, image)
            cv2.waitKey(5)

        cap.release()
        if window_created:
            cv2.destroyWindow(window_name)
        landmarker.close()

    def update_output_labels(self, x, y, x_raw=0, y_raw=0):
        if hasattr(self, 'x_widgets'):
            self.x_widgets["out_label"].configure(text=f"{x:.3f}")
            self.x_widgets["raw_label"].configure(text=f"raw: {x_raw:.4f}")
            self.x_widgets["bar"].set(max(0, min(1, abs(x))))
        if hasattr(self, 'y_widgets'):
            self.y_widgets["out_label"].configure(text=f"{y:.3f}")
            self.y_widgets["raw_label"].configure(text=f"raw: {y_raw:.4f}")
            self.y_widgets["bar"].set(max(0, min(1, abs(y))))

    def on_close(self):
        self.running = False
        self.save_current_group_ui()
        self.config["groups"] = self.groups_data
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except:
            pass
        self.sock.close()
        self.root.destroy()


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = FaceTrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
