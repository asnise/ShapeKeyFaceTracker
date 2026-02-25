import dearpygui.dearpygui as dpg
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import json
import os
import socket
import math
import urllib.request
import logging
import sys
import shutil
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Resource Path Handling ---
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Paths for writing (next to EXE or Script)
BASE_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MODEL_FILE = os.path.join(BASE_DIR, "face_landmarker.task")
REF_MAP_FILE = os.path.join(BASE_DIR, "face_mesh.png")
LOG_FILE = os.path.join(BASE_DIR, "tracker_log.txt")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# URLs (Fallbacks)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
REF_MAP_URL = "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png"
# MediaPipe landmark indices for iris tracking

# MediaPipe landmark indices for iris tracking
EYE_R = {"iris": 468, "inner": 133, "outer": 33, "top": 159, "bottom": 145}
EYE_L = {"iris": 473, "inner": 362, "outer": 263, "top": 386, "bottom": 374}

# --- Default Presets (Embedded) ---
DEFAULT_PRESETS = {
    "Mouth Standard": {
        "x": {"radius_min": 0.2935, "radius_max": 0.3967, "out_min": -1.0, "out_max": 1.0, "sens": 1.0, "point_a": 291, "point_b": 61, "mode": "2pt (Dist)", "exp_power": 1.2, "lerp_en": False, "lerp_fac": 0.15},
        "y": {"radius_min": 0.0625, "radius_max": 0.4397, "out_min": -1.0, "out_max": 1.0, "sens": 1.25, "point_a": 0, "point_b": 16, "mode": "2pt (Dist)", "exp_power": 1.2, "lerp_en": False, "lerp_fac": 0.15}
    },
    "Eyes Tracking": {
        "x": {"radius_min": 0.02, "radius_max": 0.25, "out_min": 0.0, "out_max": 1.0, "sens": 1.0, "point_a": 468, "point_b": 33, "mode": "iris", "exp_power": 0.75, "lerp_en": False, "lerp_fac": 0.15},
        "y": {"radius_min": 0.02, "radius_max": 0.25, "out_min": 0.0, "out_max": -1.0, "sens": 1.0, "point_a": 473, "point_b": 386, "mode": "iris", "exp_power": 1.2, "lerp_en": False, "lerp_fac": 0.15}
    },
    "Eye Blink (Y)": {
        "x": {"radius_min": 0.02, "radius_max": 0.25, "out_min": 0.0, "out_max": 1.0, "sens": 1.0, "point_a": 0, "point_b": 0, "mode": "None", "exp_power": 1.2, "lerp_en": False, "lerp_fac": 0.15},
        "y": {"radius_min": 0.0415, "radius_max": 0.1083, "out_min": -1.0, "out_max": 0.0, "sens": 1.5, "point_a": 374, "point_b": 475, "mode": "2pt (Dist)", "exp_power": 0.816, "lerp_en": False, "lerp_fac": 0.01}
    }
}

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def normalize_value(val, radius_min, radius_max, out_min, out_max):
    if radius_max <= radius_min: return out_min
    if val <= radius_min: return out_min
    elif val >= radius_max: return out_max
    normalized = (val - radius_min) / (radius_max - radius_min)
    return out_min + (out_max - out_min) * normalized

class FaceTrackerAppDPG:
    def __init__(self):
        self.running = True
        self.camera_show = True # For Privacy
        self.camera_active = True # For Backend
        self.send_enabled = True
        self.draw_mesh = True
        self.initialized = False
        
        self.latest_landmarks = None
        self.active_points_x = set()
        self.active_points_y = set()
        self.hover_id = -1
        self.selected_id = -1
        self.fps = 0
        
        # State Data
        self.config = self.load_config()
        self.groups_data = self.config.get("groups", {})
        self.current_group_name = ""
        self.lerp_values = {}
        self._iris_ema = {"R": {"x": 0.0, "y": 0.0}, "L": {"x": 0.0, "y": 0.0}}
        self.current_vals = {"x": 0.0, "y": 0.0, "rx": 0.0, "ry": 0.0}
        
        # Network
        self.target_address = (self.config.get("blender_ip", "127.0.0.1"), self.config.get("blender_port", 5000))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        
        self._ensure_assets()
        
        # DPG UI Setup
        dpg.create_context()
        self.setup_textures()
        self.build_ui()
        self.setup_theme()
        
        # Global Event Handlers
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._handle_click)
        
        dpg.create_viewport(title="ShapeKey Face Tracker (DPG High-Performance)", width=1400, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        dpg.set_viewport_resize_callback(self._on_resize)
        
        self.tracker_thread = threading.Thread(target=self.run_tracker_loop, daemon=True)
        self.tracker_thread.start()

    def _ensure_assets(self):
        # 1. Try to unpack from EXE (MEIPASS) if they don't exist locally
        for filename in ["face_landmarker.task", "face_mesh.png"]:
            local_path = os.path.join(BASE_DIR, filename)
            if not os.path.exists(local_path):
                bundled_path = get_resource_path(filename)
                if os.path.exists(bundled_path):
                    logger.info(f"Unpacking bundled asset: {filename}")
                    shutil.copy2(bundled_path, local_path)

        # 2. If still missing, download (emergency fallback)
        if not os.path.exists(MODEL_FILE):
            logger.info("Model missing and not bundled. Downloading...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        if not os.path.exists(REF_MAP_FILE):
            try:
                logger.info("Mesh map missing and not bundled. Downloading...")
                req = urllib.request.Request(REF_MAP_URL, headers={'User-Agent': 'Mozilla/5.0'})
                with open(REF_MAP_FILE, 'wb') as f:
                    f.write(urllib.request.urlopen(req).read())
            except Exception as e:
                logger.error(f"Failed to download mesh map: {e}")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except: pass
        return {"blender_ip": "127.0.0.1", "blender_port": 5000, "camera_index": 0, "groups": {}}

    def save_config(self):
        self._sync_ui_to_data()
        self.config["groups"] = self.groups_data
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            self._show_toast("✓ All Settings Saved", (0, 255, 120))
        except Exception as e:
            self._show_toast(f"✗ Save Error: {e}", (255, 50, 50))

    def _show_toast(self, message, color):
        dpg.set_value("toast_text", message)
        dpg.configure_item("toast_text", color=color)
        threading.Timer(2.0, lambda: dpg.set_value("toast_text", "")).start()

    def _on_resize(self):
        if not dpg.is_dearpygui_running(): return
        w = dpg.get_viewport_width()
        h = dpg.get_viewport_height()
        
        # Threshold for switching to vertical stack
        is_narrow = w < 1250
        
        # Toggle Layout Mode
        if dpg.does_item_exist("main_layout_group"):
            dpg.configure_item("main_layout_group", horizontal=not is_narrow)

        # Calculate dynamic heights
        if not is_narrow:
            # DESKTOP MODE (3 Columns)
            main_content_h = h - 60
            dpg.configure_item("left_col", width=450, height=main_content_h)
            dpg.configure_item("mid_col", width=660, height=main_content_h)
            dpg.configure_item("right_col", width=-1, height=main_content_h)
            
            if dpg.does_item_exist("inspector_scroll"): dpg.configure_item("inspector_scroll", height=main_content_h - 260)
            if dpg.does_item_exist("cam_window"): dpg.configure_item("cam_window", height=main_content_h - 20)
            if dpg.does_item_exist("log_child_window"): dpg.configure_item("log_child_window", height=main_content_h - 60)
        else:
            # MOBILE/NARROW MODE (Vertical Stack)
            dpg.configure_item("left_col", width=-1, height=500)
            dpg.configure_item("mid_col", width=-1, height=800)
            dpg.configure_item("right_col", width=-1, height=300)
            
            if dpg.does_item_exist("inspector_scroll"): dpg.configure_item("inspector_scroll", height=240)
            if dpg.does_item_exist("cam_window"): dpg.configure_item("cam_window", height=780)
            if dpg.does_item_exist("log_child_window"): dpg.configure_item("log_child_window", height=220)

    def fetch_groups(self):
        try:
            req = json.dumps({"type": "GET_GROUPS"})
            self.sock.sendto(req.encode('utf-8'), self.target_address)
            data, addr = self.sock.recvfrom(2048)
            resp = json.loads(data.decode('utf-8'))
            if resp.get("type") == "GROUPS":
                groups = resp.get("groups", [])
                self._merge_groups_data(groups)
                self._show_toast(f"✓ Fetched {len(groups)} groups", (100, 255, 100))
        except Exception as e:
            self._show_toast("✗ Blender Timeout/Error", (255, 100, 100))

    def _merge_groups_data(self, blender_groups):
        self._sync_ui_to_data()
        # Remove groups not in blender
        for k in list(self.groups_data.keys()):
            if k not in blender_groups: del self.groups_data[k]
        # Add new groups
        for g in blender_groups:
            if g not in self.groups_data: self.groups_data[g] = {"x": {}, "y": {}}
        
        items = list(self.groups_data.keys())
        dpg.configure_item("group_combo", items=items)
        if items and not self.current_group_name:
            dpg.set_value("group_combo", items[0])
            self.on_group_select(None, items[0])

    def open_mesh_map(self):
        try:
            if os.name == 'nt': os.startfile(REF_MAP_FILE)
            else:
                import subprocess
                subprocess.call(["xdg-open", REF_MAP_FILE])
        except: self._show_toast("✗ Could not open mesh map", (255, 100, 100))

    def setup_textures(self):
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(640, 480, np.zeros((480, 640, 4), dtype=np.float32), 
                               tag="camera_texture", format=dpg.mvFormat_Float_rgba)

    def setup_theme(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (32, 32, 38))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 70, 110))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 50, 70))
        dpg.bind_theme(global_theme)

    def build_ui(self):
        with dpg.window(tag="Primary Window"):
            # Initialization Overlay
            with dpg.window(label="Initializing", modal=True, show=True, tag="init_window", no_title_bar=True, no_move=True, no_resize=True, width=400, height=150, pos=(400, 300)):
                dpg.add_spacer(height=20)
                dpg.add_text("   🚀 INITIALIZING SYSTEM...", color=(100, 200, 255))
                dpg.add_spacer(height=10)
                dpg.add_loading_indicator(style=1, radius=3, color=(100, 200, 255))
                dpg.add_spacer(height=10)
                dpg.add_text("Please wait while we setup assets...", tag="init_status", color=(150, 150, 150))

            with dpg.group(horizontal=True, tag="main_layout_group"):
                # --- LEFT COLUMN: Settings ---
                with dpg.child_window(width=450, border=True, tag="left_col"):
                    dpg.add_text("⚡ SHAPEKEY TRACKER PROFESSIONAL", color=(100, 200, 255))
                    dpg.add_text("", tag="toast_text")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="Show Camera", default_value=True, callback=lambda s,v: setattr(self, 'camera_show', v))
                        dpg.add_checkbox(label="Send UDP", default_value=True, callback=lambda s,v: setattr(self, 'send_enabled', v))
                        dpg.add_checkbox(label="Mesh", default_value=True, callback=lambda s,v: setattr(self, 'draw_mesh', v))
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="🔄 FETCH FROM BLENDER", width=210, callback=self.fetch_groups)
                        dpg.add_button(label="📷 OPEN MESH MAP", width=210, callback=self.open_mesh_map)
                    
                    dpg.add_separator()
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Active Group:")
                        self.group_combo = dpg.add_combo(items=list(self.groups_data.keys()), tag="group_combo", width=250, callback=self.on_group_select)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="+ NEW GROUP", width=200, callback=self.on_add_group)
                        dpg.add_button(label="- DELETE GROUP", width=204, callback=self.on_remove_group)

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="📤 EXPORT PRESET", width=200, callback=self.on_export_preset)
                        dpg.add_button(label="📥 IMPORT PRESET", width=204, callback=self.on_import_preset)

                    dpg.add_spacer(height=5)
                    dpg.add_text("Quick Auto Mapping:", color=(200, 200, 100))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="👄 MOUTH", width=132, callback=lambda: self._apply_hardcoded_preset("Mouth Standard"))
                        dpg.add_button(label="👁 EYES", width=132, callback=lambda: self._apply_hardcoded_preset("Eyes Tracking"))
                        dpg.add_button(label="😑 BLINK", width=132, callback=lambda: self._apply_hardcoded_preset("Eye Blink (Y)"))

                    dpg.add_spacer(height=10)
                    with dpg.child_window(tag="inspector_scroll", border=False):
                        with dpg.tab_bar():
                            for axis in ["X", "Y"]:
                                with dpg.tab(label=f"Edit {axis} Mapping"):
                                    self.build_axis_ui(axis.lower())

                    dpg.add_spacer(height=5)
                    dpg.add_button(label="SAVE PERMANENTLY", width=-1, height=45, callback=self.save_config)

                # --- MIDDLE COLUMN: Viewport ---
                with dpg.group(width=660, tag="mid_col"):
                    with dpg.child_window(border=True, tag="cam_window"):
                        dpg.add_image("camera_texture", width=640, height=480, tag="cam_image")
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text("FPS: 0", tag="fps_text", color=(0, 255, 0))
                            dpg.add_spacer(width=20)
                            dpg.add_text("Hover ID: -1", tag="hover_id_text", color=(255, 255, 0))
                            dpg.add_spacer(width=20)
                            dpg.add_text("Selected: None", tag="selected_id_text", color=(255, 100, 100))

                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=5)
                        dpg.add_text("🕹 LIVE JOYSTICK & STATUS")
                        with dpg.group(horizontal=True):
                            with dpg.drawlist(width=160, height=160):
                                dpg.draw_rectangle((0, 0), (160, 160), color=(150, 150, 150), thickness=2)
                                dpg.draw_rectangle((2, 2), (158, 158), fill=(30, 30, 35))
                                dpg.draw_line((0, 80), (160, 80), color=(60, 60, 65))
                                dpg.draw_line((80, 0), (80, 160), color=(60, 60, 65))
                                dpg.draw_circle((80, 80), 8, color=(100, 200, 255), fill=(80, 150, 255, 150), tag="joy_dot")
                                dpg.draw_line((80, 80), (80, 80), color=(100, 200, 255, 100), tag="joy_line")
                            
                            with dpg.group():
                                dpg.add_spacer(width=20)
                                dpg.add_text("Current Values:", color=(100, 150, 255))
                                with dpg.group(horizontal=True):
                                    dpg.add_text("X:", color=(150, 150, 150))
                                    dpg.add_text("0.000", tag="val_xo")
                                    dpg.add_text("raw:", color=(80, 80, 80))
                                    dpg.add_text("0.0000", tag="val_xr")
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Y:", color=(150, 150, 150))
                                    dpg.add_text("0.000", tag="val_yo")
                                    dpg.add_text("raw:", color=(80, 80, 80))
                                    dpg.add_text("0.0000", tag="val_yr")
                
                # --- RIGHT COLUMN: Realtime Log ---
                with dpg.child_window(width=-1, border=True, tag="right_col"):
                    dpg.add_text("📝 REAL-TIME CONSOLE LOG", color=(255, 200, 100))
                    dpg.add_separator()
                    with dpg.child_window(tag="log_child_window", border=False):
                        dpg.add_text("", tag="realtime_log_text")

    def build_axis_ui(self, axis):
        p = f"{axis}_"
        dpg.add_text("Mode Selection:")
        dpg.add_radio_button(["None", "2pt (Dist)", "1pt (Proj)", "iris"], horizontal=True, tag=p+"mode", 
                             callback=lambda: self._sync_ui_to_data())
        
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Base Pt A", width=140, tag=p+"pt_a", callback=lambda: self._sync_ui_to_data())
            dpg.add_button(label="Assign Mouse", small=True, callback=lambda: self._assign_selected(p+"pt_a"))
            
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Base Pt B", width=140, tag=p+"pt_b", callback=lambda: self._sync_ui_to_data())
            dpg.add_button(label="Assign Mouse", small=True, callback=lambda: self._assign_selected(p+"pt_b"))

        with dpg.group(horizontal=True):
            dpg.add_text("Presets:")
            dpg.add_button(label=f"Eye {axis.upper()} R", small=True, callback=lambda: self._apply_eye_preset(axis, "R"))
            dpg.add_button(label=f"Eye {axis.upper()} L", small=True, callback=lambda: self._apply_eye_preset(axis, "L"))

        dpg.add_spacer(height=5)
        self._add_slider_text("Radius Min", p+"rmin", 0.0, 2.0)
        self._add_slider_text("Radius Max", p+"rmax", 0.0, 2.0)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="CALIBRATE MIN", width=210, callback=lambda: self.calibrate(axis, "min"))
            dpg.add_button(label="CALIBRATE MAX", width=210, callback=lambda: self.calibrate(axis, "max"))

        dpg.add_separator()
        self._add_slider_text("Out Min", p+"omin", -1.0, 1.0)
        self._add_slider_text("Out Max", p+"omax", -1.0, 1.0)
        self._add_slider_text("Sensitivity", p+"sens", 0.1, 5.0)
        self._add_slider_text("Exp Power", p+"exp", 0.5, 3.0)
        
        dpg.add_checkbox(label="Enable Smoothing", tag=p+"lerp_en", callback=lambda: self._sync_ui_to_data())
        self._add_slider_text("Smooth Speed", p+"lerp_fac", 0.01, 0.5)

    def _add_slider_text(self, label, tag, min_v=0.0, max_v=1.0, format="%.3f"):
        with dpg.group(horizontal=True):
            with dpg.group(width=110): # Using a sub-group to enforce width for the label area
                dpg.add_text(f"{label}:")
            # Slider with no label and no internal format display (to avoid clutter)
            dpg.add_slider_float(width=190, min_value=min_v, max_value=max_v, tag=tag, 
                                 format="", callback=lambda: self._sync_ui_to_data())
            # Precise input field linked to the same tag (source)
            dpg.add_input_float(width=100, source=tag, step=0, step_fast=0, 
                                format=format, callback=lambda: self._sync_ui_to_data())

    def _apply_eye_preset(self, axis, eye_key):
        e = EYE_R if eye_key == "R" else EYE_L
        p = f"{axis}_"
        dpg.set_value(p+"mode", "iris")
        if axis == "x":
            dpg.set_value(p+"pt_a", e["iris"])
            dpg.set_value(p+"pt_b", e["outer"])
        else:
            dpg.set_value(p+"pt_a", e["iris"])
            dpg.set_value(p+"pt_b", e["top"]) # Using top as ref for Y
        
        dpg.set_value(p+"rmin", 0.02)
        dpg.set_value(p+"rmax", 0.25)
        dpg.set_value(p+"omin", 0.0)
        dpg.set_value(p+"omax", 1.0)
        self._sync_ui_to_data()
        self._show_toast(f"Applied Eye {eye_key} {axis.upper()} Preset", (100, 200, 255))

    def _assign_selected(self, tag):
        if self.selected_id != -1:
            dpg.set_value(tag, self.selected_id)
            self._sync_ui_to_data()

    def _handle_click(self):
        if dpg.is_item_hovered("cam_image"):
            if self.hover_id != -1:
                self.selected_id = self.hover_id
                dpg.set_value("selected_id_text", f"Selected ID: {self.selected_id}")

    def calibrate(self, axis, which):
        raw = self.current_vals["rx"] if axis == "x" else self.current_vals["ry"]
        dpg.set_value(f"{axis}_r{which}", abs(raw))
        self._sync_ui_to_data()

    def on_group_select(self, sender, app_data):
        self.current_group_name = app_data
        self._populate_ui_from_data(app_data)

    def on_add_group(self):
        name = f"Group_{int(time.time()*100)%1000}"
        self.groups_data[name] = {"x": {"mode":"2pt (Dist)", "radius_max":1.0, "out_max":1.0, "sens":1.0, "exp_power":1.2}, 
                                  "y": {"mode":"2pt (Dist)", "radius_max":1.0, "out_max":1.0, "sens":1.0, "exp_power":1.2}}
        dpg.configure_item("group_combo", items=list(self.groups_data.keys()))
        dpg.set_value("group_combo", name)
        self.on_group_select(None, name)

    def on_remove_group(self):
        if self.current_group_name in self.groups_data:
            del self.groups_data[self.current_group_name]
            items = list(self.groups_data.keys())
            dpg.configure_item("group_combo", items=items)
            if items: 
                dpg.set_value("group_combo", items[0])
                self.on_group_select(None, items[0])

    def on_export_preset(self):
        if not self.current_group_name:
            self._show_toast("⚠ Select a group first", (255, 200, 0))
            return
        
        self._sync_ui_to_data()
        group_data = self.groups_data[self.current_group_name]
        filename = f"preset_{self.current_group_name}.json"
        
        try:
            with open(filename, "w") as f:
                json.dump(group_data, f, indent=4)
            self._show_toast(f"✓ Exported to {filename}", (0, 255, 120))
            # Open the folder to show the user where the file is
            if os.name == 'nt': os.startfile(os.getcwd())
        except Exception as e:
            self._show_toast(f"✗ Export Error: {e}", (255, 50, 50))

    def on_import_preset(self):
        if not self.current_group_name:
            self._show_toast("⚠ Select/Create a group first", (255, 200, 0))
            return

        # Simplified file selection for DPG (looking for local json files)
        # In a real app we'd use a file dialog, but for now we look for preset_*.json
        files = [f for f in os.listdir(".") if f.endswith(".json") and f != "config.json"]
        if not files:
            self._show_toast("✗ No .json presets found in folder", (255, 100, 100))
            return

        def _do_import(sender, file_name):
            try:
                with open(file_name, "r") as f:
                    new_data = json.load(f)
                self.groups_data[self.current_group_name] = new_data
                self._populate_ui_from_data(self.current_group_name)
                dpg.delete_item("import_window")
                self._show_toast(f"✓ Imported {file_name}", (0, 255, 120))
            except Exception as e:
                self._show_toast(f"✗ Import Error: {e}", (255, 50, 50))

        if dpg.does_item_exist("import_window"): dpg.delete_item("import_window")
        
        with dpg.window(label="Select Preset to Import", modal=True, tag="import_window", width=300, height=200, pos=(400, 300)):
            dpg.add_text("Choose a file to load into current group:")
            for f in files:
                dpg.add_button(label=f, width=-1, callback=_do_import, user_data=f)
            dpg.add_spacer(height=10)
            dpg.add_button(label="Cancel", width=-1, callback=lambda: dpg.delete_item("import_window"))

    def _apply_hardcoded_preset(self, preset_name):
        if not self.current_group_name:
            self._show_toast("⚠ Select a group first", (255, 200, 0))
            return
        
        preset = DEFAULT_PRESETS.get(preset_name)
        if preset:
            self.groups_data[self.current_group_name] = preset
            self._populate_ui_from_data(self.current_group_name)
            self._show_toast(f"✓ Applied {preset_name} Preset", (100, 255, 150))

    def _sync_ui_to_data(self):
        if not self.current_group_name: return
        group = self.groups_data[self.current_group_name]
        for axis in ["x", "y"]:
            if axis not in group: group[axis] = {}
            a, p = group[axis], f"{axis}_"
            a["mode"] = dpg.get_value(p+"mode")
            a["point_a"] = dpg.get_value(p+"pt_a")
            a["point_b"] = dpg.get_value(p+"pt_b")
            a["radius_min"] = dpg.get_value(p+"rmin")
            a["radius_max"] = dpg.get_value(p+"rmax")
            a["out_min"] = dpg.get_value(p+"omin")
            a["out_max"] = dpg.get_value(p+"omax")
            a["sens"] = dpg.get_value(p+"sens")
            a["exp_power"] = dpg.get_value(p+"exp")
            a["lerp_en"] = dpg.get_value(p+"lerp_en")
            a["lerp_fac"] = dpg.get_value(p+"lerp_fac")

    def _populate_ui_from_data(self, group_name):
        data = self.groups_data.get(group_name, {})
        for axis in ["x", "y"]:
            a, p = data.get(axis, {}), f"{axis}_"
            dpg.set_value(p+"mode", a.get("mode", "2pt (Dist)"))
            dpg.set_value(p+"pt_a", a.get("point_a", 0))
            dpg.set_value(p+"pt_b", a.get("point_b", 0))
            dpg.set_value(p+"rmin", a.get("radius_min", 0.0))
            dpg.set_value(p+"rmax", a.get("radius_max", 1.0))
            dpg.set_value(p+"omin", a.get("out_min", 0.0))
            dpg.set_value(p+"omax", a.get("out_max", 1.0))
            dpg.set_value(p+"sens", a.get("sens", 1.0))
            dpg.set_value(p+"exp", a.get("exp_power", 1.2))
            dpg.set_value(p+"lerp_en", a.get("lerp_en", False))
            dpg.set_value(p+"lerp_fac", a.get("lerp_fac", 0.15))

    def run_tracker_loop(self):
        logger.info("Unpacking assets...")
        self._ensure_assets()
        dpg.set_value("init_status", "Checking Camera Access...")
        
        try:
            landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
                running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True
            ))
            logger.info("Mediapipe Landmarker initialized.")
        except Exception as e:
            logger.error(f"Landmarker init failed: {e}")
            dpg.set_value("init_status", "FATAL ERROR: See tracker_log.txt")
            return

        cam_idx = self.config.get("camera_index", 0)
        logger.info(f"Opening camera index: {cam_idx}")
        cap = cv2.VideoCapture(cam_idx)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {cam_idx}")
            dpg.set_value("init_status", f"CAMERA ERROR: Index {cam_idx} not found")
            return

        dpg.set_value("init_status", "Ready!")
        time.sleep(1.2)
        dpg.delete_item("init_window")
        self.initialized = True
        logger.info("System Initialized.")

        last_t = time.time()
        while self.running:
            success, raw_frame = cap.read()
            if not success: continue
            
            raw_frame = cv2.flip(raw_frame, 1)
            f_h, f_w = raw_frame.shape[:2]
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            results = landmarker.detect_for_video(mp_img, int(time.time() * 1000))
            payload = {}
            self.active_points_x = set()
            self.active_points_y = set()
            self.hover_id = -1

            if results.face_landmarks:
                lms = results.face_landmarks[0]
                self.latest_landmarks = lms
                
                # Face metrics
                f_width = calculate_distance(lms[234], lms[454]) or 1.0
                fx_v = (lms[454].x-lms[234].x, lms[454].y-lms[234].y, lms[454].z-lms[234].z)
                fy_v = (lms[152].x-lms[10].x, lms[152].y-lms[10].y, lms[152].z-lms[10].z)
                
                # Precise Point Hover Logic
                mouse_screen = dpg.get_mouse_pos(local=False)
                rect_min = dpg.get_item_rect_min("cam_image")
                if rect_min:
                    mx, my = mouse_screen[0] - rect_min[0], mouse_screen[1] - rect_min[1]
                    if 0 <= mx <= 640 and 0 <= my <= 480:
                        min_d = 0.05
                        for i, lm in enumerate(lms):
                            d = math.sqrt((mx/640 - lm.x)**2 + (my/480 - lm.y)**2)
                            if d < min_d: min_d, self.hover_id = d, i

                # Axis Logic
                for gn, mapings in self.groups_data.items():
                    out_d = {"x": 0.0, "y": 0.0}
                    for axis in ["x", "y"]:
                        m = mapings.get(axis, {})
                        if "point_a" not in m: continue
                        mode, ia, ib = m.get("mode", ""), int(m.get("point_a", 0)), int(m.get("point_b", 0))
                        
                        raw = 0.0
                        if mode == "None":
                            raw = 0.0
                        elif "iris" in mode:
                            ek = "L" if ib == EYE_L["outer"] or ib == EYE_L["top"] else "R"
                            e = EYE_L if ek == "L" else EYE_R
                            pa, pi, po = lms[e["iris"]], lms[e["inner"]], lms[e["outer"]]
                            pt, pb = lms[e["top"]], lms[e["bottom"]]
                            ew = abs(pi.x-po.x) or 1.0
                            eh = abs(pt.y-pb.y) or 0.1
                            
                            # Vertical axis is often less sensitive, adjust or relax blink protection
                            if eh > 0.05 * ew: # Relaxed from 0.25 to allow tracking more closed eyes
                                if axis == "x":
                                    inst = (pa.x - (pi.x+po.x)/2)/ew
                                else:
                                    # Y Axis: pa.y increases downwards. Look Up -> pa.y small -> inst negative.
                                    # We multiply by -1 if we want Look Up to be Positive.
                                    # But we'll keep it raw and let user set mins/maxes, 
                                    # just ensure it's normalized to the horizontal width for scale stability.
                                    inst = (pa.y - (pt.y+pb.y)/2)/ew * 2.0 # Extra gain for Y as vertical travel is smaller
                                    
                                self._iris_ema[ek][axis] += 0.3 * (inst - self._iris_ema[ek][axis])
                            
                            raw = self._iris_ema[ek][axis]
                            pex = m.get("exp_power", 1.2)
                            raw = math.copysign(abs(raw)**pex, raw)
                            
                            points = [e["iris"], e["inner"], e["outer"], e["top"], e["bottom"]]
                            if axis == "x": self.active_points_x.update(points)
                            else: self.active_points_y.update(points)
                        elif "1pt" in mode:
                            pa, pb = lms[ia], lms[ib]
                            vec = (pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)
                            f_v = fx_v if axis=="x" else fy_v
                            dot = (vec[0]*f_v[0] + vec[1]*f_v[1] + vec[2]*f_v[2])
                            raw = (dot / f_width) * 10.0
                            if axis == "x": self.active_points_x.update([ia, ib])
                            else: self.active_points_y.update([ia, ib])
                        elif "2pt" in mode: # Explicit 2pt
                            raw = calculate_distance(lms[ia], lms[ib]) / f_width
                            if axis == "x": self.active_points_x.update([ia, ib])
                            else: self.active_points_y.update([ia, ib])
                        else:
                            raw = 0.0

                        if gn == self.current_group_name: self.current_vals[f"r{axis}"] = raw
                        
                        rmin, rmax, omin, omax = m.get("radius_min", 0.0), m.get("radius_max", 1.0), m.get("out_min", 0.0), m.get("out_max", 1.0)
                        if "2pt" in mode: val = normalize_value(raw, rmin, rmax, omin, omax)
                        else: val = normalize_value(abs(raw), rmin, rmax, omin, omax) * math.copysign(1, raw)
                        
                        val *= m.get("sens", 1.0)
                        if m.get("lerp_en"):
                            lk = gn+axis
                            if lk not in self.lerp_values: self.lerp_values[lk] = val
                            self.lerp_values[lk] += (val - self.lerp_values[lk]) * m.get("lerp_fac", 0.15)
                            val = self.lerp_values[lk]
                        
                        out_d[axis] = max(-1.0, min(1.0, val))
                        if gn == self.current_group_name: self.current_vals[axis] = out_d[axis]
                    payload[gn] = out_d

                # Update Texture for UI (Respect Privacy)
            if self.camera_show:
                display_frame = raw_frame.copy()
            else:
                display_frame = np.zeros((f_h, f_w, 3), dtype=np.uint8)

            # Visual Rendering (Drawing results on display_frame)
            if self.draw_mesh and results.face_landmarks:
                lms = results.face_landmarks[0]
                for i, lm in enumerate(lms):
                    px, py = int(lm.x*f_w), int(lm.y*f_h)
                    if i == self.selected_id: color, size = (0, 0, 255), 3 # Red
                    elif i == self.hover_id: color, size = (0, 255, 255), 4 # Yellow
                    elif i in self.active_points_x and i in self.active_points_y: color, size = (255, 255, 255), 2 # White (Both)
                    elif i in self.active_points_x: color, size = (255, 100, 0), 2 # Orange (X)
                    elif i in self.active_points_y: color, size = (0, 150, 255), 2 # Blue (Y)
                    else: color, size = (0, 255, 100), 1 # Green
                    cv2.circle(display_frame, (px, py), size, color, -1)

            # Send Network
            if self.send_enabled and payload:
                try: self.sock.sendto(json.dumps(payload).encode(), self.target_address)
                except: pass

            # Update DPG Texture
            f_res = cv2.resize(display_frame, (640, 480))
            tex_data = np.array(cv2.cvtColor(f_res, cv2.COLOR_BGR2RGBA), dtype=np.float32).flatten() / 255.0
            dpg.set_value("camera_texture", tex_data)
            
            # Update Telemetry Stats
            dpg.set_value("fps_text", f"FPS: {int(1.0/(time.time()-last_t))}"); last_t = time.time()
            dpg.set_value("hover_id_text", f"Hover ID: {self.hover_id}")
            if self.current_group_name:
                dpg.set_value("val_xr", f"{self.current_vals['rx']:.4f}")
                dpg.set_value("val_xo", f"{self.current_vals['x']:.3f}")
                dpg.set_value("val_yr", f"{self.current_vals['ry']:.4f}")
                dpg.set_value("val_yo", f"{self.current_vals['y']:.3f}")
                # Update Joystick Marker
                jx = 80 + (self.current_vals['x'] * 80)
                jy = 80 - (self.current_vals['y'] * 80)
                dpg.configure_item("joy_dot", center=(jx, jy))
                dpg.configure_item("joy_line", p2=(jx, jy))

        cap.release()

    def start(self):
        dpg.set_primary_window("Primary Window", True)
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        self.running = False
        dpg.destroy_context()

if __name__ == "__main__":
    app = FaceTrackerAppDPG()
    app.start()
