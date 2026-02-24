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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Constants & Paths ---
if getattr(threading, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "face_landmarker.task")
REF_MAP_FILE = os.path.join(SCRIPT_DIR, "face_mesh.png")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
REF_MAP_URL = "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png"

# MediaPipe landmark indices for iris tracking
EYE_R = {"iris": 468, "inner": 133, "outer": 33, "top": 159, "bottom": 145}
EYE_L = {"iris": 473, "inner": 362, "outer": 263, "top": 386, "bottom": 374}

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
        
        self.latest_landmarks = None
        self.active_points = set()
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
        
        dpg.create_viewport(title="ShapeKey Face Tracker (DPG High-Performance)", width=1200, height=1000)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        self.tracker_thread = threading.Thread(target=self.run_tracker_loop, daemon=True)
        self.tracker_thread.start()

    def _ensure_assets(self):
        if not os.path.exists(MODEL_FILE):
            print("Downloading model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        if not os.path.exists(REF_MAP_FILE):
            try:
                print("Downloading mesh map...")
                req = urllib.request.Request(REF_MAP_URL, headers={'User-Agent': 'Mozilla/5.0'})
                with open(REF_MAP_FILE, 'wb') as f:
                    f.write(urllib.request.urlopen(req).read())
            except: pass

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
            with dpg.group(horizontal=True):
                # --- LEFT COLUMN: Settings ---
                with dpg.child_window(width=450, border=True):
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

                    dpg.add_spacer(height=10)
                    with dpg.child_window(height=600, border=False):
                        with dpg.tab_bar():
                            for axis in ["X", "Y"]:
                                with dpg.tab(label=f"Edit {axis} Mapping"):
                                    self.build_axis_ui(axis.lower())

                    dpg.add_spacer(height=5)
                    dpg.add_button(label="SAVE PERMANENTLY", width=-1, height=45, callback=self.save_config)

                # --- RIGHT COLUMN: Viewport ---
                with dpg.group():
                    with dpg.child_window(width=-1, height=520, border=True, tag="cam_window"):
                        dpg.add_image("camera_texture", width=640, height=480, tag="cam_image")
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text("FPS: 0", tag="fps_text", color=(0, 255, 0))
                            dpg.add_spacer(width=20)
                            dpg.add_text("Hover ID: -1", tag="hover_id_text", color=(255, 255, 0))
                            dpg.add_spacer(width=20)
                            dpg.add_text("Selected: None", tag="selected_id_text", color=(255, 100, 100))

                    with dpg.child_window(width=-1, height=-1, border=True):
                        dpg.add_text("LIVE JOYSTICK FEEDBACK")
                        with dpg.group(horizontal=True):
                            with dpg.drawlist(width=200, height=200):
                                dpg.draw_rectangle((0, 0), (200, 200), color=(150, 150, 150), thickness=2)
                                dpg.draw_rectangle((2, 2), (198, 198), fill=(30, 30, 35))
                                dpg.draw_line((0, 100), (200, 100), color=(60, 60, 65))
                                dpg.draw_line((100, 0), (100, 200), color=(60, 60, 65))
                                dpg.draw_circle((100, 100), 8, color=(100, 200, 255), fill=(80, 150, 255, 150), tag="joy_dot")
                                dpg.draw_line((100, 100), (100, 100), color=(100, 200, 255, 100), tag="joy_line")
                            
                            with dpg.group():
                                dpg.add_spacer(width=20)
                                dpg.add_text("Current Status:", color=(100, 150, 255))
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

    def build_axis_ui(self, axis):
        p = f"{axis}_"
        dpg.add_text("Mode Selection:")
        dpg.add_radio_button(["2pt (Dist)", "1pt (Proj)", "iris"], horizontal=True, tag=p+"mode", 
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
        dpg.add_slider_float(label="Radius Min", max_value=2.0, tag=p+"rmin", callback=lambda: self._sync_ui_to_data())
        dpg.add_slider_float(label="Radius Max", max_value=2.0, tag=p+"rmax", callback=lambda: self._sync_ui_to_data())
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="CALIBRATE MIN", width=200, callback=lambda: self.calibrate(axis, "min"))
            dpg.add_button(label="CALIBRATE MAX", width=200, callback=lambda: self.calibrate(axis, "max"))

        dpg.add_separator()
        dpg.add_slider_float(label="Out Min", min_value=-1.0, max_value=1.0, tag=p+"omin", callback=lambda: self._sync_ui_to_data())
        dpg.add_slider_float(label="Out Max", min_value=-1.0, max_value=1.0, tag=p+"omax", callback=lambda: self._sync_ui_to_data())
        dpg.add_slider_float(label="Sensitivity", min_value=0.1, max_value=5.0, tag=p+"sens", callback=lambda: self._sync_ui_to_data())
        dpg.add_slider_float(label="Exp Power", min_value=0.5, max_value=3.0, tag=p+"exp", callback=lambda: self._sync_ui_to_data())
        
        dpg.add_checkbox(label="Enable Smoothing", tag=p+"lerp_en", callback=lambda: self._sync_ui_to_data())
        dpg.add_slider_float(label="Smooth Speed", min_value=0.01, max_value=0.5, tag=p+"lerp_fac", callback=lambda: self._sync_ui_to_data())

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
        landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
            running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True
        ))
        cap = cv2.VideoCapture(self.config.get("camera_index", 0))
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
            self.active_points = set()
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
                        if "iris" in mode:
                            ek = "L" if ib == EYE_L["outer"] or ib == EYE_L["top"] else "R"
                            e = EYE_L if ek == "L" else EYE_R
                            pa, pi, po = lms[e["iris"]], lms[e["inner"]], lms[e["outer"]]
                            pt, pb = lms[e["top"]], lms[e["bottom"]]
                            ew = abs(pi.x-po.x) or 1.0
                            if abs(pt.y-pb.y) > 0.25 * ew:
                                inst = (pa.x - (pi.x+po.x)/2)/ew if axis=="x" else (pa.y - (pt.y+pb.y)/2)/ew
                                self._iris_ema[ek][axis] += 0.3 * (inst - self._iris_ema[ek][axis])
                            raw = self._iris_ema[ek][axis]
                            pex = m.get("exp_power", 1.2)
                            raw = math.copysign(abs(raw)**pex, raw)
                            self.active_points.update([e["iris"], e["inner"], e["outer"], e["top"], e["bottom"]])
                        elif "1pt" in mode:
                            pa, pb = lms[ia], lms[ib]
                            vec = (pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)
                            f_v = fx_v if axis=="x" else fy_v
                            dot = (vec[0]*f_v[0] + vec[1]*f_v[1] + vec[2]*f_v[2])
                            raw = (dot / f_width) * 10.0
                            self.active_points.update([ia, ib])
                        else: # 2pt Distance
                            raw = calculate_distance(lms[ia], lms[ib]) / f_width
                            self.active_points.update([ia, ib])

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
                    elif i in self.active_points: color, size = (255, 100, 0), 2 # Blue
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
                jx = 100 + (self.current_vals['x'] * 100)
                jy = 100 - (self.current_vals['y'] * 100)
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
