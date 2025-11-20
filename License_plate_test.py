"""
License Plate App: YOLOv8 + EasyOCR (no Tesseract)

- Loads YOLOv8 model (best.pt) to detect license plates
- Uses EasyOCR to read text from the cropped plate region
- You can:
    - Open an image from disk
    - Capture a frame from webcam
- GUI shows:
    - Full image with detections drawn
    - Cropped license plate
    - Recognized plate text
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
import re
import easyocr  # OCR engine (no external exe needed)

# =======================
# CONFIG
# =======================

# Path to your trained YOLO weights
MODEL_PATH = r"C:\Users\minkh\Desktop\Persnl_11\best_l1_1.pt"   # change to full path if needed

# Initialize EasyOCR reader once (English; GPU off for simplicity)
ocr_reader = easyocr.Reader(['en'], gpu=False)


class PlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector (YOLOv8 + EasyOCR)")
        self.root.geometry("1000x720")

        # Try to load YOLO model
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model not found",
                f"Cannot find model file:\n{MODEL_PATH}\n\n"
                f"Put your best.pt there or change MODEL_PATH.",
            )
            self.model = None
        else:
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception as e:
                messagebox.showerror("Error loading model", str(e))
                self.model = None

        # References to PhotoImage to prevent garbage collection
        self.full_img_tk = None
        self.crop_img_tk = None

        # Text variable for recognized plate
        self.plate_text_var = tk.StringVar(value="(no plate yet)")

        self._build_ui()

    # ---------------- UI LAYOUT ----------------

    def _build_ui(self):
        # Top frame: images side by side
        top_frame = tk.Frame(self.root)
        top_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Left: full image with detections
        left_frame = tk.Frame(top_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(left_frame, text="Detected Image", font=("Arial", 14)).pack(pady=5)
        self.full_img_label = tk.Label(left_frame, bg="gray")
        self.full_img_label.pack(fill="both", expand=True)

        # Right: cropped plate
        right_frame = tk.Frame(top_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        tk.Label(right_frame, text="Cropped License Plate", font=("Arial", 14)).pack(pady=5)
        self.crop_img_label = tk.Label(right_frame, bg="gray")
        self.crop_img_label.pack(fill="both", expand=True)

        # Middle frame: recognized text
        text_frame = tk.Frame(self.root)
        text_frame.pack(side="top", fill="x", padx=10, pady=(0, 5))

        tk.Label(text_frame, text="Recognized Plate:", font=("Arial", 12)).pack(
            side="left", padx=(0, 5)
        )
        self.plate_text_entry = tk.Entry(
            text_frame,
            textvariable=self.plate_text_var,
            font=("Consolas", 14),
            width=25,
        )
        self.plate_text_entry.pack(side="left", padx=(0, 10))

        # Bottom frame: buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side="bottom", fill="x", pady=10)

        tk.Button(
            btn_frame,
            text="Capture from Camera",
            command=self.capture_from_camera,
            width=20,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Open Image",
            command=self.open_image,
            width=15,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Quit",
            command=self.root.destroy,
            width=10,
        ).pack(side="right", padx=10)

    # ---------------- HELPERS ----------------

    def _cv_to_tk(self, cv_img, max_size=(640, 480)):
        """Convert OpenCV BGR image to Tkinter PhotoImage, with simple resize."""
        h, w = cv_img.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h))

        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        return ImageTk.PhotoImage(pil_img)

    def _read_plate_text(self, crop_bgr):
        """
        Run OCR on cropped plate (BGR image) using EasyOCR.
        Returns cleaned text (letters and digits).
        """
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            # detail=0 -> returns list of strings only
            results = ocr_reader.readtext(crop_rgb, detail=0)
            if not results:
                return ""

            # Take the longest text result as plate candidate
            text = max(results, key=len)

            # Clean similar to plate format: A-Z, 0-9 only
            cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            return cleaned or text.strip()
        except Exception as e:
            print("EasyOCR error:", e)
            return ""

    def _run_detection(self, cv_img):
        """Run YOLO detection, update GUI with detection, crop, and OCR."""
        if self.model is None:
            messagebox.showerror(
                "Model not loaded",
                "Model could not be loaded. Check MODEL_PATH and restart.",
            )
            return

        try:
            results = self.model(cv_img)[0]  # single result object
        except Exception as e:
            messagebox.showerror("Detection error", str(e))
            return

        # YOLO draw boxes on the image (BGR)
        plotted = results.plot()

        # Show full image
        self.full_img_tk = self._cv_to_tk(plotted, max_size=(640, 480))
        self.full_img_label.config(image=self.full_img_tk)

        # Reset text
        self.plate_text_var.set("(no plate)")

        # Take first detection as plate
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            h, w = cv_img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            crop = cv_img[y1:y2, x1:x2]
            if crop.size == 0:
                messagebox.showwarning(
                    "Crop warning",
                    "Detected box is invalid, cannot crop.",
                )
                self.crop_img_label.config(image="", text="No crop")
                self.plate_text_var.set("(crop error)")
                return

            # Show cropped plate
            self.crop_img_tk = self._cv_to_tk(crop, max_size=(400, 300))
            self.crop_img_label.config(image=self.crop_img_tk)

            # OCR with EasyOCR
            text = self._read_plate_text(crop)
            if text.strip():
                self.plate_text_var.set(text)
            else:
                self.plate_text_var.set("(no text found)")
        else:
            messagebox.showinfo("No plate found", "No license plate detected.")
            self.crop_img_label.config(image="", text="No plate")
            self.plate_text_var.set("(no plate)")

    # ---------------- BUTTON ACTIONS ----------------

    def open_image(self):
        """Open an image from disk and run detection+OCR."""
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        cv_img = cv2.imread(file_path)
        if cv_img is None:
            messagebox.showerror("Error", f"Could not open image:\n{file_path}")
            return

        self._run_detection(cv_img)

    def capture_from_camera(self):
        """Capture a frame from webcam and run detection+OCR."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not open camera.")
            return

        messagebox.showinfo(
            "Camera",
            "Camera window will open.\n\n"
            "Press 'c' to capture a frame.\n"
            "Press 'q' or ESC to cancel.",
        )

        captured_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Camera - press 'c' to capture, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                captured_frame = frame.copy()
                break
            elif key == ord("q") or key == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is not None:
            self._run_detection(captured_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlateApp(root)
    root.mainloop()
