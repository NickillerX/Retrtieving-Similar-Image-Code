import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image
import threading
import os
import shutil
import torch
import numpy as np

from Final_Pipeline6 import (  # Ensure you rename your pipeline file to Final_Pipeline5.py
    CONFIG, FeatureExtractor, load_feature_vectors, preprocess_image,
    search_similar, run_feature_extraction
)


class ImageRetrievalApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__();
        ctk.set_appearance_mode("dark");
        ctk.set_default_color_theme("green");
        self.title("Similar Image Retrieval App");
        self.geometry("1200x800")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        self.model = None;
        self.feature_vectors = None;
        self.image_paths = None;
        self.image_labels = None;
        self.query_image_path = None;
        self.query_feature_vector = None;
        self.model_full_path = None;
        self.current_results = [];
        self.model_name = "resnet50";
        self.create_widgets();
        self.after(100, self.post_init_setup)

    def post_init_setup(self):
        self.check_device();self.update_status("App is ready.")

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1);
        self.grid_columnconfigure(1, weight=1);
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0);
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew");
        self.sidebar_frame.grid_rowconfigure(13, weight=1);
        self.main_content_frame = ctk.CTkFrame(self, fg_color="gray17", corner_radius=0);
        self.main_content_frame.grid(row=0, column=1, sticky="nsew");
        self.create_sidebar_widgets();
        self.create_main_content_widgets();
        self.check_all_button_states()

    def create_sidebar_widgets(self):
        ctk.CTkLabel(self.sidebar_frame, text="Controls", font=("Helvetica", 18, "bold")).grid(row=0, column=0, pady=10,
                                                                                               padx=20, sticky="w")
        self.device_var = tk.StringVar(value=self.device.type);
        device_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent");
        device_frame.grid(row=1, column=0, pady=(0, 10), padx=20, sticky="ew");
        ctk.CTkLabel(device_frame, text="Device:").pack(side="left");
        self.radio_cpu = ctk.CTkRadioButton(device_frame, text="CPU", variable=self.device_var, value="cpu",
                                            command=self.on_device_change);
        self.radio_cpu.pack(side="left", padx=(10, 5));
        self.radio_gpu = ctk.CTkRadioButton(device_frame, text="GPU", variable=self.device_var, value="cuda",
                                            command=self.on_device_change);
        self.radio_gpu.pack(side="left", padx=5)
        ctk.CTkLabel(self.sidebar_frame, text="1. Select Model:", font=("Helvetica", 14)).grid(row=2, column=0,
                                                                                               pady=(10, 5), padx=20,
                                                                                               sticky="w")
        self.model_options = list(CONFIG["model_dims"].keys());
        self.model_var = ctk.StringVar(value=self.model_options[2]);
        self.model_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=self.model_options, variable=self.model_var,
                                            command=self.on_model_change);
        self.model_menu.grid(row=3, column=0, pady=5, padx=20, sticky="ew")
        self.load_model_btn = ctk.CTkButton(self.sidebar_frame, text="Load Model Weights",
                                            command=self.load_model_from_file);
        self.load_model_btn.grid(row=4, column=0, pady=5, padx=20, sticky="ew");
        self.model_path_label = ctk.CTkLabel(self.sidebar_frame, text="No model weights loaded", wraplength=200,
                                             justify="left", font=("Helvetica", 10));
        self.model_path_label.grid(row=5, column=0, padx=20, sticky="w")
        ctk.CTkLabel(self.sidebar_frame, text="2. Load Features:", font=("Helvetica", 14)).grid(row=6, column=0,
                                                                                                pady=(10, 5), padx=20,
                                                                                                sticky="w");
        self.load_features_btn = ctk.CTkButton(self.sidebar_frame, text="Load Feature File (.pkl)",
                                               command=self.load_features_from_file);
        self.load_features_btn.grid(row=7, column=0, pady=5, padx=20, sticky="ew");
        self.feature_path_label = ctk.CTkLabel(self.sidebar_frame, text="No features loaded", wraplength=200,
                                               justify="left", font=("Helvetica", 10));
        self.feature_path_label.grid(row=8, column=0, padx=20, sticky="w")
        ctk.CTkLabel(self.sidebar_frame, text="3. Or, Extract New Features:", font=("Helvetica", 14)).grid(row=9,
                                                                                                           column=0,
                                                                                                           pady=(10, 5),
                                                                                                           padx=20,
                                                                                                           sticky="w")
        self.extract_btn = ctk.CTkButton(self.sidebar_frame, text="Run Feature Extraction",
                                         command=self.start_extraction, state="disabled");
        self.extract_btn.grid(row=10, column=0, pady=5, padx=20, sticky="ew")
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="", font=("Helvetica", 12), text_color="green");
        self.status_label.grid(row=13, column=0, pady=(20, 10), padx=20, sticky="swe")

    def create_main_content_widgets(self):
        ctk.CTkLabel(self.main_content_frame, text="📷 Image Retrieval", font=("Helvetica", 24, "bold")).pack(pady=20);
        self.upload_frame = ctk.CTkFrame(self.main_content_frame, width=400, height=100, corner_radius=10);
        self.upload_frame.pack(pady=10);
        self.upload_frame.pack_propagate(False);
        self.upload_label = ctk.CTkLabel(self.upload_frame, text="Drag or Click to Upload an Image",
                                         font=("Helvetica", 16));
        self.upload_label.pack(expand=True);
        self.upload_frame.bind("<Button-1>", self.browse_image);
        self.drop_target_register(DND_FILES);
        self.dnd_bind("<<Drop>>", self.load_dropped_image);
        self.query_image_canvas = ctk.CTkLabel(self.main_content_frame, text="No Image Selected", fg_color="gray20",
                                               width=200, height=200);
        self.query_image_canvas.pack(pady=10);
        self.controls_frame = ctk.CTkFrame(self.main_content_frame, fg_color="transparent");
        self.controls_frame.pack(pady=10);
        self.slider_k_label = ctk.CTkLabel(self.controls_frame, text="Top K: 5");
        self.slider_k_label.grid(row=0, column=0, pady=5, padx=10);
        self.slider_k = ctk.CTkSlider(self.controls_frame, from_=1, to=20, number_of_steps=19,
                                      command=self.on_slider_change);
        self.slider_k.set(5);
        self.slider_k.grid(row=1, column=0, pady=5, padx=10);
        self.slider_thresh_label = ctk.CTkLabel(self.controls_frame, text="Similarity: 0.70");
        self.slider_thresh_label.grid(row=0, column=1, pady=5, padx=10);
        self.slider_thresh = ctk.CTkSlider(self.controls_frame, from_=0.1, to=1.0, number_of_steps=90,
                                           command=self.on_slider_change);
        self.slider_thresh.set(0.7);
        self.slider_thresh.grid(row=1, column=1, pady=5, padx=10);
        self.retrieve_button = ctk.CTkButton(self.controls_frame, text="Retrieve Similar Images",
                                             command=self.retrieve_images);
        self.retrieve_button.grid(row=0, column=2, rowspan=2, pady=20, padx=20)
        self.action_buttons_frame = ctk.CTkFrame(self.main_content_frame, fg_color="transparent");
        self.action_buttons_frame.pack(pady=10);
        self.save_button = ctk.CTkButton(self.action_buttons_frame, text="Save Results", command=self.save_results,
                                         state="disabled");
        self.save_button.pack(side="left", padx=10)
        self.carousel_frame = ctk.CTkScrollableFrame(self.main_content_frame, orientation="horizontal", height=200);
        self.carousel_frame.pack(pady=10, fill="x", expand=True, padx=20)

    def update_status(self, msg, color="white"):
        self.status_label.configure(text=msg, text_color=color)

    def on_device_change(self):
        self.device = torch.device(
            self.device_var.get()); self.model = None; self.model_full_path = None; self.model_path_label.configure(
            text="No model weights loaded"); self.update_status("Device changed. Reload model.",
                                                                "yellow"); self.check_all_button_states()

    def on_model_change(self, name):
        self.model_name = name; self.model = None; self.feature_vectors = None; self.model_path_label.configure(
            text="No model weights loaded"); self.feature_path_label.configure(
            text="No features loaded"); self.update_status(f"Selected: {self.model_name}",
                                                           "white"); self.check_all_button_states()

    def check_device(self):
        if not torch.cuda.is_available(): self.radio_gpu.configure(state="disabled"); self.device_var.set("cpu")
        self.update_status(f"{self.device.type.upper()} active", "green")

    def check_all_button_states(self):
        # --- Logic for Retrieval and Saving (unchanged) ---
        can_retrieve = all([self.query_image_path, self.model, self.feature_vectors is not None])
        self.retrieve_button.configure(state="normal" if can_retrieve else "disabled")

        can_save = bool(self.current_results)
        self.save_button.configure(state="normal" if can_save else "disabled")

        # --- CORRECTED Logic for Action Buttons ---

        # The "Load Features" button should always be available.
        # It is only temporarily disabled by the extraction process itself.
        self.load_features_btn.configure(state="normal")

        # The "Extract Features" button should ONLY be enabled AFTER model weights are loaded.
        can_extract = self.model is not None
        self.extract_btn.configure(state="normal" if can_extract else "disabled")

    def load_model_from_file(self):
        fp = filedialog.askopenfilename(title=f"Select {self.model_name} Model", filetypes=[("PyTorch Model", "*.pth")])
        if fp: self.model_full_path = fp;self.model_path_label.configure(text=os.path.basename(fp));self.update_status(
            f"Loading {self.model_name}...", "yellow");threading.Thread(target=self._load_model_threaded, args=(fp,),
                                                                        daemon=True).start()

    def _load_model_threaded(self, path):
        try:
            self.model = FeatureExtractor(self.model_name, path, self.device); self.after(0, self.update_status,
                                                                                          f"{self.model_name} loaded.",
                                                                                          "green")
        except Exception as e:
            self.model = None; self.after(0,
                                          lambda exc=e: messagebox.showerror("Model Load Error", str(exc))); self.after(
                0, self.update_status, "Model load failed.", "red")
        self.after(0, self.check_all_button_states)

    def load_features_from_file(self):
        fp = filedialog.askopenfilename(title="Select Feature File", filetypes=[("Pickle file", "*.pkl")]);
        if not fp: return
        root_folder = filedialog.askdirectory(title="Select image dataset root folder")
        if not root_folder: return
        dim = CONFIG["model_dims"].get(self.model_name);
        if not dim: messagebox.showerror("Config Error", f"No feature dimension for {self.model_name}."); return
        self.update_status("Loading features...", "yellow");
        self.feature_path_label.configure(text=os.path.basename(fp));
        threading.Thread(target=self._load_features_threaded, args=(fp, dim, root_folder), daemon=True).start()

    def _load_features_threaded(self, path, dim, root):
        features, paths, labels = load_feature_vectors(path, dim, root)
        if features is not None:
            self.feature_vectors, self.image_paths, self.image_labels = features, paths, labels; self.after(0,
                                                                                                            self.update_status,
                                                                                                            f"Loaded {len(features)} features.",
                                                                                                            "green")
        else:
            self.feature_vectors, self.image_paths, self.image_labels = None, None, None; self.after(0,
                                                                                                     lambda: messagebox.showerror(
                                                                                                         "Feature Load Error",
                                                                                                         "Failed to load.")); self.after(
                0, self.update_status, "Feature load failed.", "red")
        self.after(0, self.check_all_button_states)

    def browse_image(self, e=None):
        fp = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")]); (
                    fp and self.display_query_image(fp))

    def load_dropped_image(self, e):
        fp = e.data.strip('{}'); (os.path.isfile(fp) and self.display_query_image(fp))

    def display_query_image(self, path):
        self.query_image_path = path;self.query_feature_vector = None;img = Image.open(path).convert(
            "RGB");img.thumbnail((200, 200));self.query_image_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(
        img.width, img.height));self.query_image_canvas.configure(image=self.query_image_tk,
                                                                  text="");self.upload_label.configure(
            text=f"📁 {os.path.basename(path)}");self.check_all_button_states()

    def on_slider_change(self, v):
        self.slider_k_label.configure(text=f"Top K: {int(self.slider_k.get())}");self.slider_thresh_label.configure(
            text=f"Similarity: {self.slider_thresh.get():.2f}")

    def retrieve_images(self):
        self.retrieve_button.configure(state="disabled", text="Searching...");[w.destroy() for w in
                                                                               self.carousel_frame.winfo_children()];self.current_results = [];threading.Thread(
            target=self._retrieve_and_display_threaded, daemon=True).start()

    def _retrieve_and_display_threaded(self):
        try:
            tensor = preprocess_image(self.query_image_path);
            vec = self.model(tensor);
            self.query_feature_vector = vec
            results = search_similar(vec, self.feature_vectors, self.image_paths, k=int(self.slider_k.get()),
                                     thresh=self.slider_thresh.get())
            self.current_results = results;
            self.after(0, self._display_results, results)
        except Exception as e:
            self.after(0, lambda exc=e: messagebox.showerror("Search Error", str(exc)))
        finally:
            self.after(0, self.retrieve_button.configure,
                       {"state": "normal", "text": "Retrieve Similar Images"});self.after(0,
                                                                                          self.check_all_button_states)

    def _display_results(self, results):
        if not results:
            ctk.CTkLabel(self.carousel_frame, text="No similar images", font=("Helvetica", 16)).pack(
                expand=True);self.update_status("No results found.", "yellow")
        else:
            self.result_images = [];[self._add_result_image(p, s) for p, s in results];self.update_status(
                f"Found {len(results)} images.", "green")

    def _add_result_image(self, p, s):
        img = Image.open(p).convert("RGB");img.thumbnail((150, 150));tk_img = ctk.CTkImage(light_image=img,
                                                                                           dark_image=img, size=(
            img.width, img.height));self.result_images.append(tk_img);c = ctk.CTkFrame(self.carousel_frame,
                                                                                       fg_color="transparent");c.pack(
            side="left", padx=10, pady=5);ctk.CTkLabel(c, image=tk_img, text="").pack();ctk.CTkLabel(c,
                                                                                                     text=f"Sim: {s:.2f}",
                                                                                                     font=("Helvetica",
                                                                                                           12)).pack()

    def save_results(self):
        if not self.current_results: messagebox.showwarning("No Results", "Nothing to save."); return
        save_dir = filedialog.askdirectory(title="Select Save Folder");
        if not save_dir: return
        try:
            shutil.copy(self.query_image_path,
                        os.path.join(save_dir, f"query_{os.path.basename(self.query_image_path)}"));[
                shutil.copy(p, os.path.join(save_dir, f"retrieved_{i + 1}_{os.path.basename(p)}")) for i, (p, _) in
                enumerate(self.current_results)];messagebox.showinfo("Success",
                                                                     f"Saved {len(self.current_results) + 1} images.")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def start_extraction(self):
        dataset_path = filedialog.askdirectory(title="Select Dataset Folder to Extract From")
        if not dataset_path: self.update_status("Extraction cancelled."); return
        save_path = filedialog.asksaveasfilename(title="Save Extracted Features As", defaultextension=".pkl",
                                                 filetypes=[("Pickle file", "*.pkl")])
        if not save_path: self.update_status("Extraction cancelled."); return

        self.extract_btn.configure(state="disabled")
        self.load_features_btn.configure(state="disabled")

        threading.Thread(target=self._run_extraction_threaded, args=(dataset_path, save_path), daemon=True).start()

    def _run_extraction_threaded(self, dataset_path, save_path):
        self.update_status("Starting feature extraction...", "yellow")

        # This function call remains the same
        success, message = run_feature_extraction(
            self.model, dataset_path, save_path
        )

        # --- THIS IS THE CORRECTED PART ---
        # After the process finishes, show the result message
        self.after(0, lambda: messagebox.showinfo("Extraction Status", message))

        # Then, call the main state-checking function. This is the single source
        # of truth for what state the buttons should be in.
        self.after(0, self.check_all_button_states)

        # Finally, update the status label
        self.after(0, self.update_status, "Extraction finished." if success else "Extraction failed.",
                   "green" if success else "red")

if __name__ == "__main__":
    app = ImageRetrievalApp()
    app.mainloop()