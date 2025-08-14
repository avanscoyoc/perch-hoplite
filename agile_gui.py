import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd

# --- Perch-Hoplite Imports ---
# (These are copied from the notebook)
from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
from perch_hoplite.agile import source_info
from perch_hoplite.db import brutalism
from perch_hoplite.db import score_functions
from perch_hoplite.db import search_results
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.zoo import model_configs
from perch_hoplite.agile import embedding_display # For QueryDisplay

class AgileApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Agile Labeling GUI")
        self.geometry("800x800")

        # --- State Variables ---
        self.db = None
        self.embedding_model = None
        self.audio_filepath_loader = None
        self.data_manager = None
        self.linear_classifier = None
        self.annotator_id = 'linnaeus_gui'

        self.search_results = []
        self.result_labels = {} # {embedding_id: 'pos'/'neg'/None}
        self.current_result_index = -1

        # --- Load Data ---
        # NOTE: Adjust this path to your database location
        self.db_path = '/mnt/mojave/Birds'
        self.load_model_and_db()

        # --- UI Setup ---
        self._setup_widgets()
        self._bind_hotkeys()

    def _setup_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Control Frame ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Query URI (e.g., xc971940):").pack(side=tk.LEFT, padx=5)
        self.query_uri_entry = ttk.Entry(control_frame, width=20)
        self.query_uri_entry.insert(0, 'xc971940')
        self.query_uri_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Query Label:").pack(side=tk.LEFT, padx=5)
        self.query_label_entry = ttk.Entry(control_frame, width=15)
        self.query_label_entry.insert(0, 'Coyote')
        self.query_label_entry.pack(side=tk.LEFT, padx=5)

        self.search_button = ttk.Button(control_frame, text="Search", command=self.run_search)
        self.search_button.pack(side=tk.LEFT, padx=10)

        # --- Spectrogram Display ---
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Navigation and Labeling Frame ---
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)

        self.prev_button = ttk.Button(nav_frame, text="< Prev (Left)", command=self.show_prev_result)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.play_button = ttk.Button(nav_frame, text="Play Audio (P)", command=self.play_current_audio)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.label_var = tk.StringVar()
        self.pos_radio = ttk.Radiobutton(nav_frame, text="Positive (Space)", variable=self.label_var, value='pos', command=self.set_label)
        self.neg_radio = ttk.Radiobutton(nav_frame, text="Negative (N)", variable=self.label_var, value='neg', command=self.set_label)
        self.pos_radio.pack(side=tk.LEFT, padx=10)
        self.neg_radio.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(nav_frame, text="Next (Right) >", command=self.show_next_result)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(nav_frame, text="Status: Idle")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # --- Bottom Action Frame ---
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)

        self.save_button = ttk.Button(action_frame, text="Submit Labels", command=self.save_labels)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.train_button = ttk.Button(action_frame, text="Train & Review Classifier", command=self.train_and_review)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.write_button = ttk.Button(action_frame, text="Write Inference CSV", command=self.write_inference_csv)
        self.write_button.pack(side=tk.LEFT, padx=5)

    def _bind_hotkeys(self):
        self.bind('<Left>', lambda e: self.show_prev_result())
        self.bind('<Right>', lambda e: self.show_next_result())
        self.bind('<space>', lambda e: self.toggle_positive_label())
        self.bind('n', lambda e: self.toggle_negative_label())
        self.bind('p', lambda e: self.play_current_audio())

    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        self.update_idletasks()

    def load_model_and_db(self):
        self.update_status("Loading database and model...")
        try:
            self.db = sqlite_usearch_impl.SQLiteUsearchDB.create(self.db_path)
            db_model_config = self.db.get_metadata('model_config')
            embed_config = self.db.get_metadata('audio_sources')
            model_class = model_configs.get_model_class(db_model_config.model_key)
            self.embedding_model = model_class.from_config(db_model_config.model_config)
            audio_sources = source_info.AudioSources.from_config_dict(embed_config)
            window_size_s = getattr(self.embedding_model, 'window_size_s', 5.0)
            self.audio_filepath_loader = audio_loader.make_filepath_loader(
                audio_sources=audio_sources,
                window_size_s=window_size_s,
                sample_rate_hz=self.embedding_model.sample_rate,
            )
            self.update_status(f"DB loaded with {self.db.count_embeddings()} embeddings.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database/model:\n{e}")
            self.destroy()

    def run_search(self):
        self.update_status("Searching...")
        uri = self.query_uri_entry.get()
        if not uri:
            messagebox.showwarning("Input required", "Please enter a query URI.")
            return

        try:
            query = embedding_display.QueryDisplay(uri=uri, window_size_s=5.0, sample_rate_hz=self.embedding_model.sample_rate)
            query_embedding = self.embedding_model.embed(query.get_audio_window()).embeddings[0, 0]

            score_fn = score_functions.get_score_fn('dot', target_score=None)
            results, _ = brutalism.threaded_brute_search(self.db, query_embedding, 50, score_fn=score_fn)

            self.search_results = results.search_results
            self.result_labels = {r.embedding_id: None for r in self.search_results}
            self.current_result_index = 0
            self.display_current_result()
            self.update_status(f"Search complete. Found {len(self.search_results)} results.")
        except Exception as e:
            self.update_status("Search failed.")
            messagebox.showerror("Search Error", f"An error occurred during search:\n{e}")

    def display_current_result(self):
        if not self.search_results or self.current_result_index < 0:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No result to display.', ha='center', va='center')
            self.canvas.draw()
            return

        result = self.search_results[self.current_result_index]
        audio_window, sr = self.audio_filepath_loader(result.filename, result.offset_s)

        self.ax.clear()
        self.ax.specgram(audio_window, Fs=sr, cmap='viridis')
        self.ax.set_title(f"Result {self.current_result_index + 1}/{len(self.search_results)}\nID: {result.embedding_id} File: {os.path.basename(result.filename)}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.fig.tight_layout()
        self.canvas.draw()

        # Update radio buttons
        current_label = self.result_labels.get(result.embedding_id)
        self.label_var.set(current_label)

    def show_next_result(self):
        if self.current_result_index < len(self.search_results) - 1:
            self.current_result_index += 1
            self.display_current_result()

    def show_prev_result(self):
        if self.current_result_index > 0:
            self.current_result_index -= 1
            self.display_current_result()

    def play_current_audio(self):
        if self.current_result_index != -1:
            result = self.search_results[self.current_result_index]
            audio_window, sr = self.audio_filepath_loader(result.filename, result.offset_s)
            self.update_status(f"Playing audio for result {self.current_result_index + 1}...")
            sd.play(audio_window, samplerate=sr)
            sd.wait()
            self.update_status(f"Status: Idle")


    def set_label(self):
        if self.current_result_index != -1:
            result = self.search_results[self.current_result_index]
            self.result_labels[result.embedding_id] = self.label_var.get()

    def toggle_positive_label(self):
        if self.current_result_index != -1:
            result = self.search_results[self.current_result_index]
            if self.result_labels.get(result.embedding_id) == 'pos':
                self.label_var.set('') # Unset
            else:
                self.label_var.set('pos')
            self.set_label()

    def toggle_negative_label(self):
        if self.current_result_index != -1:
            result = self.search_results[self.current_result_index]
            if self.result_labels.get(result.embedding_id) == 'neg':
                self.label_var.set('') # Unset
            else:
                self.label_var.set('neg')
            self.set_label()

    def save_labels(self):
        self.update_status("Saving labels...")
        new_lbls, prev_lbls = 0, 0
        for emb_id, label_val in self.result_labels.items():
            if label_val is None:
                continue
            
            label_type = source_info.LabelType.POSITIVE if label_val == 'pos' else source_info.LabelType.NEGATIVE
            lbl = source_info.Label(
                embedding_id=emb_id,
                label_str=self.query_label_entry.get(),
                label_type=label_type,
                annotator_id=self.annotator_id
            )
            check = self.db.insert_label(lbl, skip_duplicates=True)
            new_lbls += check
            prev_lbls += (1 - check)
        
        self.update_status("Labels saved.")
        messagebox.showinfo("Labels Saved", f"New labels: {new_lbls}\nPreviously existing labels: {prev_lbls}")

    def _get_data_manager(self):
        return classifier_data.AgileDataManager(
            target_labels=None, # Auto-populate from DB
            db=self.db,
            train_ratio=0.9,
            min_eval_examples=1,
            batch_size=128,
            weak_negatives_batch_size=128,
            rng=np.random.default_rng(seed=5)
        )

    def train_and_review(self):
        self.update_status("Training classifier...")
        try:
            self.data_manager = self._get_data_manager()
            target_labels = self.data_manager.get_target_labels()
            if not target_labels:
                messagebox.showerror("Training Error", "No labels found in DB to train on.")
                self.update_status("Training failed.")
                return

            self.linear_classifier, eval_scores = classifier.train_linear_classifier(
                data_manager=self.data_manager,
                learning_rate=1e-3,
                weak_neg_weight=0.05,
                num_train_steps=128,
            )
            self.update_status("Training complete. Reviewing results...")

            # Now review the results from the new classifier
            target_label = self.query_label_entry.get()
            if target_label not in target_labels:
                messagebox.showwarning("Review Warning", f"Label '{target_label}' not in trained labels. Using first available: {target_labels[0]}")
                target_label = target_labels[0]

            target_label_idx = target_labels.index(target_label)
            class_query = self.linear_classifier.beta[:, target_label_idx]
            bias = self.linear_classifier.beta_bias[target_label_idx]
            score_fn = score_functions.get_score_fn('dot', bias=bias, target_score=None)
            
            results, _ = brutalism.threaded_brute_search(
                self.db, class_query, 50, score_fn=score_fn, sample_size=1_000_000
            )
            
            self.search_results = results.search_results
            self.result_labels = {r.embedding_id: None for r in self.search_results}
            self.current_result_index = 0
            self.display_current_result()
            self.update_status(f"Review search complete. Found {len(self.search_results)} results.")

        except Exception as e:
            self.update_status("Training/Review failed.")
            messagebox.showerror("Error", f"An error occurred during training/review:\n{e}")

    def write_inference_csv(self):
        if not self.linear_classifier:
            messagebox.showerror("Error", "No classifier has been trained yet.")
            return
        
        output_path = simpledialog.askstring("Output File", "Enter the path for the output CSV file:",
                                             initialvalue=os.path.join(self.db_path, "inference_results.csv"))
        if not output_path:
            return

        self.update_status(f"Writing inference to {output_path}...")
        try:
            classifier.write_inference_csv(
                self.linear_classifier, self.db, output_path, logit_threshold=1.0, labels=None
            )
            self.update_status("Inference CSV written.")
            messagebox.showinfo("Success", f"Inference results saved to:\n{output_path}")
        except Exception as e:
            self.update_status("Write inference failed.")
            messagebox.showerror("Error", f"Failed to write inference CSV:\n{e}")


if __name__ == "__main__":
    app = AgileApp()
    app.mainloop()