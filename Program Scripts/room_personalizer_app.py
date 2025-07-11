import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
from data_preprocessing import transform_new
from pathlib import Path 

BASE_DIR  = Path(__file__).resolve().parents[1]   
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Main application class for the Room Personalizer GUI
class RoomPersonalizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # window title 
        self.title("Room Personalizer")  
        # fixed window size
        self.geometry("500x450")        
        # disable resizing
        self.resizable(False, False)     

        # Apply a clean ttk theme for widgets
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('TCombobox', font=('Segoe UI', 10))
        style.configure('TSpinbox', font=('Segoe UI', 10))

        # Load ML artifacts and room data once at startup
        self._load_artifacts()

        # Create container frames for inputs, button, and output display
        self.input_frame = ttk.LabelFrame(self, text="Guest Details", padding=10)
        self.input_frame.pack(fill='x', padx=10, pady=5)
        self.button_frame = ttk.Frame(self, padding=(10, 0))
        self.button_frame.pack(fill='x')
        self.output_frame = ttk.LabelFrame(self, text="Suggestions", padding=10)
        self.output_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Build UI elements in each section
        self._build_inputs()
        self._build_button()
        self._build_output()

    def _load_artifacts(self):
    # modeller
        self.kmeans           = joblib.load(MODEL_DIR / "kmeans.pkl")           
        self.temp_model       = joblib.load(MODEL_DIR / "temp_reg.pkl")        
        self.dimmer_model     = joblib.load(MODEL_DIR / "dimmer_reg.pkl")       
        self.empty_start_model= joblib.load(MODEL_DIR / "empty_start_reg.pkl") 
        self.empty_end_model  = joblib.load(MODEL_DIR / "empty_end_reg.pkl")   
        self.minibar_lookup   = joblib.load(MODEL_DIR / "minibar_rules.pkl")   
        self.dir_model        = joblib.load(MODEL_DIR / "dir_model.pkl")       
        self.dir_encoder      = joblib.load(MODEL_DIR / "dir_encoder.pkl")     

    # veri dosyası
        self.rooms_df = pd.read_excel(                                        
        DATA_DIR / "final_synced_main_guest_names_dataset.xlsx",
        usecols=["Room Number", "Room Type", "Room Direction"]
    )

        # Add lowercase columns for case-insensitive matching
        self.rooms_df['_type_lower'] = self.rooms_df['Room Type'].str.strip().str.lower()
        self.rooms_df['_dir_lower'] = self.rooms_df['Room Direction'].str.strip().str.lower()

    def _build_inputs(self):
        # Guest count input
        ttk.Label(self.input_frame, text="Number of guests:").grid(row=0, column=0, sticky='w')
        self.guest_count = ttk.Spinbox(
            self.input_frame, from_=1, to=10, width=5, justify='center'
        )
        self.guest_count.set(1)
        self.guest_count.grid(row=0, column=1, sticky='w', padx=5)

        # Gender selector
        ttk.Label(self.input_frame, text="Gender:").grid(row=0, column=2, sticky='w', padx=(20,0))
        self.gender_var = tk.StringVar(value="Female")
        self.gender_cb = ttk.Combobox(
            self.input_frame, textvariable=self.gender_var,
            values=["Female", "Male"], state="readonly", width=10
        )
        self.gender_cb.grid(row=0, column=3, sticky='w', padx=5)

        # Country entry
        ttk.Label(self.input_frame, text="Country:").grid(row=1, column=0, sticky='w', pady=4)
        self.country_entry = ttk.Entry(self.input_frame)
        self.country_entry.grid(row=1, column=1, columnspan=3, sticky='ew', padx=5)

        # Room type dropdown
        ttk.Label(self.input_frame, text="Room type:").grid(row=2, column=0, sticky='w')
        types = sorted(self.rooms_df['Room Type'].unique())
        self.room_type_var = tk.StringVar(value=types[0])
        self.room_type_cb = ttk.Combobox(
            self.input_frame, textvariable=self.room_type_var,
            values=types, state="readonly"
        )
        self.room_type_cb.grid(row=2, column=1, columnspan=3, sticky='ew', padx=5)

        # Entrance time input
        ttk.Label(
            self.input_frame,
            text="Entrance time (YYYY-MM-DD HH:MM):"
        ).grid(row=3, column=0, columnspan=4, sticky='w', pady=(10,4))
        self.entrance_entry = ttk.Entry(self.input_frame)
        self.entrance_entry.insert(0, "2025-07-19 15:30")
        self.entrance_entry.grid(row=4, column=0, columnspan=4, sticky='ew', padx=5)

        # Make all columns share extra space evenly
        for c in range(4):
            self.input_frame.columnconfigure(c, weight=1)

    def _build_button(self):
        # Button to trigger suggestions
        self.suggest_btn = ttk.Button(
            self.button_frame, text="Suggest Settings", command=self.on_suggest
        )
        self.suggest_btn.pack(pady=10)

    def _build_output(self):
        # Table-like display for results
        cols = ("Parameter", "Value")
        self.tree = ttk.Treeview(
            self.output_frame, columns=cols, show='headings', height=8
        )
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center', width=180)
        self.tree.pack(fill='both', expand=True)

    def _format_interval(self, start_hr, end_hr):
        # Ensure start is before end and format as H:00–H:00
        if end_hr <= start_hr:
            start_hr, end_hr = end_hr, start_hr
        sh = int(np.floor(start_hr)) % 24
        eh = int(np.ceil(end_hr)) % 24
        if eh <= sh:
            eh = (sh + 1) % 24
        return f"{sh:02d}.00–{eh:02d}.00"

    def on_suggest(self):
        # Read and validate inputs
        try:
            guests = int(self.guest_count.get())
        except ValueError:
            messagebox.showerror("Invalid", "Please enter a valid number of guests.")
            return

        # Build record dict from inputs
        record = {
            "Guests Count": guests,
            "Guest Gender": self.gender_var.get(),
            "Guest Country": self.country_entry.get().strip(),
            "Room Type": self.room_type_var.get(),
            "Entrance Time": self.entrance_entry.get().strip()
        }

        # Generate suggestions or show error
        try:
            rec = self._suggest(record)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Display each suggestion in the table
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for k, v in rec.items():
            self.tree.insert('', 'end', values=(k, v))

    def _suggest(self, record, n_minibar=3):
        # Convert inputs and predict settings
        X_sp = transform_new(record)
        cid = int(self.kmeans.predict(X_sp)[0])
        X_full = np.hstack([X_sp.toarray(), [[cid]]])

        temp_c = float(self.temp_model.predict(X_full)[0])
        dimmer = int(np.clip(self.dimmer_model.predict(X_full)[0], 0, 100))
        s_hr = float(self.empty_start_model.predict(X_full)[0])
        e_hr = float(self.empty_end_model.predict(X_full)[0])
        interval = self._format_interval(s_hr, e_hr)
        items = self.minibar_lookup.get(cid, ["Sparkling Water"])[:n_minibar]

        # Predict room direction and select room number
        code = self.dir_model.predict(X_full)[0]
        dir_label = self.dir_encoder.inverse_transform([code])[0]
        # Case-insensitive matching against lowercase helper columns
        rtype = record['Room Type'].strip().lower()
        dlow = dir_label.strip().lower()
        cand = self.rooms_df[
            (self.rooms_df['_type_lower'] == rtype) &
            (self.rooms_df['_dir_lower'] == dlow)
        ]
        if cand.empty:
            cand = self.rooms_df[self.rooms_df['_type_lower'] == rtype]
        if cand.empty:
            cand = self.rooms_df
        room_num = cand.sample(1)['Room Number'].iloc[0]

        return {
            "Temperature (°C)": f"{temp_c:.1f}",
            "Dimmer Level (%)": str(dimmer),
            "Empty Window": interval,
            "Minibar": ", ".join(items),
            "Direction": dir_label,
            "Room No.": str(room_num)
        }

if __name__ == "__main__":
    app = RoomPersonalizerApp()
    app.mainloop()
