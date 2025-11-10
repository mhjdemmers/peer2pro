from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from main import run_matching


class MatchingGUI(tk.Tk):
    """Simple desktop wrapper around the run_matching pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Peer2Pro Matching")
        self.geometry("640x320")

        self.students_var = tk.StringVar()
        self.mentors1_var = tk.StringVar()
        self.mentors2_var = tk.StringVar()
        self.export_var = tk.StringVar(value=str(Path("./DATASETS/matches.csv")))
        self.n_type1_var = tk.StringVar(value="4")
        self.n_type2_var = tk.StringVar(value="")

        self._build_layout()

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)

        current_row = 0
        current_row = self._add_file_picker(
            row=current_row,
            label="Students CSV:",
            variable=self.students_var,
        )
        current_row = self._add_file_picker(
            row=current_row,
            label="Mentors Type 1 CSV:",
            variable=self.mentors1_var,
        )
        current_row = self._add_file_picker(
            row=current_row,
            label="Mentors Type 2 CSV (optional):",
            variable=self.mentors2_var,
        )

        export_frame = tk.Frame(self)
        export_entry = tk.Entry(export_frame, textvariable=self.export_var, width=60)
        export_entry.pack(side="left", fill="x", expand=True)
        tk.Button(
            export_frame,
            text="Save as...",
            command=lambda: self._set_save_path(self.export_var),
        ).pack(side="right")
        tk.Label(self, text="Export CSV:").grid(row=current_row, column=0, sticky="w", padx=6, pady=6)
        export_frame.grid(row=current_row, column=1, sticky="we", padx=6, pady=6)
        current_row += 1

        tk.Label(self, text="Type 1 mentors per student:").grid(row=current_row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(self, textvariable=self.n_type1_var, width=10).grid(
            row=current_row,
            column=1,
            sticky="w",
            padx=6,
            pady=6,
        )
        current_row += 1

        tk.Label(self, text="Type 2 mentors per student (optional):").grid(row=current_row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(self, textvariable=self.n_type2_var, width=10).grid(
            row=current_row,
            column=1,
            sticky="w",
            padx=6,
            pady=6,
        )
        current_row += 1

        self.run_button = tk.Button(
            self,
            text="Run Matching",
            bg="#4caf50",
            fg="white",
            command=self._start_run,
        )
        self.run_button.grid(row=current_row, column=0, columnspan=2, sticky="we", padx=6, pady=10)
        current_row += 1

        self.log_box = tk.Text(self, height=6, state="disabled")
        self.log_box.grid(row=current_row, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        self.rowconfigure(current_row, weight=1)

    def _add_file_picker(self, *, row: int, label: str, variable: tk.StringVar) -> int:
        tk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        frame = tk.Frame(self)
        entry = tk.Entry(frame, textvariable=variable, width=60)
        entry.pack(side="left", fill="x", expand=True)
        tk.Button(
            frame,
            text="Browse...",
            command=lambda: self._set_open_path(variable),
        ).pack(side="right")
        frame.grid(row=row, column=1, sticky="we", padx=6, pady=6)
        return row + 1

    def _set_open_path(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if path:
            variable.set(path)

    def _set_save_path(self, variable: tk.StringVar) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if path:
            variable.set(path)

    def _append_log(self, message: str) -> None:
        self.log_box.config(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _start_run(self) -> None:
        if not self.students_var.get():
            messagebox.showerror("Missing file", "Please select a students CSV file.")
            return
        if not self.mentors1_var.get():
            messagebox.showerror("Missing file", "Please select a mentors type 1 CSV file.")
            return

        try:
            n_type1 = int(self.n_type1_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Type 1 mentors per student must be an integer.")
            return

        n_type2: int | None = None
        if self.mentors2_var.get().strip() and self.n_type2_var.get().strip():
            try:
                n_type2 = int(self.n_type2_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    "Type 2 mentors per student must be an integer or left empty.",
                )
                return

        run_args = dict(
            students_input_path=self.students_var.get(),
            mentors_type1_path=self.mentors1_var.get(),
            mentors_type2_path=self.mentors2_var.get().strip() or None,
            n_type1=n_type1,
            n_type2=n_type2,
            export_path=self.export_var.get(),
        )

        self.run_button.config(state="disabled")
        self._append_log("Starting matching...")

        thread = threading.Thread(target=self._run_matching_thread, args=(run_args,), daemon=True)
        thread.start()

    def _run_matching_thread(self, run_args: dict) -> None:
        try:
            result = run_matching(**run_args)
            self._append_log("Matching completed.")
            if result is None:
                self._append_log("No matches returned.")
            else:
                self._append_log(f"Matches exported to: {run_args['export_path']}")
            messagebox.showinfo("Done", "Matching finished successfully.")
        except SystemExit as exc:
            self._append_log(f"SystemExit: {exc}")
            messagebox.showerror("Error", str(exc))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self._append_log("Error during matching:\n" + tb)
            messagebox.showerror("Error", f"An error occurred:\n{exc}")
        finally:
            self.run_button.config(state="normal")


def main() -> None:
    app = MatchingGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
