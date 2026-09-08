"""
Application de prise de notes de cours magistraux par transcription vocale.

Phase actuelle : interface + logique développées et testées sous Windows,
avec le backend faster-whisper (cross-platform, CPU). Le backend final
whisper.cpp + Core ML (Apple Silicon) sera branché et testé sur le
MacBook Air M2 ce week-end — voir transcriber.py.
"""

from __future__ import annotations

import threading
import tkinter as tk

import customtkinter as ctk

from audio_capture import AudioRecorder, TranscriptionWorker, list_input_devices
from session_manager import Session
from transcriber import get_default_backend

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LectureNotesApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Echo")
        self.geometry("700x600")
        self.minsize(500, 400)

        self.session: Session | None = None
        self.recorder: AudioRecorder | None = None
        self.worker: TranscriptionWorker | None = None
        self.backend = None  # chargé en tâche de fond (le modèle Whisper est lourd à init)
        self._backend_ready = threading.Event()

        self._build_ui()
        self._load_devices()
        self._load_backend_async()

    # ---------- UI ----------

    def _build_ui(self):
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(top_frame, text="Titre du cours :").grid(row=0, column=0, padx=(0, 8), pady=6, sticky="w")
        self.title_entry = ctk.CTkEntry(top_frame, placeholder_text="ex: Histoire - Révolution industrielle")
        self.title_entry.grid(row=0, column=1, padx=(0, 8), pady=6, sticky="ew")

        ctk.CTkLabel(top_frame, text="Micro :").grid(row=1, column=0, padx=(0, 8), pady=6, sticky="w")
        self.device_menu = ctk.CTkOptionMenu(top_frame, values=["Chargement..."])
        self.device_menu.grid(row=1, column=1, padx=(0, 8), pady=6, sticky="ew")

        top_frame.columnconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(self, text="Chargement du modèle de transcription...", text_color="gray")
        self.status_label.pack(pady=(0, 8))

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=(0, 8))

        self.start_button = ctk.CTkButton(
            button_frame, text="Démarrer", command=self._on_start, state="disabled"
        )
        self.start_button.grid(row=0, column=0, padx=6)

        self.stop_button = ctk.CTkButton(
            button_frame, text="Arrêter", command=self._on_stop, state="disabled", fg_color="#8b3a3a"
        )
        self.stop_button.grid(row=0, column=1, padx=6)

        self.transcript_box = ctk.CTkTextbox(self, wrap="word", font=("Segoe UI", 13))
        self.transcript_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.transcript_box.configure(state="disabled")

    def _load_devices(self):
        devices = list_input_devices()
        if not devices:
            self.device_menu.configure(values=["Aucun micro détecté"])
            return
        self._devices = devices
        names = [d["name"] for d in devices]
        self.device_menu.configure(values=names)
        self.device_menu.set(names[0])

    def _load_backend_async(self):
        def _load():
            self.backend = get_default_backend()
            self._backend_ready.set()
            self.after(0, self._on_backend_ready)

        threading.Thread(target=_load, daemon=True).start()

    def _on_backend_ready(self):
        self.status_label.configure(text="Prêt.")
        self.start_button.configure(state="normal")

    # ---------- Logique session ----------

    def _on_start(self):
        title = self.title_entry.get().strip() or "Cours sans titre"
        self.session = Session(title)

        selected_name = self.device_menu.get()
        device_index = next(
            (d["index"] for d in getattr(self, "_devices", []) if d["name"] == selected_name),
            None,
        )

        self.worker = TranscriptionWorker(self.backend, on_text_ready=self._on_text_ready)
        self.worker.start()

        self.recorder = AudioRecorder(device_index, on_segment_ready=self.worker.submit)
        self.recorder.start()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.title_entry.configure(state="disabled")
        self.device_menu.configure(state="disabled")
        self.status_label.configure(text=f"Enregistrement en cours — {self.session.path.name}")

        self._clear_transcript()

    def _on_stop(self):
        if self.recorder:
            self.recorder.stop()
        if self.worker:
            self.worker.stop()
        if self.session:
            self.session.close()
            self.status_label.configure(text=f"Session sauvegardée : {self.session.path}")

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.title_entry.configure(state="normal")
        self.device_menu.configure(state="normal")

    def _on_text_ready(self, text: str, started_at: float):
        # appelé depuis le thread de transcription -> on repasse sur le thread UI
        self.after(0, lambda: self._append_transcript(text, started_at))
        if self.session:
            self.session.append_segment(text, started_at)

    # ---------- Helpers UI thread-safe ----------

    def _append_transcript(self, text: str, started_at: float):
        elapsed = started_at - self.session.started_at.timestamp() if self.session else 0
        minutes, seconds = divmod(max(0, int(elapsed)), 60)
        self.transcript_box.configure(state="normal")
        self.transcript_box.insert("end", f"[{minutes:02d}:{seconds:02d}] {text}\n\n")
        self.transcript_box.see("end")
        self.transcript_box.configure(state="disabled")

    def _clear_transcript(self):
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.configure(state="disabled")

    def on_closing(self):
        if self.recorder is not None and self.stop_button.cget("state") == "normal":
            self._on_stop()
        self.destroy()


if __name__ == "__main__":
    app = LectureNotesApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()