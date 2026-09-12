"""
Application de prise de notes de cours magistraux par transcription vocale.

Interface thème "HUD" (fond sombre, accents orange/teal, typo monospace,
coins nets) — inspirée d'esthétiques cockpit/terminal sci-fi.

Phase actuelle : interface + logique développées et testées sous Windows,
avec le backend faster-whisper (cross-platform, CPU). Le backend final
whisper.cpp + Core ML (Apple Silicon) sera branché et testé sur le
MacBook Air M2 ce week-end — voir transcriber.py.
"""

from __future__ import annotations

import re
import threading

import customtkinter as ctk

from audio_capture import AudioRecorder, TranscriptionWorker, list_input_devices
from course_schedule import JOURS, get_current_entry, load_schedule, save_schedule
from session_manager import Session, list_sessions, read_session_content
from transcriber import get_default_backend

# ---------------------------------------------------------------------------
# Palette + typographie "HUD"
# ---------------------------------------------------------------------------

BG_APP = "#0a0d0e"
BG_PANEL = "#11161a"
BG_CARD = "#161d21"
BORDER = "#2b3438"
ORANGE = "#ff6b35"
ORANGE_DIM = "#8c4a2e"
TEAL = "#2dd4bf"
TEAL_DIM = "#3a6b64"
TEXT = "#e6e6e6"
TEXT_DIM = "#6b7680"
RED = "#e6473f"

MONO = "Consolas"

ctk.set_appearance_mode("dark")


def mono_font(size=12, weight="normal"):
    return ctk.CTkFont(family=MONO, size=size, weight=weight)


class SectionHeader(ctk.CTkFrame):
    """Petit en-tête de section : label orange majuscule + ligne de séparation."""

    def __init__(self, master, text: str, accent: str = ORANGE):
        super().__init__(master, fg_color="transparent")
        label = ctk.CTkLabel(
            self,
            text=f"// {text.upper()}",
            font=mono_font(11, "bold"),
            text_color=accent,
            anchor="w",
        )
        label.pack(side="left")
        line = ctk.CTkFrame(self, fg_color=BORDER, height=1)
        line.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=(2, 0))


class StatusChip(ctk.CTkFrame):
    """Puce de statut : carré coloré + texte, façon voyant de tableau de bord."""

    def __init__(self, master, text: str, color: str = TEXT_DIM):
        super().__init__(master, fg_color="transparent")
        self.dot = ctk.CTkLabel(self, text="■", font=mono_font(12), text_color=color, width=14)
        self.dot.pack(side="left")
        self.label = ctk.CTkLabel(self, text=text, font=mono_font(11, "bold"), text_color=TEXT)
        self.label.pack(side="left", padx=(4, 0))

    def set(self, text: str, color: str):
        self.label.configure(text=text)
        self.dot.configure(text_color=color)


class SessionCard(ctk.CTkFrame):
    """Carte cliquable représentant une session passée dans la sidebar."""

    def __init__(self, master, title: str, date_str: str, on_click):
        super().__init__(
            master, fg_color=BG_CARD, corner_radius=0, border_width=1, border_color=BORDER,
            cursor="hand2",
        )
        title_label = ctk.CTkLabel(
            self, text=title.upper(), font=mono_font(11, "bold"), text_color=TEXT,
            anchor="w", justify="left", wraplength=190,
        )
        title_label.pack(fill="x", padx=10, pady=(8, 0))
        date_label = ctk.CTkLabel(
            self, text=date_str, font=mono_font(10), text_color=TEAL, anchor="w",
        )
        date_label.pack(fill="x", padx=10, pady=(2, 8))

        for widget in (self, title_label, date_label):
            widget.bind("<Button-1>", lambda e: on_click())


class ScheduleCard(ctk.CTkFrame):
    """Carte de créneau façon liste de tâches : bandeau coloré, heure en
    accent, titre, infos en sous-texte — toute la carte est cliquable."""

    def __init__(self, master, creneau: dict, on_click):
        super().__init__(
            master, fg_color=BG_CARD, corner_radius=8, cursor="hand2",
        )
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=8)

        header_row = ctk.CTkFrame(inner, fg_color="transparent")
        header_row.pack(fill="x")
        bullet = ctk.CTkLabel(header_row, text="●", font=mono_font(10), text_color=ORANGE, width=12)
        bullet.pack(side="left")
        time_label = ctk.CTkLabel(
            header_row, text=f"{creneau.get('debut', '?')} – {creneau.get('fin', '?')}",
            font=mono_font(10, "bold"), text_color=ORANGE,
        )
        time_label.pack(side="left", padx=(4, 0))
        edit_hint = ctk.CTkLabel(header_row, text="✎", font=mono_font(10), text_color=TEAL_DIM)
        edit_hint.pack(side="right")

        title_label = ctk.CTkLabel(
            inner, text=creneau.get("titre", "(sans titre)"), font=mono_font(11, "bold"),
            text_color=TEXT, anchor="w", justify="left", wraplength=145,
        )
        title_label.pack(fill="x", pady=(4, 0))

        widgets = [self, inner, header_row, bullet, time_label, title_label]

        if creneau.get("info"):
            info_label = ctk.CTkLabel(
                inner, text=creneau["info"], font=mono_font(9), text_color=TEXT_DIM,
                anchor="w", justify="left", wraplength=145,
            )
            info_label.pack(fill="x", pady=(2, 0))
            widgets.append(info_label)

        for widget in widgets:
            widget.bind("<Button-1>", lambda e: on_click())


class EchoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ECHO")
        self.geometry("1000x650")
        self.minsize(760, 500)
        self.configure(fg_color=BG_APP)

        self.session: Session | None = None
        self.recorder: AudioRecorder | None = None
        self.worker: TranscriptionWorker | None = None
        self.backend = None
        self._recording = False
        self._elapsed_seconds = 0
        self._blink_on = True
        self._viewing_past_session = False
        self._auto_filled_title = ""

        self.schedule_view_visible = False
        self.schedule_data: dict = {}
        self.schedule_day_columns: dict[str, ctk.CTkScrollableFrame] = {}

        self._build_ui()
        self._load_devices()
        self._load_backend_async()
        self._refresh_session_list()
        self._apply_current_course()
        self._schedule_check_loop()

    # ------------------------------------------------------------------
    # Construction de l'UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # --- barre du haut ---
        top_bar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=60)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        title_block = ctk.CTkFrame(top_bar, fg_color="transparent")
        title_block.pack(side="left", padx=16)
        ctk.CTkLabel(
            title_block, text="ECHO", font=mono_font(20, "bold"), text_color=ORANGE,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_block, text="// SYSTEME DE TRANSCRIPTION", font=mono_font(10),
            text_color=TEAL_DIM,
        ).pack(anchor="w")

        self.engine_chip = StatusChip(top_bar, "MOTEUR: INITIALISATION", ORANGE_DIM)
        self.engine_chip.pack(side="right", padx=16)

        # --- corps : sidebar + zone principale ---
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # sidebar
        sidebar = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=0, width=230)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        self.sidebar = sidebar

        sidebar_header = SectionHeader(sidebar, "Sessions", ORANGE)
        sidebar_header.pack(fill="x", padx=12, pady=(14, 8))

        self.session_list_frame = ctk.CTkScrollableFrame(
            sidebar, fg_color="transparent",
        )
        self.session_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.back_to_live_button = ctk.CTkButton(
            sidebar, text="< RETOUR AU DIRECT", font=mono_font(10, "bold"),
            fg_color=BG_CARD, hover_color=BORDER, text_color=TEAL, corner_radius=0,
            border_width=1, border_color=TEAL_DIM, command=self._return_to_live,
        )
        # affiché seulement quand on consulte une session passée

        self.schedule_button = ctk.CTkButton(
            sidebar, text="EMPLOI DU TEMPS", font=mono_font(10, "bold"),
            fg_color=BG_CARD, hover_color=BORDER, text_color=ORANGE, corner_radius=0,
            border_width=1, border_color=ORANGE_DIM, command=self._toggle_schedule_view,
        )
        self.schedule_button.pack(fill="x", padx=10, pady=(0, 10), side="bottom")

        # zone principale
        main = ctk.CTkFrame(body, fg_color=BG_APP)
        main.grid(row=0, column=1, sticky="nsew")
        self.main_area = main

        # --- vue "session" (contrôle + transcription), affichée par défaut ---
        self.session_view = ctk.CTkFrame(main, fg_color=BG_APP)
        self.session_view.pack(fill="both", expand=True)

        # panneau de contrôle
        control_card = ctk.CTkFrame(self.session_view, fg_color=BG_PANEL, corner_radius=0, border_width=1, border_color=BORDER)
        control_card.pack(fill="x", padx=16, pady=(16, 8))

        control_header = SectionHeader(control_card, "Nouvelle session", TEAL)
        control_header.pack(fill="x", padx=14, pady=(12, 10))

        row1 = ctk.CTkFrame(control_card, fg_color="transparent")
        row1.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(row1, text="TITRE", font=mono_font(10, "bold"), text_color=TEXT_DIM, width=70, anchor="w").pack(side="left")
        self.title_entry = ctk.CTkEntry(
            row1, placeholder_text="ex: Histoire — Révolution industrielle",
            font=mono_font(12), fg_color=BG_CARD, border_color=BORDER, corner_radius=0,
            text_color=TEXT,
        )
        self.title_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.schedule_info_label = ctk.CTkLabel(
            control_card, text="", font=mono_font(10), text_color=TEAL, anchor="w",
        )
        self.schedule_info_label.pack(fill="x", padx=(14 + 78, 14), pady=(0, 4))

        row2 = ctk.CTkFrame(control_card, fg_color="transparent")
        row2.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkLabel(row2, text="MICRO", font=mono_font(10, "bold"), text_color=TEXT_DIM, width=70, anchor="w").pack(side="left")
        self.device_menu = ctk.CTkOptionMenu(
            row2, values=["Chargement..."], font=mono_font(11), fg_color=BG_CARD,
            button_color=BORDER, button_hover_color=ORANGE_DIM, corner_radius=0,
            text_color=TEXT, dropdown_fg_color=BG_CARD,
        )
        self.device_menu.pack(side="left", fill="x", expand=True, padx=(8, 12))

        self.start_button = ctk.CTkButton(
            row2, text="DEMARRER", font=mono_font(11, "bold"), width=110, corner_radius=0,
            fg_color=ORANGE, hover_color="#e0562a", text_color="#0a0d0e",
            command=self._on_start, state="disabled",
        )
        self.start_button.pack(side="left", padx=(0, 6))

        self.stop_button = ctk.CTkButton(
            row2, text="ARRETER", font=mono_font(11, "bold"), width=110, corner_radius=0,
            fg_color=BG_CARD, hover_color=BORDER, text_color=RED, border_width=1,
            border_color=RED, command=self._on_stop, state="disabled",
        )
        self.stop_button.pack(side="left")

        # bandeau statut d'enregistrement (rec dot + chrono)
        rec_row = ctk.CTkFrame(control_card, fg_color="transparent")
        rec_row.pack(fill="x", padx=14, pady=(0, 12))
        self.rec_dot = ctk.CTkLabel(rec_row, text="●", font=mono_font(12), text_color=TEXT_DIM)
        self.rec_dot.pack(side="left")
        self.rec_status_label = ctk.CTkLabel(
            rec_row, text="EN ATTENTE", font=mono_font(11, "bold"), text_color=TEXT_DIM,
        )
        self.rec_status_label.pack(side="left", padx=(6, 0))
        self.timer_label = ctk.CTkLabel(
            rec_row, text="00:00:00", font=mono_font(11, "bold"), text_color=TEAL,
        )
        self.timer_label.pack(side="right")

        # panneau transcription
        transcript_card = ctk.CTkFrame(self.session_view, fg_color=BG_PANEL, corner_radius=0, border_width=1, border_color=BORDER)
        transcript_card.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        self.transcript_header_label = ctk.CTkLabel(
            transcript_card, text="// JOURNAL EN DIRECT", font=mono_font(11, "bold"),
            text_color=ORANGE, anchor="w",
        )
        self.transcript_header_label.pack(fill="x", padx=14, pady=(12, 8))

        self.transcript_box = ctk.CTkTextbox(
            transcript_card, wrap="word", font=mono_font(12), fg_color=BG_APP,
            text_color=TEXT, corner_radius=0, border_width=1, border_color=BORDER,
        )
        self.transcript_box.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.transcript_box.configure(state="disabled")

        # --- vue "emploi du temps" (Lundi -> Vendredi), cachée par défaut ---
        self.schedule_view = ctk.CTkFrame(main, fg_color=BG_APP)
        self._build_schedule_view(self.schedule_view)

        # barre du bas
        bottom_bar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=32)
        bottom_bar.pack(fill="x", side="bottom")
        bottom_bar.pack_propagate(False)
        self.footer_label = ctk.CTkLabel(
            bottom_bar, text="ECHO v0.1 — BACKEND: FASTER-WHISPER (CPU)", font=mono_font(9),
            text_color=TEXT_DIM,
        )
        self.footer_label.pack(side="left", padx=12)

    def _build_schedule_view(self, parent):
        header = SectionHeader(parent, "Emploi du temps — Lundi à Vendredi", TEAL)
        header.pack(fill="x", padx=16, pady=(16, 10))

        columns_row = ctk.CTkFrame(parent, fg_color="transparent")
        columns_row.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        for i in range(5):
            columns_row.grid_columnconfigure(i, weight=1, uniform="day")
        columns_row.grid_rowconfigure(0, weight=1)

        for i, jour in enumerate(JOURS[:5]):
            col = ctk.CTkFrame(
                columns_row, fg_color=BG_PANEL, corner_radius=8, border_width=1,
                border_color=BORDER,
            )
            col.grid(row=0, column=i, sticky="nsew", padx=6)

            ctk.CTkLabel(
                col, text=jour.upper(), font=mono_font(11, "bold"), text_color=ORANGE,
            ).pack(pady=(12, 8))

            list_frame = ctk.CTkScrollableFrame(col, fg_color="transparent")
            list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.schedule_day_columns[jour] = list_frame

            add_btn = ctk.CTkButton(
                col, text="+ AJOUTER", font=mono_font(10, "bold"), corner_radius=8,
                fg_color=ORANGE, hover_color="#e0562a", text_color=BG_APP,
                command=lambda j=jour: self._open_creneau_form(j, None, None),
            )
            add_btn.pack(fill="x", padx=8, pady=(0, 12))

    # ------------------------------------------------------------------
    # Chargement micro / modèle
    # ------------------------------------------------------------------

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
            self.after(0, self._on_backend_ready)

        threading.Thread(target=_load, daemon=True).start()

    def _on_backend_ready(self):
        self.engine_chip.set("MOTEUR: PRET", TEAL)
        self.start_button.configure(state="normal")

    # ------------------------------------------------------------------
    # Emploi du temps (vue intégrée, remplace la zone principale)
    # ------------------------------------------------------------------

    def _toggle_schedule_view(self):
        if self.schedule_view_visible:
            self.schedule_view.pack_forget()
            self.session_view.pack(fill="both", expand=True)
            self.schedule_button.configure(text="EMPLOI DU TEMPS")
            self.schedule_view_visible = False
        else:
            self.session_view.pack_forget()
            self.schedule_data = load_schedule()
            self._refresh_schedule_columns()
            self.schedule_view.pack(fill="both", expand=True)
            self.schedule_button.configure(text="◂ RETOUR AU DIRECT")
            self.schedule_view_visible = True

    def _refresh_schedule_columns(self):
        for jour, frame in self.schedule_day_columns.items():
            for widget in frame.winfo_children():
                widget.destroy()

            creneaux = self.schedule_data.get(jour) or []
            if not creneaux:
                ctk.CTkLabel(
                    frame, text="Aucun créneau", font=mono_font(9), text_color=TEXT_DIM,
                ).pack(pady=10)
                continue

            for idx, creneau in enumerate(creneaux):
                card = ScheduleCard(
                    frame, creneau,
                    on_click=lambda j=jour, c=creneau, i=idx: self._open_creneau_form(j, c, i),
                )
                card.pack(fill="x", pady=(0, 6))

    def _open_creneau_form(self, jour: str, creneau: dict | None, idx: int | None):
        form = ctk.CTkToplevel(self)
        form.title(f"Créneau — {jour.capitalize()}")
        form.geometry("340x300")
        form.configure(fg_color=BG_APP)
        form.transient(self)

        fields: dict[str, ctk.CTkEntry] = {}

        def add_field(label: str, key: str, default: str = ""):
            field_row = ctk.CTkFrame(form, fg_color="transparent")
            field_row.pack(fill="x", padx=16, pady=6)
            ctk.CTkLabel(
                field_row, text=label, font=mono_font(10, "bold"), text_color=TEXT_DIM,
                width=60, anchor="w",
            ).pack(side="left")
            entry = ctk.CTkEntry(
                field_row, font=mono_font(11), fg_color=BG_CARD, border_color=BORDER,
                corner_radius=0, text_color=TEXT,
            )
            entry.insert(0, default)
            entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
            fields[key] = entry

        add_field("DEBUT", "debut", creneau.get("debut", "") if creneau else "")
        add_field("FIN", "fin", creneau.get("fin", "") if creneau else "")
        add_field("TITRE", "titre", creneau.get("titre", "") if creneau else "")
        add_field("INFOS", "info", creneau.get("info", "") if creneau else "")
        ctk.CTkLabel(
            form, text="(ex. infos : salle, professeur...)", font=mono_font(9),
            text_color=TEXT_DIM,
        ).pack(anchor="w", padx=16 + 68)

        error_label = ctk.CTkLabel(form, text="", font=mono_font(10), text_color=RED)
        error_label.pack(pady=(8, 0))

        def on_save():
            debut = fields["debut"].get().strip()
            fin = fields["fin"].get().strip()
            titre = fields["titre"].get().strip()
            info = fields["info"].get().strip()

            time_pattern = r"^\d{1,2}:\d{2}$"
            if not re.match(time_pattern, debut) or not re.match(time_pattern, fin):
                error_label.configure(text="Format horaire invalide (attendu HH:MM)")
                return

            new_entry = {"debut": debut, "fin": fin, "titre": titre or "Cours sans titre"}
            if info:
                new_entry["info"] = info

            creneaux = self.schedule_data.setdefault(jour, [])
            if idx is None:
                creneaux.append(new_entry)
            else:
                creneaux[idx] = new_entry
            creneaux.sort(key=lambda c: c.get("debut", ""))

            form.destroy()
            self._save_schedule_and_refresh()

        save_btn = ctk.CTkButton(
            form, text="ENREGISTRER", font=mono_font(11, "bold"), corner_radius=0,
            fg_color=ORANGE, hover_color="#e0562a", text_color=BG_APP, command=on_save,
        )
        save_btn.pack(pady=(14, 6))

        if idx is not None:
            def on_delete():
                creneaux = self.schedule_data.setdefault(jour, [])
                if 0 <= idx < len(creneaux):
                    del creneaux[idx]
                form.destroy()
                self._save_schedule_and_refresh()

            delete_btn = ctk.CTkButton(
                form, text="SUPPRIMER CE CRENEAU", font=mono_font(10, "bold"), corner_radius=0,
                fg_color=BG_CARD, hover_color=RED, text_color=RED, border_width=1,
                border_color=RED, command=on_delete,
            )
            delete_btn.pack(pady=(0, 10))

    def _save_schedule_and_refresh(self):
        save_schedule(self.schedule_data)
        self._refresh_schedule_columns()
        self._apply_current_course()

    def _apply_current_course(self):
        """Pré-remplit le titre (et les infos) avec le cours en cours selon
        l'emploi du temps, sans écraser un titre que l'utilisateur a tapé
        lui-même."""
        if self._recording or self._viewing_past_session:
            return

        current_text = self.title_entry.get().strip()
        entry = get_current_entry()

        if entry:
            titre = entry.get("titre", "Cours")
            if current_text == "" or current_text == self._auto_filled_title:
                self.title_entry.delete(0, "end")
                self.title_entry.insert(0, titre)
                self._auto_filled_title = titre
            info = entry.get("info", "")
            self.schedule_info_label.configure(text=f"// {info}" if info else "")
        else:
            if current_text == self._auto_filled_title:
                self.title_entry.delete(0, "end")
                self._auto_filled_title = ""
            self.schedule_info_label.configure(text="")

    def _schedule_check_loop(self):
        self._apply_current_course()
        self.after(60_000, self._schedule_check_loop)

    # ------------------------------------------------------------------
    # Liste des sessions passées
    # ------------------------------------------------------------------

    def _refresh_session_list(self):
        for widget in self.session_list_frame.winfo_children():
            widget.destroy()

        sessions = list_sessions()
        if not sessions:
            ctk.CTkLabel(
                self.session_list_frame, text="AUCUNE SESSION\nENREGISTREE",
                font=mono_font(10), text_color=TEXT_DIM, justify="left",
            ).pack(anchor="w", pady=8)
            return

        for summary in sessions:
            card = SessionCard(
                self.session_list_frame, summary.title, summary.date_str,
                on_click=lambda s=summary: self._view_past_session(s),
            )
            card.pack(fill="x", pady=(0, 6))

    def _view_past_session(self, summary):
        if self._recording:
            return  # on ne quitte pas une session en cours d'enregistrement
        self._viewing_past_session = True
        self.transcript_header_label.configure(text=f"// {summary.title.upper()}")
        content = read_session_content(summary.path)
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", content)
        self.transcript_box.configure(state="disabled")
        self.back_to_live_button.pack(fill="x", padx=10, pady=(0, 10), side="bottom")

    def _return_to_live(self):
        self._viewing_past_session = False
        self.back_to_live_button.pack_forget()
        self.transcript_header_label.configure(text="// JOURNAL EN DIRECT")
        self._clear_transcript()

    # ------------------------------------------------------------------
    # Logique session / enregistrement
    # ------------------------------------------------------------------

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

        self._recording = True
        self._elapsed_seconds = 0
        self._viewing_past_session = False
        self.back_to_live_button.pack_forget()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.title_entry.configure(state="disabled")
        self.device_menu.configure(state="disabled")
        self.rec_status_label.configure(text="ENREGISTREMENT", text_color=RED)
        self.transcript_header_label.configure(text="// JOURNAL EN DIRECT")

        self._clear_transcript()
        self._tick_timer()
        self._blink_rec_dot()

    def _on_stop(self):
        if self.recorder:
            self.recorder.stop()
        if self.worker:
            self.worker.stop()
        if self.session:
            self.session.close()

        self._recording = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.title_entry.configure(state="normal")
        self.device_menu.configure(state="normal")
        self.rec_status_label.configure(text="EN ATTENTE", text_color=TEXT_DIM)
        self.rec_dot.configure(text_color=TEXT_DIM)
        self._refresh_session_list()

    def _on_text_ready(self, text: str, started_at: float):
        self.after(0, lambda: self._append_transcript(text, started_at))
        if self.session:
            self.session.append_segment(text, started_at)

    # ------------------------------------------------------------------
    # Helpers UI thread-safe
    # ------------------------------------------------------------------

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

    def _tick_timer(self):
        if not self._recording:
            return
        hours, rem = divmod(self._elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        self.timer_label.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self._elapsed_seconds += 1
        self.after(1000, self._tick_timer)

    def _blink_rec_dot(self):
        if not self._recording:
            return
        self._blink_on = not self._blink_on
        self.rec_dot.configure(text_color=RED if self._blink_on else BG_PANEL)
        self.after(600, self._blink_rec_dot)

    def on_closing(self):
        if self._recording:
            self._on_stop()
        self.destroy()


if __name__ == "__main__":
    app = EchoApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()