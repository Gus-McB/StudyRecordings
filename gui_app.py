import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import threading
import queue
from datetime import datetime

# Import your existing modules
from src.processRecordings import process_directory
from src.transcriptAnalytics import analyse_multiple_transcripts

class TranscriptAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("StudyRecordings - Audio Transcript Analysis")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Configure modern style
        self.setup_styles()
        
        # Queue for thread communication
        self.queue = queue.Queue()
        
        # Variables
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "TranscriptAnalysis"))
        self.processing = False
        
        self.create_widgets()
        self.setup_drag_drop()
        
        # Start checking queue for updates
        self.root.after(100, self.check_queue)
    
    def setup_styles(self):
        """Configure modern gray color scheme"""
        style = ttk.Style()
        
        # Configure colors
        self.colors = {
            'bg_dark': '#2b2b2b',
            'bg_medium': '#3c3c3c',
            'bg_light': '#4a4a4a',
            'accent': '#007acc',
            'accent_hover': '#005a9e',
            'text_light': '#ffffff',
            'text_medium': '#cccccc',
            'text_dark': '#999999',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336'
        }
        
        # Configure ttk styles
        style.theme_use('clam')
        style.configure('Modern.TFrame', background=self.colors['bg_medium'])
        style.configure('Card.TFrame', background=self.colors['bg_light'], relief='raised', borderwidth=1)
        style.configure('Modern.TLabel', background=self.colors['bg_medium'], foreground=self.colors['text_light'])
        style.configure('Title.TLabel', background=self.colors['bg_medium'], foreground=self.colors['text_light'], font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', background=self.colors['bg_medium'], foreground=self.colors['text_medium'], font=('Arial', 10))
        style.configure('Modern.TEntry', fieldbackground=self.colors['bg_light'], foreground=self.colors['text_light'], borderwidth=1)
        style.configure('Modern.TButton', background=self.colors['bg_light'], foreground=self.colors['text_light'])
        style.configure('Accent.TButton', background=self.colors['accent'], foreground=self.colors['text_light'], font=('Arial', 11, 'bold'))
        style.configure('Modern.TCheckbutton', background=self.colors['bg_medium'], foreground=self.colors['text_light'])
        style.configure('Modern.TLabelframe', background=self.colors['bg_medium'], foreground=self.colors['text_light'])
        style.configure('Modern.TLabelframe.Label', background=self.colors['bg_medium'], foreground=self.colors['text_light'])
    
    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root, style='Modern.TFrame', padding="20")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        # Header section
        header_frame = ttk.Frame(main_container, style='Modern.TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 30))
        header_frame.columnconfigure(0, weight=1)
        
        # Title and subtitle
        title_label = ttk.Label(header_frame, text="🎙️ StudyRecordings Analysis", style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 5))
        
        subtitle_label = ttk.Label(header_frame, text="Advanced Audio Transcript Analysis with Machine Learning", style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, pady=(0, 10))
        
        # Main content area - split into left and right
        content_frame = ttk.Frame(main_container, style='Modern.TFrame')
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # LEFT SIDE - Drag & Drop
        self.create_drag_drop_section(content_frame)
        
        # RIGHT SIDE - Folder Selection
        self.create_folder_selection_section(content_frame)
        
        # Bottom section - Options and Processing
        self.create_bottom_section(main_container)
        
        # Configure main container row weights
        main_container.rowconfigure(1, weight=1)
    
    def create_drag_drop_section(self, parent):
        """Create the left side drag & drop section"""
        # Left panel
        left_panel = ttk.Frame(parent, style='Card.TFrame', padding="20")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        
        # Section title
        drag_title = ttk.Label(left_panel, text="📁 Drag & Drop", style='Modern.TLabel', font=('Arial', 12, 'bold'))
        drag_title.grid(row=0, column=0, pady=(0, 20))
        
        # Drag and drop area
        self.drop_frame = tk.Frame(left_panel, bg='#404040', relief='dashed', bd=2)
        self.drop_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        self.drop_frame.columnconfigure(0, weight=1)
        self.drop_frame.rowconfigure(0, weight=1)
        
        # Drop content
        drop_content = tk.Frame(self.drop_frame, bg='#404040')
        drop_content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        drop_content.columnconfigure(0, weight=1)
        drop_content.rowconfigure(0, weight=1)
        
        # Drop icon and text
        self.drop_icon = tk.Label(drop_content, text="📂", bg='#404040', fg='#007acc', font=('Arial', 48))
        self.drop_icon.grid(row=0, column=0, pady=(40, 10))
        
        self.drop_label = tk.Label(drop_content, 
                                 text="Drop Audio Folder Here\n\nSupports: MP3, WAV, M4A, WMA, AMR",
                                 bg='#404040', fg='#cccccc', font=('Arial', 11), justify=tk.CENTER)
        self.drop_label.grid(row=1, column=0, pady=(0, 40))
        
        # Status indicator
        self.drop_status = tk.Label(left_panel, text="No folder selected", 
                                  bg=self.colors['bg_light'], fg=self.colors['text_dark'], 
                                  font=('Arial', 9), pady=5)
        self.drop_status.grid(row=2, column=0, sticky=(tk.W, tk.E))
    
    def create_folder_selection_section(self, parent):
        """Create the right side folder selection section"""
        # Right panel
        right_panel = ttk.Frame(parent, style='Card.TFrame', padding="20")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        right_panel.columnconfigure(0, weight=1)
        
        # Section title
        folder_title = ttk.Label(right_panel, text="🗂️ Folder Selection", style='Modern.TLabel', font=('Arial', 12, 'bold'))
        folder_title.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # Input folder section
        input_section = ttk.LabelFrame(right_panel, text="Audio Files Folder", style='Modern.TLabelframe', padding="15")
        input_section.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_section.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_section, textvariable=self.input_folder, style='Modern.TEntry', font=('Arial', 10))
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        input_button = ttk.Button(input_section, text="📁 Browse for Audio Folder", 
                                command=self.browse_input_folder, style='Modern.TButton')
        input_button.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Output folder section
        output_section = ttk.LabelFrame(right_panel, text="Output Folder", style='Modern.TLabelframe', padding="15")
        output_section.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        output_section.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_section, textvariable=self.output_folder, style='Modern.TEntry', font=('Arial', 10))
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        output_button = ttk.Button(output_section, text="📁 Browse for Output Folder", 
                                 command=self.browse_output_folder, style='Modern.TButton')
        output_button.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Processing options
        options_section = ttk.LabelFrame(right_panel, text="Processing Options", style='Modern.TLabelframe', padding="15")
        options_section.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.transcribe_var = tk.BooleanVar(value=True)
        self.analyze_var = tk.BooleanVar(value=True)
        
        transcribe_cb = ttk.Checkbutton(options_section, text="🎤 Transcribe Audio Files", 
                                      variable=self.transcribe_var, style='Modern.TCheckbutton')
        transcribe_cb.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        analyze_cb = ttk.Checkbutton(options_section, text="📊 Generate Analysis Reports", 
                                   variable=self.analyze_var, style='Modern.TCheckbutton')
        analyze_cb.grid(row=1, column=0, sticky=tk.W)
        
        # Process button
        self.process_button = ttk.Button(right_panel, text="🚀 Start Processing", 
                                       command=self.start_processing, style='Accent.TButton')
        self.process_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_bottom_section(self, parent):
        """Create the bottom section with progress and logs"""
        bottom_frame = ttk.Frame(parent, style='Modern.TFrame')
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(1, weight=1)
        
        # Progress section
        progress_frame = ttk.LabelFrame(bottom_frame, text="Progress Status", style='Modern.TLabelframe', padding="15")
        progress_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready to process...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var, style='Modern.TLabel')
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Log section
        log_frame = ttk.LabelFrame(bottom_frame, text="Processing Log", style='Modern.TLabelframe', padding="15")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Text widget with scrollbar
        text_container = tk.Frame(log_frame, bg=self.colors['bg_medium'])
        text_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_container.columnconfigure(0, weight=1)
        text_container.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(text_container, height=8, wrap=tk.WORD, 
                               bg=self.colors['bg_dark'], fg=self.colors['text_light'],
                               insertbackground=self.colors['text_light'], font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        # Register all drop areas
        for widget in [self.drop_frame, self.drop_icon, self.drop_label]:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop(self, event):
        """Handle drag and drop"""
        files = self.root.tk.splitlist(event.data)
        if files:
            folder_path = files[0]
            if os.path.isdir(folder_path):
                self.input_folder.set(folder_path)
                self.log_message(f"Folder selected via drag & drop: {folder_path}")
                self.update_drop_status("✅ Folder Selected!", self.colors['success'])
                self.drop_icon.config(text="✅")
                self.drop_label.config(text="Folder Ready!\n\nClick 'Start Processing' to begin")
            else:
                # If it's a file, use its parent directory
                folder_path = os.path.dirname(folder_path)
                self.input_folder.set(folder_path)
                self.log_message(f"Parent folder selected: {folder_path}")
                self.update_drop_status("✅ Parent Folder Selected!", self.colors['success'])
                self.drop_icon.config(text="✅")
                self.drop_label.config(text="Parent Folder Ready!\n\nClick 'Start Processing' to begin")
    
    def update_drop_status(self, message, color):
        """Update the drop status indicator"""
        self.drop_status.config(text=message, fg=color)
    
    def browse_input_folder(self):
        """Browse for input folder"""
        folder = filedialog.askdirectory(title="Select Audio Files Folder")
        if folder:
            self.input_folder.set(folder)
            self.log_message(f"Input folder selected: {folder}")
            self.update_drop_status("✅ Folder Selected via Browse!", self.colors['success'])
            self.drop_icon.config(text="✅")
            self.drop_label.config(text="Folder Ready!\n\nClick 'Start Processing' to begin")
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
            self.log_message(f"Output folder selected: {folder}")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_processing(self):
        """Start the processing in a separate thread"""
        if self.processing:
            return
        
        # Validate inputs
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select an input folder")
            return
        
        if not os.path.exists(self.input_folder.get()):
            messagebox.showerror("Error", "Input folder does not exist")
            return
        
        if not (self.transcribe_var.get() or self.analyze_var.get()):
            messagebox.showerror("Error", "Please select at least one processing option")
            return
        
        # Start processing
        self.processing = True
        self.process_button.config(text="⏳ Processing...", state="disabled")
        self.progress_bar.start()
        self.update_drop_status("🔄 Processing...", self.colors['warning'])
        
        # Start processing thread
        thread = threading.Thread(target=self.process_files, daemon=True)
        thread.start()
    
    def process_files(self):
        """Process files in separate thread"""
        try:
            input_dir = self.input_folder.get()
            output_dir = self.output_folder.get()
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            self.queue.put(("progress", "Starting processing..."))
            
            # Step 1: Transcription
            if self.transcribe_var.get():
                self.queue.put(("progress", "Step 1/2: Transcribing audio files..."))
                self.queue.put(("log", "Starting audio transcription..."))
                
                try:
                    # Process audio files
                    process_directory(input_dir, output_dir)
                    self.queue.put(("log", "Audio transcription completed successfully!"))
                except Exception as e:
                    self.queue.put(("log", f"Error during transcription: {str(e)}"))
                    raise e
            
            # Step 2: Analysis
            if self.analyze_var.get():
                self.queue.put(("progress", "Step 2/2: Generating analysis reports..."))
                self.queue.put(("log", "Starting transcript analysis..."))
                
                try:
                    # Determine transcript directory
                    if self.transcribe_var.get():
                        transcript_dir = os.path.join(output_dir, "transcripts")
                    else:
                        # Look for existing transcripts in input directory
                        transcript_dir = input_dir
                    
                    # Create analysis directory
                    analysis_dir = os.path.join(output_dir, "analysis_results")
                    
                    # Run analysis
                    analyse_multiple_transcripts(transcript_dir, analysis_dir)
                    self.queue.put(("log", "Analysis completed successfully!"))
                except Exception as e:
                    self.queue.put(("log", f"Error during analysis: {str(e)}"))
                    raise e
            
            # Success
            self.queue.put(("progress", "Processing completed successfully!"))
            self.queue.put(("log", f"All results saved to: {output_dir}"))
            self.queue.put(("complete", "success"))
            
        except Exception as e:
            self.queue.put(("log", f"Processing failed: {str(e)}"))
            self.queue.put(("complete", "error"))
    
    def check_queue(self):
        """Check for messages from processing thread"""
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                
                if msg_type == "log":
                    self.log_message(msg_data)
                elif msg_type == "progress":
                    self.progress_var.set(msg_data)
                elif msg_type == "complete":
                    self.processing = False
                    self.progress_bar.stop()
                    self.process_button.config(text="🚀 Start Processing", state="normal")
                    
                    if msg_data == "success":
                        self.update_drop_status("✅ Processing Complete!", self.colors['success'])
                        messagebox.showinfo("Success", 
                                          f"Processing completed successfully!\n\nResults saved to:\n{self.output_folder.get()}")
                        # Open output folder
                        try:
                            os.startfile(self.output_folder.get())  # Windows
                        except:
                            try:
                                os.system(f'open "{self.output_folder.get()}"')  # macOS
                            except:
                                os.system(f'xdg-open "{self.output_folder.get()}"')  # Linux
                    else:
                        self.update_drop_status("❌ Processing Failed", self.colors['error'])
                        messagebox.showerror("Error", "Processing failed. Check the log for details.")
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)

def main():
    # Create the main window
    root = TkinterDnD.Tk()
    app = TranscriptAnalysisGUI(root)
    
    # Set window icon (optional)
    try:
        root.iconbitmap("icon.ico")  # Add an icon file if you have one
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
