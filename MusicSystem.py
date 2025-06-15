import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import lfilter, butter, filtfilt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from transformers import pipeline
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# ======================
# AUDIO PROCESSING
# ======================
def play_audio(audio, sample_rate):
    """Play audio using sounddevice"""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        messagebox.showerror("Playback Error", str(e))

# ======================
# ENHANCED FUZZY LOGIC SYSTEM
# ======================
class FuzzyFXController:
    def __init__(self):
        # Input variables
        self.energy = ctrl.Antecedent(np.arange(0, 11, 1), 'energy')
        self.complexity = ctrl.Antecedent(np.arange(0, 11, 1), 'complexity')
        self.emotion = ctrl.Antecedent(np.arange(-10, 11, 1), 'emotion')  # -10=negative, 0=neutral, 10=positive
        
        # Output variables
        self.pitch = ctrl.Consequent(np.arange(-5, 6, 1), 'pitch')
        self.reverb = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'reverb')
        self.distortion = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'distortion')
        self.tempo = ctrl.Consequent(np.arange(0.5, 2.1, 0.1), 'tempo')
        
        # Membership functions
        for var in [self.energy, self.complexity]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 5])
            var['medium'] = fuzz.trimf(var.universe, [0, 5, 10])
            var['high'] = fuzz.trimf(var.universe, [5, 10, 10])
        
        self.emotion['negative'] = fuzz.trimf(self.emotion.universe, [-10, -10, 0])
        self.emotion['neutral'] = fuzz.trimf(self.emotion.universe, [-5, 0, 5])
        self.emotion['positive'] = fuzz.trimf(self.emotion.universe, [0, 10, 10])
        
        self.pitch['down'] = fuzz.trimf(self.pitch.universe, [-5, -5, 0])
        self.pitch['neutral'] = fuzz.trimf(self.pitch.universe, [-2, 0, 2])
        self.pitch['up'] = fuzz.trimf(self.pitch.universe, [0, 5, 5])
        
        self.reverb['dry'] = fuzz.trimf(self.reverb.universe, [0, 0, 0.3])
        self.reverb['medium'] = fuzz.trimf(self.reverb.universe, [0.2, 0.5, 0.8])
        self.reverb['wet'] = fuzz.trimf(self.reverb.universe, [0.6, 1, 1])
        
        self.distortion['clean'] = fuzz.trimf(self.distortion.universe, [0, 0, 0.3])
        self.distortion['warm'] = fuzz.trimf(self.distortion.universe, [0.2, 0.5, 0.8])
        self.distortion['heavy'] = fuzz.trimf(self.distortion.universe, [0.6, 1, 1])
        
        self.tempo['slow'] = fuzz.trimf(self.tempo.universe, [0.5, 0.5, 1])
        self.tempo['normal'] = fuzz.trimf(self.tempo.universe, [0.8, 1, 1.2])
        self.tempo['fast'] = fuzz.trimf(self.tempo.universe, [1, 1.5, 2])
        
        # Rules
        self.system = ctrl.ControlSystem([
            # Energy and complexity rules
            ctrl.Rule(self.energy['low'] & self.complexity['low'], 
                     [self.pitch['down'], self.reverb['wet'], self.distortion['clean']]),
            ctrl.Rule(self.energy['medium'] & self.complexity['medium'],
                     [self.pitch['neutral'], self.reverb['medium'], self.distortion['warm']]),
            ctrl.Rule(self.energy['high'] & self.complexity['high'],
                     [self.pitch['up'], self.reverb['dry'], self.distortion['heavy']]),
            
            # Emotion rules
            ctrl.Rule(self.emotion['negative'], 
                     [self.pitch['down'], self.tempo['slow']]),
            ctrl.Rule(self.emotion['neutral'], 
                     [self.pitch['neutral'], self.tempo['normal']]),
            ctrl.Rule(self.emotion['positive'], 
                     [self.pitch['up'], self.tempo['fast']])
        ])
    
    def get_effects(self, energy, complexity, emotion=0):
        sim = ctrl.ControlSystemSimulation(self.system)
        sim.input['energy'] = energy
        sim.input['complexity'] = complexity
        sim.input['emotion'] = emotion
        sim.compute()
        return {
            'pitch_shift': sim.output['pitch'],
            'reverb': sim.output['reverb'],
            'distortion': sim.output['distortion'],
            'tempo': sim.output['tempo']
        }

# ======================
# ENHANCED TRANSFORMER STYLE ANALYZER
# ======================
class StyleTransformer:
    def __init__(self):
        try:
            self.pipe = pipeline("text-generation", model="gpt2")
        except Exception as e:
            messagebox.showwarning("Model Warning", f"GPT-2 model not available. Using fallback.\nError: {str(e)}")
            self.pipe = None
    
    def analyze_style(self, description):
        if self.pipe is None:
            return 5, 5, 0  # Default values if model not available
            
        prompt = f """
        Convert this music style into energy (0-10), complexity (0-10), and emotion (-10 to +10):
        "{description}"
        
        Examples:
        - "lofi hiphop" → 3,4,2
        - "metalcore" → 9,7,-5
        - "happy pop" → 7,5,8
        - "sad jazz" → 4,6,-7
        
        Respond ONLY with three numbers separated by commas: 
        """
        try:
            response = self.pipe(prompt, max_length=40,truncation=True)[0]['generated_text']
            parts = [float(x) for x in response.strip().split(',')[:3]]
            if len(parts) != 3:
                raise ValueError("Model did not return three values")
            return (
                np.clip(float(parts[0]), 0, 10),
                np.clip(float(parts[1]), 0, 10),
                np.clip(float(parts[2]), -10, 10)
            )
        except:
            return 5, 5, 0  # Default values

# ======================
# ENHANCED AUDIO ENGINE
# ======================
class AudioFXEngine:
    def __init__(self, filepath=None):
        self.original = None
        self.current = None
        self.sr = None
        self.history = []
        self.spectrogram_fig = None
        self.waveform_fig = None
        if filepath:
            self.load_audio(filepath)
    
    def load_audio(self, filepath):
        try:
            self.original, self.sr = librosa.load(filepath, sr=None)
            self.current = self.original.copy()
            self.history = []
            self.create_visualizations()
            return True
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load audio: {str(e)}")
            return False
    
    def create_visualizations(self):
        plt.close('all')  # Close previous figures
        
        # Create waveform plot
        self.waveform_fig, ax = plt.subplots(figsize=(6, 2))
        librosa.display.waveshow(self.current, sr=self.sr, ax=ax)
        ax.set_title('Waveform')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        self.waveform_fig.tight_layout()
        
        # Create spectrogram plot
        self.spectrogram_fig, ax = plt.subplots(figsize=(6, 2))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.current)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log', ax=ax)
        self.spectrogram_fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title('Spectrogram')
        self.spectrogram_fig.tight_layout()
    
    def pitch_shift(self, steps):
        try:
            self.current = librosa.effects.pitch_shift(
                self.current, sr=self.sr, n_steps=steps)
            self.history.append(f"Pitch shift: {steps} semitones")
            self.create_visualizations()
        except Exception as e:
            messagebox.showerror("Effect Error", str(e))
    
    def reverb(self, wetness):
        try:
        # Shorter impulse (e.g., 0.1 seconds)
            impulse_length = int(self.sr * 0.1)
            impulse = np.exp(-np.linspace(0, 3, impulse_length))
            impulse /= np.sum(impulse)  # Normalize
            wet_signal = lfilter(impulse, [1], self.current) * wetness
            self.current = self.current * (1 - wetness) + wet_signal
            self.history.append(f"Reverb: {wetness*100:.0f}% wet")
            self.create_visualizations()
        except Exception as e:
            messagebox.showerror("Effect Error", str(e))
    
    def distortion(self, amount):
        try:
            if amount > 0:
                self.current = np.tanh(self.current * (1 + amount * 10))
                self.history.append(f"Distortion: {amount*100:.0f}%")
                self.create_visualizations()
        except Exception as e:
            messagebox.showerror("Effect Error", str(e))
    
    def change_tempo(self, factor):
        try:
            self.current = librosa.effects.time_stretch(self.current, rate=factor)
            self.history.append(f"Tempo change: {factor:.2f}x")
            self.create_visualizations()
        except Exception as e:
            messagebox.showerror("Effect Error", str(e))
    
    def apply_filter(self, lowcut=None, highcut=None):
        try:
            if lowcut or highcut:
                nyquist = 0.5 * self.sr
                low = lowcut / nyquist if lowcut else 0
                high = highcut / nyquist if highcut else 1
                b, a = butter(4, [low, high], btype='band')
                self.current = filtfilt(b, a, self.current)
                self.history.append(f"Filter: {lowcut or 'None'}Hz-{highcut or 'None'}Hz")
                self.create_visualizations()
        except Exception as e:
            messagebox.showerror("Effect Error", str(e))
    
    def reset(self):
        if self.original is not None:
            self.current = self.original.copy()
            self.history = []
            self.create_visualizations()
            return True
        return False
    
    def save(self, filename):
        try:
            sf.write(filename, self.current, self.sr)
            return True
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            return False
    
    def play(self):
        if self.current is not None:
            threading.Thread(target=play_audio, args=(self.current, self.sr)).start()

# ======================
# TKINTER GUI
# ======================
class MusicFXApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Music Improvisation Tool")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
        # Initialize systems
        self.fuzzy = FuzzyFXController()
        self.transformer = StyleTransformer()
        self.engine = AudioFXEngine()
        
        # Create GUI
        self.create_widgets()
        self.setup_layout()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def create_widgets(self):
        # File controls
        self.file_frame = ttk.LabelFrame(self.root, text="Audio File")
        self.load_btn = ttk.Button(self.file_frame, text="Load Audio", command=self.load_audio)
        self.file_label = ttk.Label(self.file_frame, text="No file loaded")
        self.duration_label = ttk.Label(self.file_frame, text="Duration: 0.0s")
        
        # Manual effects
        self.manual_frame = ttk.LabelFrame(self.root, text="Manual Effects")
        
        # Pitch
        self.pitch_frame = ttk.Frame(self.manual_frame)
        self.pitch_label = ttk.Label(self.pitch_frame, text="Pitch Shift (semitones):")
        self.pitch_slider = ttk.Scale(self.pitch_frame, from_=-12, to=12, orient=tk.HORIZONTAL)
        self.pitch_entry = ttk.Entry(self.pitch_frame, width=5)
        self.pitch_btn = ttk.Button(self.pitch_frame, text="Apply", command=self.apply_pitch)
        
        # Reverb
        self.reverb_frame = ttk.Frame(self.manual_frame)
        self.reverb_label = ttk.Label(self.reverb_frame, text="Reverb (wetness):")
        self.reverb_slider = ttk.Scale(self.reverb_frame, from_=0, to=1, orient=tk.HORIZONTAL)
        self.reverb_entry = ttk.Entry(self.reverb_frame, width=5)
        self.reverb_btn = ttk.Button(self.reverb_frame, text="Apply", command=self.apply_reverb)
        
        # Distortion
        self.dist_frame = ttk.Frame(self.manual_frame)
        self.dist_label = ttk.Label(self.dist_frame, text="Distortion:")
        self.dist_slider = ttk.Scale(self.dist_frame, from_=0, to=1, orient=tk.HORIZONTAL)
        self.dist_entry = ttk.Entry(self.dist_frame, width=5)
        self.dist_btn = ttk.Button(self.dist_frame, text="Apply", command=self.apply_distortion)
        
        # Tempo
        self.tempo_frame = ttk.Frame(self.manual_frame)
        self.tempo_label = ttk.Label(self.tempo_frame, text="Tempo (factor):")
        self.tempo_slider = ttk.Scale(self.tempo_frame, from_=0.5, to=2, orient=tk.HORIZONTAL)
        self.tempo_entry = ttk.Entry(self.tempo_frame, width=5)
        self.tempo_btn = ttk.Button(self.tempo_frame, text="Apply", command=self.apply_tempo)
        
        # Filter
        self.filter_frame = ttk.Frame(self.manual_frame)
        self.filter_label = ttk.Label(self.filter_frame, text="Filter (Hz):")
        self.lowcut_label = ttk.Label(self.filter_frame, text="Low:")
        self.lowcut_entry = ttk.Entry(self.filter_frame, width=6)
        self.highcut_label = ttk.Label(self.filter_frame, text="High:")
        self.highcut_entry = ttk.Entry(self.filter_frame, width=6)
        self.filter_btn = ttk.Button(self.filter_frame, text="Apply", command=self.apply_filter)
        
        # AI Processing
        self.ai_frame = ttk.LabelFrame(self.root, text="AI Processing")
        self.style_label = ttk.Label(self.ai_frame, text="Describe music style:")
        self.style_entry = ttk.Entry(self.ai_frame, width=30)
        self.emotion_label = ttk.Label(self.ai_frame, text="Emotion (-10 to +10):")
        self.emotion_slider = ttk.Scale(self.ai_frame, from_=-10, to=10, orient=tk.HORIZONTAL)
        self.ai_btn = ttk.Button(self.ai_frame, text="Process with AI", command=self.ai_process)
        
        # Visualization
        self.viz_frame = ttk.LabelFrame(self.root, text="Visualization")
        self.waveform_canvas = None
        self.spectrogram_canvas = None
        
        # Controls
        self.control_frame = ttk.Frame(self.root)
        self.play_btn = ttk.Button(self.control_frame, text="Play", command=self.play_audio)
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_audio)
        self.reset_btn = ttk.Button(self.control_frame, text="Reset", command=self.reset_audio)
        self.save_btn = ttk.Button(self.control_frame, text="Save...", command=self.save_audio)
        
        
        
        # Bind events
        self.pitch_slider.bind("<Motion>", lambda e: self.pitch_entry.delete(0, tk.END) or self.pitch_entry.insert(0, f"{self.pitch_slider.get():.1f}"))
        self.reverb_slider.bind("<Motion>", lambda e: self.reverb_entry.delete(0, tk.END) or self.reverb_entry.insert(0, f"{self.reverb_slider.get():.2f}"))
        self.dist_slider.bind("<Motion>", lambda e: self.dist_entry.delete(0, tk.END) or self.dist_entry.insert(0, f"{self.dist_slider.get():.2f}"))
        self.tempo_slider.bind("<Motion>", lambda e: self.tempo_entry.delete(0, tk.END) or self.tempo_entry.insert(0, f"{self.tempo_slider.get():.2f}"))
        # Set default values
        self.reset_sliders()

    def reset_sliders(self):
        self.pitch_slider.set(0)
        self.pitch_entry.delete(0, tk.END)
        self.pitch_entry.insert(0, "0.0")
        self.reverb_slider.set(0)
        self.reverb_entry.delete(0, tk.END)
        self.reverb_entry.insert(0, "0.0")
        self.dist_slider.set(0)
        self.dist_entry.delete(0, tk.END)
        self.dist_entry.insert(0, "0.0")
        self.tempo_slider.set(1.0)
        self.tempo_entry.delete(0, tk.END)
        self.tempo_entry.insert(0, "1.0")
        self.lowcut_entry.delete(0, tk.END)
        self.lowcut_entry.insert(0, "20")
        self.highcut_entry.delete(0, tk.END)
        self.highcut_entry.insert(0, "20000")
        self.emotion_slider.set(0)
        self.style_entry.delete(0, tk.END)

     
    def setup_layout(self):
        # File frame
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)
        self.load_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label.pack(side=tk.LEFT, padx=5)
        self.duration_label.pack(side=tk.LEFT, padx=5)
        
        # Manual effects frame
        self.manual_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Pitch
        self.pitch_frame.pack(fill=tk.X, padx=5, pady=2)
        self.pitch_label.pack(side=tk.LEFT)
        self.pitch_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.pitch_entry.pack(side=tk.LEFT, padx=5)
        self.pitch_btn.pack(side=tk.LEFT, padx=5)
        
        # Reverb
        self.reverb_frame.pack(fill=tk.X, padx=5, pady=2)
        self.reverb_label.pack(side=tk.LEFT)
        self.reverb_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.reverb_entry.pack(side=tk.LEFT, padx=5)
        self.reverb_btn.pack(side=tk.LEFT, padx=5)
        
        # Distortion
        self.dist_frame.pack(fill=tk.X, padx=5, pady=2)
        self.dist_label.pack(side=tk.LEFT)
        self.dist_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.dist_entry.pack(side=tk.LEFT, padx=5)
        self.dist_btn.pack(side=tk.LEFT, padx=5)
        
        # Tempo
        self.tempo_frame.pack(fill=tk.X, padx=5, pady=2)
        self.tempo_label.pack(side=tk.LEFT)
        self.tempo_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.tempo_entry.pack(side=tk.LEFT, padx=5)
        self.tempo_btn.pack(side=tk.LEFT, padx=5)
        
        # Filter
        self.filter_frame.pack(fill=tk.X, padx=5, pady=2)
        self.filter_label.pack(side=tk.LEFT)
        self.lowcut_label.pack(side=tk.LEFT, padx=5)
        self.lowcut_entry.pack(side=tk.LEFT)
        self.highcut_label.pack(side=tk.LEFT, padx=5)
        self.highcut_entry.pack(side=tk.LEFT)
        self.filter_btn.pack(side=tk.LEFT, padx=5)
        
        # AI frame
        self.ai_frame.pack(fill=tk.X, padx=5, pady=5)
        self.style_label.pack(side=tk.LEFT, padx=5)
        self.style_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.emotion_label.pack(side=tk.LEFT, padx=5)
        self.emotion_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.ai_btn.pack(side=tk.LEFT, padx=5)
        
        # Visualization frame
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        
    
    def load_audio(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac"), ("All Files", "*.*")])
        if filepath:
            if self.engine.load_audio(filepath):
                self.file_label.config(text=os.path.basename(filepath))
                duration = len(self.engine.current)/self.engine.sr
                self.duration_label.config(text=f"Duration: {duration:.1f}s")
                self.update_visualizations()
                self.update_status(f"Loaded: {os.path.basename(filepath)}")
            else:
                self.update_status("Failed to load audio file")
    
    def update_visualizations(self):
        # Clear existing canvases
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        if self.engine.waveform_fig and self.engine.spectrogram_fig:
            # Waveform
            waveform_canvas = FigureCanvasTkAgg(self.engine.waveform_fig, master=self.viz_frame)
            waveform_widget = waveform_canvas.get_tk_widget()
            waveform_widget.pack(fill=tk.BOTH, expand=True)
            
            # Spectrogram
            spectrogram_canvas = FigureCanvasTkAgg(self.engine.spectrogram_fig, master=self.viz_frame)
            spectrogram_widget = spectrogram_canvas.get_tk_widget()
            spectrogram_widget.pack(fill=tk.BOTH, expand=True)
    
    def apply_pitch(self):
        try:
            steps = float(self.pitch_entry.get())
            if -12 <= steps <= 12:
                self.engine.pitch_shift(steps)
                self.update_history()
                self.update_visualizations()
                self.update_status(f"Applied pitch shift: {steps} semitones")
            else:
                messagebox.showwarning("Invalid Value", "Pitch shift must be between -12 and 12 semitones")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def apply_reverb(self):
        try:
            wetness = float(self.reverb_entry.get())
            if 0 <= wetness <= 1:
                self.engine.reverb(wetness)
                self.update_history()
                self.update_visualizations()
                self.update_status(f"Applied reverb: {wetness*100:.0f}% wet")
            else:
                messagebox.showwarning("Invalid Value", "Wetness must be between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def apply_distortion(self):
        try:
            amount = float(self.dist_entry.get())
            if 0 <= amount <= 1:
                self.engine.distortion(amount)
                self.update_history()
                self.update_visualizations()
                self.update_status(f"Applied distortion: {amount*100:.0f}%")
            else:
                messagebox.showwarning("Invalid Value", "Distortion must be between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def apply_tempo(self):
        try:
            factor = float(self.tempo_entry.get())
            if 0.5 <= factor <= 2:
                self.engine.change_tempo(factor)
                self.update_history()
                self.update_visualizations()
                self.update_status(f"Changed tempo: {factor:.2f}x")
            else:
                messagebox.showwarning("Invalid Value", "Tempo factor must be between 0.5 and 2")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def apply_filter(self):
        try:
            lowcut = float(self.lowcut_entry.get()) if self.lowcut_entry.get() else None
            highcut = float(self.highcut_entry.get()) if self.highcut_entry.get() else None
            self.engine.apply_filter(lowcut, highcut)
            self.update_history()
            self.update_visualizations()
            self.update_status(f"Applied filter: {lowcut or 'None'}Hz-{highcut or 'None'}Hz")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid frequencies")
    
    def ai_process(self):
        style = self.style_entry.get().strip()
        emotion = self.emotion_slider.get()
        
        if not style:
            messagebox.showwarning("Input Needed", "Please describe a music style")
            return
        
        self.update_status("Analyzing style with AI...")
        
        def process():
            try:
                energy, complexity, ai_emotion = self.transformer.analyze_style(style)
                # Blend user emotion with AI emotion
                final_emotion = (emotion + ai_emotion) / 2
                
                effects = self.fuzzy.get_effects(energy, complexity, final_emotion)
                
                self.root.after(0, lambda: self.apply_ai_effects(effects, energy, complexity, final_emotion))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("AI Error", str(e)))
        
        threading.Thread(target=process).start()
    
    def apply_ai_effects(self, effects, energy, complexity, emotion):
        self.engine.pitch_shift(effects['pitch_shift'])
        self.engine.reverb(effects['reverb'])
        self.engine.distortion(effects['distortion'])
        self.engine.change_tempo(effects['tempo'])
        
        self.update_history()
        self.update_visualizations()
        
        info = (
            f" -- Music Improvisation and Production --\n"
            f"- Energy: {energy:.1f}/10\n"
            f"- Complexity: {complexity:.1f}/10\n"
            f"- Emotion: {emotion:.1f}\n\n"
            f"Applied Effects:\n"
            f"- Pitch: {effects['pitch_shift']:.1f} semitones\n"
            f"- Reverb: {effects['reverb']:.2f}\n"
            f"- Distortion: {effects['distortion']:.2f}\n"
            f"- Tempo: {effects['tempo']:.2f}x"
        )
        
        self.history_text.insert(tk.END, info + "\n\n")
        self.update_status("AI processing complete")
    
    def play_audio(self):
        if self.engine.current is not None:
            self.update_status("Playing...")
            self.engine.play()
            self.update_status("Playback finished")
    
    def stop_audio(self):
        sd.stop()
        self.update_status("Playback stopped")
    
    def reset_audio(self):
        if self.engine.reset():
            self.reset_sliders()
            self.update_history()
            self.update_visualizations()
            self.update_status("All effects reset")
        else:
            self.update_status("No audio to reset")
            
    def save_audio(self):
        if self.engine.current is None:
            messagebox.showwarning("No Audio", "No audio to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")])
        
        if filepath:
            if self.engine.save(filepath):
                self.update_status(f"Saved to {filepath}")
            else:
                self.update_status("Failed to save file")
    
    def update_history(self):
        self.history_text.delete(1.0, tk.END)
        for item in self.engine.history:
            self.history_text.insert(tk.END, item + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicFXApp(root)
    root.mainloop()
