import numpy as np
import librosa
import sounddevice as sd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
from datetime import datetime
import json

class VoiceHealthAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 10  # seconds
        self.scaler = StandardScaler()
    
    def _numpy_to_python(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._numpy_to_python(item) for item in obj]
        return obj

    def record_audio(self):
        """Record audio from microphone"""
        print("Recording...")
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        return recording.flatten()

    def extract_voice_features(self, audio_data):
        """Extract relevant voice biomarkers"""
        # Extract various audio features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        # Calculate statistics for each feature
        features = {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),  # Convert to Python float
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),  # Convert to Python float
            'pitch': librosa.pyin(audio_data, sr=self.sample_rate, fmin=50, fmax=600)[0],
            'energy': float(np.sum(np.abs(audio_data)**2) / len(audio_data))  # Convert to Python float
        }
        return features

    def analyze_liver_condition(self, features):
        """Analyze voice biomarkers for liver disease indicators"""
        risk_factors = {
            'tremor': bool(self._detect_voice_tremor(features['mfcc_std'])),  # Convert to Python bool
            'weakness': float(self._analyze_voice_strength(features['energy'])),  # Convert to Python float
            'pitch_stability': bool(self._analyze_pitch_stability(features['pitch']))  # Convert to Python bool
        }
        
        risk_score = float(sum(risk_factors.values()) / len(risk_factors))  # Convert to Python float
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }

    def monitor_speech_patterns(self, features, baseline_features=None):
        """Monitor changes in speech patterns over time"""
        if baseline_features is None:
            return {'message': 'Baseline features needed for comparison'}
            
        changes = {
            'pitch_change': float(self._calculate_change(
                features['pitch'],
                baseline_features['pitch']
            )),
            'energy_change': float(self._calculate_change(
                features['energy'],
                baseline_features['energy']
            )),
            'articulation_change': float(self._calculate_change(
                features['mfcc_mean'],
                baseline_features['mfcc_mean']
            ))
        }
        return changes

    def analyze_emotions(self, features):
        """Analyze emotional state from voice"""
        emotions = {
            'stress_level': float(self._calculate_stress_level(features)),
            'anxiety_level': float(self._calculate_anxiety_level(features)),
            'depression_indicators': self._numpy_to_python(self._detect_depression_indicators(features))
        }
        return emotions

    def process_symptom_report(self, audio_data):
        """Process voice-activated symptom reporting"""
        symptoms = {
            'timestamp': datetime.now().isoformat(),
            'duration': float(len(audio_data) / self.sample_rate),
            'processed': True
        }
        return symptoms

    def _detect_voice_tremor(self, mfcc_std):
        """Detect voice tremor from MFCC standard deviation"""
        return np.mean(mfcc_std) > 0.5

    def _analyze_voice_strength(self, energy):
        """Analyze voice strength from energy"""
        return 1 - min(energy / 100, 1)

    def _analyze_pitch_stability(self, pitch):
        """Analyze pitch stability"""
        return np.std(pitch[~np.isnan(pitch)]) > 0.1

    def _calculate_change(self, current, baseline):
        """Calculate percentage change between current and baseline values"""
        if isinstance(current, np.ndarray):
            return float(np.mean(np.abs(current - baseline) / baseline) * 100)
        return float(abs(current - baseline) / baseline * 100)

    def _calculate_stress_level(self, features):
        """Calculate stress level from voice features"""
        return float(min(features['energy'] / 100 + np.mean(features['mfcc_std']), 1.0))

    def _calculate_anxiety_level(self, features):
        """Calculate anxiety level from voice features"""
        return float(min(np.mean(features['zero_crossing_rate_mean']) * 2, 1.0))

    def _detect_depression_indicators(self, features):
        """Detect depression indicators from voice features"""
        indicators = {
            'monotone': bool(features['pitch'].std() < 0.1),
            'low_energy': bool(features['energy'] < 50),
            'slow_speech': bool(features['zero_crossing_rate_mean'] < 0.1)
        }
        return indicators

    def run_health_assessment(self):
        """Run a complete health assessment"""
        try:
            # Record audio
            audio_data = self.record_audio()
            
            # Extract features
            features = self.extract_voice_features(audio_data)
            
            # Run all analyses
            assessment = {
                'liver_analysis': self.analyze_liver_condition(features),
                'speech_patterns': self.monitor_speech_patterns(features),
                'emotional_state': self.analyze_emotions(features),
                'symptom_report': self.process_symptom_report(audio_data),
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert all numpy types to Python native types before returning
            return self._numpy_to_python(assessment)
            
        except Exception as e:
            return {'error': str(e)}

    def save_assessment(self, assessment, filename):
        """Save assessment results to file"""
        with open(filename, 'w') as f:
            json.dump(assessment, f, indent=4)

# Example usage
if __name__ == "__main__":
    analyzer = VoiceHealthAnalyzer()
    
    # Run complete health assessment
    assessment = analyzer.run_health_assessment()
    
    # Save results
    analyzer.save_assessment(assessment, 'health_assessment_results.json')
    
    print("Assessment completed and saved to file.")