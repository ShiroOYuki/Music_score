import librosa
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Callable
from utils import AudioTools, AudioFeatures

class Audio:
    """
    Attributes
    -------
        filepath (str): 
            The file path to the audio file.
        y (np.ndarray): 
            The audio time series data loaded from the file. 
            Initially set to None and later populated when the audio is loaded.
        sr (int): 
            The sampling rate of the loaded audio file. Initially set to None 
            and assigned when the audio is loaded.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.y = None
        self.sr = None
        self._read_audio()
        
    def _read_audio(self):
        # y: wav | sr: sampling rate
        self.y, self.sr = librosa.load(self.filepath, duration=10)
        
    def get_tempo(self):
        """
        Get BPM (beats per minute) for the audio signal.
        
        Returns
        -------
            tempo (np.ndarray):  
                An array of estimated tempo (in BPM). Typically, this will return a single value.
        """
        tempo = librosa.feature.tempo(y=self.y, sr=self.sr)
        return tempo

    def _extract_features(self, feature_func: Callable, **kwargs) -> tuple[np.ndarray, np.ndarray, AudioFeatures]:
        """
        Extract audio features, convert them to dB format, and compute various statistical measures.

        This is a generic function that extracts audio features using the provided `feature_func`, converts 
        the features to dB scale, and computes statistics such as kurtosis, mean, median, max, min, skew, and std.

        Arguments
        ----------
            feature_func (function):
                A Librosa feature extraction function (e.g., librosa.feature.mfcc, librosa.feature.chroma_cqt).
            kwargs (dict):
                Additional keyword arguments to pass to the feature extraction function.

        Returns
        -------
            data (np.ndarray):
                The raw extracted features in their original scale.
            db (np.ndarray):
                The features converted to dB scale for perceptual relevance.
            stats (AudioFeatures):
                An object containing statistical features (kurtosis, mean, median, max, min, skew, std) 
                computed from the dB-scaled features.
        """
        data = feature_func(y=self.y, sr=self.sr, **kwargs)
        db = AudioTools.get_db(data)
        stats = AudioTools.get_stats(db)
        return data, db, stats
    
    def get_mfcc(self, n_mfcc: Optional[int]=20) -> tuple[np.ndarray, np.ndarray, AudioFeatures]:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features, convert them to dB scale, 
        and compute their statistical properties.

        Arguments
        -------
            n_mfcc (Optional[int], optional): _Defaults to 20._
                The number of MFCCs to extract.
        
        Returns
        -------
            data (np.ndarray):
                The raw extracted features in their original scale.
            db (np.ndarray):
                The features converted to dB scale for perceptual relevance.
            stats (AudioFeatures):
                An object containing statistical features (kurtosis, mean, median, max, min, skew, std) 
                computed from the dB-scaled features.
        """
        return self._extract_features(librosa.feature.mfcc, n_mfcc=n_mfcc)
    
    def get_cqt(self):
        """
        Extract Constant-Q chromagram (CQT) features, convert them to dB scale, 
        and compute their statistical properties.
        
        Returns
        -------
            cqt (np.ndarray):
                The raw CQT features.
            db (np.ndarray):
                The CQT features in dB scale.
            stats (AudioFeatures):
                Statistical properties of the dB-scaled CQT features (kurtosis, mean, median, max, min, skew, std).
        """
        return self._extract_features(librosa.feature.chroma_cqt)
    
    def get_cens(self):
        """
        Extract CENS (Chroma Energy Normalized) features, convert them to dB scale, 
        and compute their statistical properties.
        
        Returns
        -------
            cens (np.ndarray):
                The raw CENS features.
            db (np.ndarray):
                The CENS features in dB scale.
            stats (AudioFeatures):
                Statistical properties of the dB-scaled CENS features (kurtosis, mean, median, max, min, skew, std).
        """
        return self._extract_features(librosa.feature.chroma_cens)
    
    def get_mel(self):
        """
        Extract Mel-spectrogram features, convert them to dB scale, 
        and compute their statistical properties.
        
        Returns
        -------
            mel (np.ndarray):
                The raw Mel-spectrogram features.
            db (np.ndarray):
                The Mel-spectrogram features in dB scale.
            stats (AudioFeatures):
                Statistical properties of the dB-scaled Mel-spectrogram features (kurtosis, mean, median, max, min, skew, std).
        """
        return self._extract_features(librosa.feature.melspectrogram)
    
if __name__ == "__main__":
    audio = Audio(r"D:\Code\Music_score\src\test.mp3")
    tempo = audio.get_tempo()
    
    print(f"Tempo: {tempo}")
    
    MFCC, D_MFCC, mfcc_stats = audio.get_mfcc()
    CQT, D_CQT, cqt_stats = audio.get_cqt()
    MEL, D_MEL, mel_stats = audio.get_mel()
    CENS, D_CENS, cens_stats = audio.get_cens()

    # AudioTools.show(CQT, sr=audio.sr, y_axis='chroma', title="CQT")
    # AudioTools.show(D_MEL, sr=audio.sr, y_axis='mel', title="MEL (dB)")
    # AudioTools.show(CENS, sr=audio.sr, y_axis='chroma', title="CENS")
    # AudioTools.show(D_MFCC, sr=audio.sr, title="MFCC (dB)")
    
    print(mfcc_stats)
    
    # cqt 用於分析音高
    # mel 主要用於視覺化
    # mfcc 丟給 LSTM/RNN 運算
    
    
        
        
    