import librosa
import numpy as np
import os
import ast

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas as pd

from typing import Literal, Optional
from scipy.stats import kurtosis, skew

class AudioFeatures:
    """
    AudioFeatures is a container class for storing various statistical features derived from audio data.

    This class holds common statistical measures such as kurtosis, maximum, mean, median, minimum, skewness, 
    and standard deviation, which are often computed from audio features like MFCCs, chroma, or spectrograms.

    Attributes
    ----------
    kurtosis : Optional[np.ndarray]
        The kurtosis of the audio feature data, describing the "tailedness" of the data distribution.
        
    max : Optional[np.ndarray]
        The maximum value for each feature over time, providing a sense of the peak values in the data.

    mean : Optional[np.ndarray]
        The mean value for each feature, giving a central tendency of the data distribution.

    median : Optional[np.ndarray]
        The median value for each feature, providing a robust measure of central tendency less affected by outliers.

    min : Optional[np.ndarray]
        The minimum value for each feature, representing the smallest observed values.

    skew : Optional[np.ndarray]
        The skewness of the audio feature data, describing the asymmetry of the data distribution.

    std : Optional[np.ndarray]
        The standard deviation for each feature, indicating how much the values fluctuate over time.
    """

    def __init__(
        self,
        kurtosis: Optional[np.ndarray] = None,
        max: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None,
        median: Optional[np.ndarray] = None,
        min: Optional[np.ndarray] = None,
        skew: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        self.kurtosis = kurtosis
        self.max = max
        self.mean = mean
        self.median = median
        self.min = min
        self.skew = skew
        self.std = std
        
    def __str__(self):
        df = pd.DataFrame({
            "kutosis": self.kurtosis,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "skew": self.skew,
            "std": self.std
        })
        return df.__str__()

    def __array__(self, dtype=None):
        return self.to_array(dtype=dtype)
    
    def to_array(self, dtype=None):
        return np.array([
            self.kurtosis,
            self.max,
            self.mean,
            self.median,
            self.min,
            self.skew,
            self.std
        ], dtype=dtype)
    
class AudioTools:
    @classmethod
    def get_max(cls, data: np.ndarray, sr=22050, y_axis: Literal["chroma", "mel"] = None, show = True, frame_end = 1000):
        """
        Get row-based max array from the input audio feature matrix.

        This method computes the maximum value for each column (time frame) across the rows (features) 
        in the input 2D audio feature matrix `data`. The result is a binary matrix where the maximum 
        value in each column is marked as 1, and all other values are set to 0.

        Optionally, the method can display the original audio feature matrix, its dB-scaled version, 
        and the binary max array using `librosa`'s visualization tools if `show` is set to True.

        Arguments
        -------
            data (np.ndarray): 
                A 2D array representing the audio feature matrix (e.g., chroma, mel-spectrogram).
            sr (int, optional): _Defaults to 22050._
                The sampling rate of the audio signal. 
            y_axis (Literal['chroma', 'mel'], optional): _Defaults to None._
                The type of y-axis for the display. If 'chroma', the chroma axis will be used. 
                If 'mel', the Mel-frequency axis will be used. 
            show (bool, optional): _Defaults to False._
                If True, the original data, its dB-scaled version, and the max array will be displayed 
                using `librosa.display.specshow`. 
            frame_end (int, optional): _Defaults to 1000._
                The number of frames to display in the visualization. Only the first `frame_end` frames 
                will be shown if `show` is True. 

        Returns
        -------
            max_d (np.ndarray): 
                A 2D binary array of the same shape as `data`, where the maximum value in each column 
                is set to 1, and all other values are set to 0.
        """
        idx = np.argmax(data, axis=0)
        max_d = np.zeros(data.shape)
        max_d[idx, np.arange(data.shape[1])] = 1
        
        if show:
            D = cls.get_db(data)
            fig, ax = plt.subplots(3, 1, sharex=True)
            cls.show(data, frame_end=frame_end, title="original", y_axis=y_axis, ax=ax[0], show=False)
            cls.show(D, frame_end=frame_end, title="dB", y_axis=y_axis, ax=ax[1], show=False)
            cls.show(max_d, frame_end=frame_end, title="max", y_axis=y_axis, ax=ax[2], show=False)
            fig.tight_layout()
            plt.show()
                
        return max_d
    
    @classmethod
    def get_db(cls, data: np.ndarray):
        return librosa.amplitude_to_db(np.abs(data), ref=np.max)
    
    @classmethod
    def show(
        cls, 
        data: np.ndarray, 
        sr: Optional[int] = 22050, 
        title: Optional[str] = "",
        frame_end: Optional[int] = 1000, 
        y_axis: Optional[Literal["chroma", "mel"]] = None, 
        ax: Optional[axes.Axes] = None,
        show: Optional[bool] = True
    ):
        """
        Display a visual representation of the provided audio feature data (such as Mel-spectrogram or Chroma).

        This method visualizes the given 2D audio feature array (e.g., Mel-spectrogram or Chroma) and allows
        customization of the axes, title, and plot display. It can display up to a certain number of frames
        specified by `frame_end`.

        Arguments
        -------
        data (np.ndarray): 
            The 2D array representing the audio feature data to display (e.g., Mel-spectrogram, Chroma).
            
        sr (Optional[int], optional): _Defaults to 22050._
            The sampling rate of the audio. 
            
        title (Optional[str], optional): _Defaults to \"\"._
            The title for the plot. 
            
        frame_end (Optional[int], optional): _Defaults to 1000._
            The number of frames to display from the data. 
        y_axis (Optional[Literal[\"chroma\", \"mel\"]], optional): _Defaults to None._
            The type of y-axis to display. Can be 'chroma' or 'mel'. 
        ax (Optional[axes.Axes], optional): _Defaults to None._
            An optional Matplotlib axes object on which to plot. If None, a new figure and axes will be created. 
        show (Optional[bool], optional): _Defaults to True._
            If True, the plot will be displayed immediately using `plt.show()`. If False, the plot will be created 
            but not shown. 
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, sharex=True)
        img = librosa.display.specshow(data[:, :frame_end], sr=sr, x_axis="time", y_axis=y_axis, ax=ax)
        plt.colorbar(img, ax=ax)
        ax.set(title=title)
        if show:
            plt.show()
    
    @classmethod
    def get_stats(cls, data: np.ndarray):
        data = data.astype(np.float64)
        return AudioFeatures(
            kurtosis = kurtosis(data, axis=1),
            max = np.max(data, axis=1),
            mean = np.mean(data, axis=1),
            median = np.median(data, axis=1),
            min = np.min(data, axis=1),
            skew = skew(data, axis=1),
            std = np.std(data, axis=1)
        )
        
    @classmethod
    def get_stats_2D(cls, data: np.ndarray):
        data = data.astype(np.float64)
        return AudioFeatures(
            kurtosis = kurtosis(data, axis=2),
            max = np.max(data, axis=2),
            mean = np.mean(data, axis=2),
            median = np.median(data, axis=2),
            min = np.min(data, axis=2),
            skew = skew(data, axis=2),
            std = np.std(data, axis=2)
        )
    
    
    
class FMA:
    def __init__(self):
        self.features = None
        self.echonest = None
        self.genres = None
        self.tracks = None
    
    def load(self, filepath: str):
        filename = os.path.basename(filepath)
        
        if 'features' in filename:
            self.features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
            return self.features

        if 'echonest' in filename:
            self.echonest = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
            return self.echonest

        if 'genres' in filename:
            self.genres = pd.read_csv(filepath, index_col=0)
            return self.genres

        if 'tracks' in filename:
            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
            
            # 將 csv 內的字串轉換為正確的資料型態
            COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                    ('track', 'genres'), ('track', 'genres_all')]
            for column in COLUMNS:
                tracks[column] = tracks[column].map(ast.literal_eval)

            # 將 pd.table 內有關時間的欄位轉換為 datetime
            COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                    ('album', 'date_created'), ('album', 'date_released'),
                    ('artist', 'date_created'), ('artist', 'active_year_begin'),
                    ('artist', 'active_year_end')]
            for column in COLUMNS:
                tracks[column] = pd.to_datetime(tracks[column])

            
            SUBSETS = ('small', 'medium', 'large')
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                        pd.CategoricalDtype(categories=SUBSETS, ordered=True))

            COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                    ('album', 'type'), ('album', 'information'),
                    ('artist', 'bio')]
            for column in COLUMNS:
                tracks[column] = tracks[column].astype('category')

            self.tracks = tracks
            
            return self.tracks
    
    def train_data(self, size: Literal['small', 'medium']):
        size = self.tracks['set', 'subset'] <= size
        
        train = self.tracks['set', 'split'] == 'training'
        val = self.tracks['set', 'split'] == 'validation'
        test = self.tracks['set', 'split'] == 'test'

        X_train = self.features.loc[size & train, 'mfcc']
        X_val = self.features.loc[size & val, 'mfcc']
        X_test = self.features.loc[size & test, 'mfcc']

        Y_train = self.tracks.loc[size & train, ('track', 'genre_top')]
        Y_val = self.tracks.loc[size & val, ('track', 'genre_top')]
        Y_test = self.tracks.loc[size & test, ('track', 'genre_top')]
        
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    
    def top_genres(self, size=Literal['small', 'medium'], show=False):
        size = self.tracks['set', 'subset'] <= size
        top_genres = self.tracks.loc[size, ('track', 'genre_top')].unique()
        
        if show:
            print("Genres".ljust(20, " ") + " |  Count")
            print("-"*28)
            for tg in top_genres:
                count = len(self.tracks.loc[size & (self.tracks['track', 'genre_top'] == tg)])
                print(f"{tg.ljust(20, " ")} | \t{count}")
        
        return top_genres
    
def min_max_scaling(data: np.ndarray):
    s, b = min(data), max(data)
    return (data - s) / (b - s)