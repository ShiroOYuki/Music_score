# Shiro's Music Dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from score import Audio
import numpy as np
import glob
import pandas as pd
from alive_progress import alive_bar

class SMDSMaker:
    def read_stats(self, filepath: str):
        audio = Audio(filepath, duration=30)
        # 0: stats
        # 1: n_mfcc
        # 2: ticks
        _, _, stats = audio.get_mfcc(80, 10)
        stats = np.array(stats).transpose(2, 1, 0)
        return stats

    def single_df(self, s):
        """
        Reshapes the input NumPy array `s` and converts it into a pandas DataFrame.

        Parameters
        -------
        s (np.ndarray, shape=(ticks, n_mfcc, stats)): 
            A 3D NumPy array containing time steps (ticks), MFCC features (n_mfcc), and statistical values (stats).
            - ticks: The number of time steps
            - n_mfcc: The number of MFCC features at each time step
            - stats: The statistical values for each MFCC feature

        Returns
        -------
        df (pd.DataFrame): 
            A pandas DataFrame with the index as 'ticks' and multi-level columns.
            The first level of the columns is the range of `n_mfcc` (total 80 features),
            and the second level corresponds to the statistical values ('kurtosis', 'max', 
            'mean', 'median', 'min', 'skew', 'std').
        """
        reshaped = s.reshape(s.shape[0], -1)

        math_stats = ['kurtosis', 'max', 'mean', 'median', 'min', 'skew', 'std']
        multi_columns = pd.MultiIndex.from_product([range(80), math_stats], names=['n_mfcc', 'stat'])
        df = pd.DataFrame(data=reshaped, columns=multi_columns)
        df.index.name = 'ticks'

        return df
    
    def make(self, dirpath: str, output_path: str):
        """
        Processes a directory of .mp3 files, extracts statistical features, and saves the result as a CSV file.

        Parameters
        ----------
        dirpath : str
            The path to the directory containing the .mp3 files.
        output_path : str
            The path where the resulting CSV file will be saved.

        Returns
        -------
        None
            The function does not return anything. The result is saved as a CSV file at the specified `output_path`.
        
        Process
        -------
        - The function searches for all .mp3 files in the given directory.
        - For each file, it reads statistical features using `self.read_stats(file)` and converts the result into a DataFrame using `self.single_df(s)`.
        - It combines all individual DataFrames into a single DataFrame with a multi-level index. The first level corresponds to `track_id` (extracted from the file name), and the second level corresponds to `tick` (time points).
        - The final DataFrame is saved as a CSV file.
        """
        files = glob.glob(os.path.join(os.path.abspath(dirpath), "*.wav"))

        n_file = len(files)
        df_list = [None]*n_file

        with alive_bar(n_file) as bar:
            for i, file in enumerate(files):
                try:
                    s = self.read_stats(file)
                    sdf = self.single_df(s)
                    df_list[i] = sdf
                    bar()
                except:
                    print(f"Error: {file}")
                    bar()
                    continue
                    
                
        def split_id(file):
            # id = os.path.splitext(os.path.basename(file))[0]
            # return id if not id.isdigit() else int(id)
            return os.path.basename(file)
            
        
        dataset = pd.concat(df_list, keys=[(i, split_id(f)) for i, f in enumerate(files)], names=['id', 'filename', 'tick'])
        print("Saving...")
        if (not os.path.isdir(os.path.dirname(output_path))):
            os.makedirs(os.path.dirname(output_path))
        dataset.to_csv(os.path.abspath(output_path), index=True)



class SMDS:
    def __init__(self):
        pass
    
    def load(self, filepath: str):
        filepath = os.path.abspath(filepath)
        df = pd.read_csv(filepath, index_col=[0, 1, 2], header=[0, 1])
        return df
    
    def to_array(self, df: pd.DataFrame) -> np.ndarray:
        return df.to_numpy(dtype=np.float64)


def test():
    dirpath = "./data/music/download"
    output = "./data/smds/smds.csv"

    maker = SMDSMaker()
    maker.make("./data/music/download", output)

    smds = SMDS()
    df = smds.load(output)
    df_array = smds.to_array(df)

    files = glob.glob(os.path.join(dirpath, "*.wav"))
    data = df_array.reshape((len(files), -1, 80, 7))
    print(data.shape)
    
    print(maker.read_stats(files[0])[100, 1, :])
    print(data[0, 100, 1, :])
    


if __name__ == '__main__':
    dirpath = "./data/gtzan/genres_original/**"
    output = "./data/smds/smds.csv"

    print("Making...")
    maker = SMDSMaker()
    maker.make(dirpath, output)
    
    print("Loading...")
    smds = SMDS()
    df = smds.load(output)
    print(df) 