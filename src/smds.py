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
        s: 
        0: ticks
        1: n_mfcc
        2: stats
        
        reshaped:
        0: ticks
        1: n_mfcc*stats
        """
        reshaped = s.reshape(s.shape[0], -1)

        math_stats = ['kurtosis', 'max', 'mean', 'median', 'min', 'skew', 'std']
        multi_columns = pd.MultiIndex.from_product([range(80), math_stats], names=['n_mfcc', 'stat'])
        df = pd.DataFrame(data=reshaped, columns=multi_columns)
        df.index.name = 'ticks'

        return df
    
    def make(self, dirpath: str, output_path: str):
        files = glob.glob(os.path.join(os.path.abspath(dirpath), "*.mp3"))

        n_file = len(files)
        df_list = [None]*n_file

        with alive_bar(n_file) as bar:
            for i, file in enumerate(files):
                s = self.read_stats(file)
                sdf = self.single_df(s)
                df_list[i] = sdf
                bar()
                
        def split_id(file):
            id = os.path.splitext(os.path.basename(file))[0]
            return id if not id.isdigit() else int(id)
            
        
        dataset = pd.concat(df_list, keys=[split_id(i) for i in files], names=['track_id', 'tick'])
        dataset.to_csv(os.path.abspath(output_path), index=True)



class SMDS:
    def __init__(self):
        pass
    
    def load(self, filepath: str):
        filepath = os.path.abspath(filepath)
        df = pd.read_csv(filepath, index_col=[0, 1], header=[0, 1, 2])
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

    files = glob.glob(os.path.join(dirpath, "*.mp3"))
    data = df_array.reshape((len(files), -1, 80, 7))
    print(data.shape)
    
    print(maker.read_stats(files[0])[100, 1, :])
    print(data[0, 100, 1, :])
    

if __name__ == '__main__':
    dirpath = "./data/fma_small/fma_small/**"
    output = "./data/smds/smds.csv"

    maker = SMDSMaker()
    maker.make(dirpath, output)
    
    smds = SMDS()
    df = smds.load(output)
    print(df) 