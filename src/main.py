import os
import absl.logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from keras import models
from yt_music import Downloader
import numpy as np
from score import Audio
from utils import min_max_scaling

class FeatureExtractor:
    def yt2mp3(self, yt_link):
        return Downloader.download(yt_link, "./data/music/main_runtime", True)

    def mfcc_to_X(self, filepath):
        audio = Audio(filepath=filepath, duration=30)
        _, _, mfcc = audio.get_mfcc(80, segment_size=10)
        mfcc = np.array(mfcc)
        mfcc = mfcc.transpose(2, 1, 0)
        mfcc = np.reshape(mfcc, (130, -1))
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.nan_to_num(mfcc, nan = 0.)
        return mfcc

    def get_features(self, filepath, encoder: models.Model):
        mfcc = self.mfcc_to_X(filepath)
        res = encoder.predict(mfcc)
        res = res.flatten()
        res = min_max_scaling(res)
        return res

def features_rebuild(features_str: str):
    return np.array(features_str.split(","), dtype=np.float32)

def main():
    extractor = FeatureExtractor()
    encoder = models.load_model("./models/best.h5")
    while True:
        yt_link = input("URL: ")
        filepath = extractor.yt2mp3(yt_link)
        if filepath is not None:
            features = ",".join(map(str, extractor.get_features(filepath, encoder)))
            print(f"Output: {features}")
            print(f"Rebuild: {features_rebuild(features)}")
            os.remove(filepath)
            
if __name__ == "__main__":
    main()