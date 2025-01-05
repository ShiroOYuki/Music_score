import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

from keras import Model
from keras.src.callbacks import Callback

import numpy as np
import sys
from typing import Callable

class CustomProgressBar(Callback):
    def __init__(self, total_epoch: int, name: str):
        self.total_epoch = total_epoch
        self.name = name
        self.count = 1
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        # 打印簡單的 epoch 進度條
        sys.stdout.write(f'\rEpoch {epoch+1}/{self.total_epoch} - loss: {logs["loss"]:.4f} - val_loss: {logs["val_loss"]:.4f}')
        
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_train_begin(self, logs=None):
        sys.stdout.write(f'# {self.name}\n')
    
    def on_train_end(self, logs=None):
        pass

class TestModel:
    class ModelSettings:
        def __init__(self, epochs: int=50, batch_size: int=32):
            self.epochs = epochs
            self.batch_size = batch_size
            
            
    def __init__(self, func: Callable[[], list[Model, Model]], settings: tuple[ModelSettings], name:str="Model"):
        self.model_func = func
        self.settings = settings
        self.name = name
        
    def _draw_loss(self, hist, fig: Figure=None, ax: Axes=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Loss Over Time', fontsize=20)
        
        ax[0].plot(hist.history['loss'], label='Training Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(hist.history['val_loss'], label='Validation Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        plt.tight_layout()
        
    def _show_loss(self, **kwargs):
        self._draw_loss(**kwargs)
        plt.show()
        
    def _draw_tsne(self, X_test, Y_test, encoder: Model, fig: Figure=None, ax: Axes=None, output_shape=20):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle('2D Visualization of Encoded Features', fontsize=20)
        
        ax.set_box_aspect(1) 
        encoded_features = encoder.predict(X_test)
        tsne = TSNE(n_components=2)
        encoded_2d = tsne.fit_transform(encoded_features.reshape(-1, output_shape))
        
        im = ax.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=Y_test, cmap='inferno', alpha=1)
        
        norm = Normalize(vmin=np.min(Y_test), vmax=np.max(Y_test))
        sm = ScalarMappable(norm=norm, cmap='inferno')
        sm.set_array([])
        
        fig.colorbar(sm, ax=ax)
        
    def _show_tsne(self, **kwargs):
        Y = kwargs.get("Y_test")
        kwargs["Y_test"] = self.encode_labels(Y)
        self._draw_tsne(**kwargs)
        plt.show()
    
    def encode_labels(self, Y):
        le = LabelEncoder()
        le.fit(Y)
        Y = le.transform(Y)
        return Y
    
    def test(self, x, y, val=None, validation_split=None, X_test=None, Y_test=None, name: str = "Model", output_shape=20):
        
        assert X_test is not None
        assert Y_test is not None
        assert val is not None or validation_split is not None
        
        count = len(self.settings)
        fig_loss, ax_loss = plt.subplots(count, 2, figsize=(6, 3*count))
        fig_tsne, ax_tsne = plt.subplots(1, count, figsize=(6*count, 6))
        
        for ax in ax_loss.flatten():
            ax.set_box_aspect(1)
            
        for ax in ax_tsne.flatten():
            ax.set_box_aspect(1)
            
            
        for i, setting in enumerate(self.settings):
            encoder, decoder = self.model_func()
            decoder.compile(optimizer='adam', loss='mse')
            if val is not None:
                hist = decoder.fit(
                    x, 
                    y, 
                    validation_data=val, 
                    epochs=setting.epochs, 
                    batch_size=setting.batch_size, 
                    verbose=0, 
                    callbacks=[
                        CustomProgressBar(
                            total_epoch=setting.epochs,
                            name = f"{name} - {i+1}"
                        )
                    ]
                )
            elif validation_split is not None:
                hist = decoder.fit(
                    x, 
                    y, 
                    validation_split=validation_split, 
                    epochs=setting.epochs, 
                    batch_size=setting.batch_size, 
                    verbose=0, 
                    callbacks=[
                        CustomProgressBar(
                            total_epoch=setting.epochs,
                            name = f"{name} - {i+1}"
                        )
                    ]
                )
            self._draw_loss(hist, fig=fig_loss, ax=ax_loss[i])
            self._draw_tsne(X_test, self.encode_labels(Y_test), encoder=encoder, fig=fig_tsne, ax=ax_tsne[i], output_shape=output_shape)
            
        fig_loss.suptitle('Loss Over Time', fontsize=20)
        fig_tsne.suptitle('2D Visualization of Encoded Features', fontsize=20)
        
        fig_loss.tight_layout()
        fig_tsne.tight_layout()
        plt.show()