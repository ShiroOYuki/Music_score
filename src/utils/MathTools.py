import numpy as np
import matplotlib.pyplot as plt

def min_max_normalization(data: np.ndarray, upper, lower):
    return (data - lower) / (upper - lower)

def min_max_scaling(data: np.ndarray):
        u, l = max(data), min(data)
        return min_max_normalization(data, u, l)

def show_matrix(musics, similarities, show_text = True):
    plt.imshow(100-similarities*100, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    plt.show()

    if show_text:
        for i, feature in enumerate(similarities):
            similarities[i][i] = 0
            similar_music_idx = np.where(similarities[i] == max(similarities[i]))[0][0]
            print(f"{i} ({musics[i]}) -> {similar_music_idx} ({musics[similar_music_idx]}) - {max(similarities[i]):.2%}")
            similarities[i][i] = 1