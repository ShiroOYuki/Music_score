import numpy as np
import matplotlib.pyplot as plt


def min_max_scaling(data: np.ndarray):
        s, b = min(data), max(data)
        return (data - s) / (b - s)

def show_matrix(musics, similarities, show_text = True):
    plt.imshow(100-similarities, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    plt.show()

    if show_text:
        for i, feature in enumerate(similarities):
            similarities[i][i] = 0
            similar_music_idx = np.where(similarities[i] == max(similarities[i]))[0][0]
            print(f"{i} ({musics[i]}) -> {similar_music_idx} ({musics[similar_music_idx]}) - {max(similarities[i])/100:.2%}")
            similarities[i][i] = 100