import bacpipe
import numpy as np

def main():
    embedding_path = r"./bacpipe_results\test_data\embeddings\2025-11-09_10-27___birdnet-test_data\audio\FewShot\CHE_01_20190101_163410_birdnet.npy"
    print(np.load(embedding_path).shape)
    # bacpipe.play(save_logs=True)

if __name__ == "__main__":
    main()
