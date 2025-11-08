# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "bacpipe",
# ]
# ///
import bacpipe

# em = bacpipe.Embedder('birdnet', **vars(bacpipe.settings))
# # the vars part is important!

# audio_file = './audio/FewShot/CHE_01_20190101_163410.wav'
# embeddings = em.get_embeddings_from_model(audio_file)

# print(embeddings.shape)

bacpipe.play(save_logs=True)
