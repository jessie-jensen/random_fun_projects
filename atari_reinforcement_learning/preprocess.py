import numpy as np


def preprocess_obs(obs):
    # crop & resize
    img = obs[1:176:2, ::2]
    # greyscale
    img = img.mean(axis=2)
    # increase contrast
    color = np.array([210, 164, 74]).mean()
    img[img==color] = 0
    # normalize
    img = ((img - 128) / 128) - 1
    # reshape
    img = img.reshape(88,80)

    return img