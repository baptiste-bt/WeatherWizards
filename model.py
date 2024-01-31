#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z |-> G_\theta(Z)
############################################################################

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>

import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_variable = noise[:, 0]  # choose the appropriate latent dimension of your model
    seed = hash(np.sum(noise)) % 2**32

    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    with open('parameters/gmm.pkl', 'rb') as f:
        model = pickle.load(f)

    # Used seed to sample from the model
    np.random.seed(seed)
    
    n_samples = noise.shape[0]

    # Sample from the model
    samples = model.sample(n_samples)[0]

    # Clip the samples to match the range of the data
    samples = np.clip(samples, 0, 14.03)

    return samples




