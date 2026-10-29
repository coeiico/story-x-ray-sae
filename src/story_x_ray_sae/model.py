"""
The model for story-x-ray-sae.

a sparse autoencoder (one hidden layer 8–32× wider than the input, ReLU with an L1
penalty, or TopK) trained on residual-stream activations of a pretrained TinyStories
model loaded with Hugging Face Transformers, such as roneneldan/TinyStories-33M.
Optional second target: the owner's own tiny-transformer trained on TinyStories.

forward() accepts return_cache=True and returns a dict of internal activations, and a
patch argument (or forward hooks) overwrites a named activation for interventions.
"""

# TODO: implement. See README.md for the plan.
