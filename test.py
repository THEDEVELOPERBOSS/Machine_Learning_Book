import tensorflow_datasets as tfds

print("TFDS Version:", tfds.__version__)

ds = tfds.load(
    "imdb_reviews",
    split="train",
    download=True,
    as_supervised=True
)

print("Success!")
# used to test bits of code that aren't working