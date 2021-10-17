import os
import numpy as np
import tensorflow as tf

# Dataset meta-info
CLASS_TO_LABEL = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
LABEL_TO_CLASS = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}
N_CLASS = 10


def build_batch(batch_indexes, memmaps, input_shape):
    batch = np.zeros((len(batch_indexes), *input_shape))

    for c in range(N_CLASS):
        examples_to_load = np.sum(batch_indexes == c)
        batch[batch_indexes == c] = memmaps[c][:examples_to_load]
        memmaps[c] = memmaps[c][examples_to_load:]

    return batch, batch_indexes


def dataset_generator(root, batch_size, seed):
    rng = np.random.default_rng(seed)

    if isinstance(root, bytes):
        root = root.decode("utf-8")

    for _, _, files in os.walk(root):
        files = [f.decode("utf-8") if isinstance(f, bytes) else f for f in files]
    memmaps = [np.load(os.path.join(root, f), mmap_mode="r") for f in files]

    lengths_of_labels = {f.split(".")[0]: mmap.shape[0] for f, mmap in zip(files, memmaps)}
    batches = []
    for label, length in lengths_of_labels.items():
        batches.extend([LABEL_TO_CLASS[label]] * length)

    batches = np.array(batches)
    rng.shuffle(batches)

    if batches.shape[0] % batch_size != 0:
        rest = batches.shape[0] % batch_size

        last_batch = batches[-rest:]
        batches = batches[:-rest].reshape((-1, batch_size))
    else:
        last_batch = None
        batches = batches.reshape((-1, batch_size))

    single_input_shape = memmaps[0].shape[1:]
    for batch in batches:
        yield build_batch(batch, memmaps, single_input_shape)

    if last_batch is not None:
        yield build_batch(last_batch, memmaps, single_input_shape)


def dataset_initializer(root, dataset_folder, batch_size, seed, input_shape):

    dataset_root = os.path.join(root, dataset_folder)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError("Dataset not found")

    for _, _, files in os.walk(dataset_root):
        if len(files) != N_CLASS:
            raise FileNotFoundError(f"Invalid dataset: found {len(files)} files but expected {N_CLASS}")

        for label in CLASS_TO_LABEL:
            if f"{label}.npy" not in files:
                raise FileNotFoundError(f"Invalid dataset: couldn't find file {label}.npy")

    return tf.data.Dataset.from_generator(
        dataset_generator,
        args=(tf.constant(dataset_root), tf.constant(batch_size), tf.constant(seed)),
        output_signature=(
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(1)
