import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import preprocessing
import utils

logger = logging.getLogger('preprocessing')
logger.setLevel(logging.INFO)


def preprocess_images(img_name_vector, image_features_extract_model):
    '''Extracts and saves image features for each image in img_name_vector.'''
    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
      utils.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    logger.info('Extracting image features. This may take a while...')
    for i, (img, path) in enumerate(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

        if i % 50 == 0:
            logger.info(f'Image feature extract, {i} batches done.')

    logger.info('Saved all the image features.')


def preprocess_text(train_captions, vocab_size):
    '''UNKS, tokenizes, and pads captions in train_captions.'''
    logger.info(f'Vocab size: {vocab_size}')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    # TODO: check why this line appears twice
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    caption_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    return caption_vector, tokenizer, train_seqs

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def create_dataset(image_name_vector,
                   caption_vector,
                   train_seqs,
                   test_size=0.2,
                   batch_size=64,
                   buffer_size=1000):
    '''Creates the dataset. Returns it shuffled and batched.'''
    logger.info(f'Train-test split {1 - test_size}/{test_size}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Buffer size: {buffer_size}')
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)
    logger.info(f'Max sequence length: {max_length}')

    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(image_name_vector,
                                                                        caption_vector,
                                                                        test_size=test_size,
                                                                        random_state=0)

    logger.info(f'Train images: {len(img_name_train)}, train captions: {len(cap_train)}, val images: { len(img_name_val)}, val captions: {len(cap_val)}')
    num_steps_train = len(img_name_train) // batch_size
    num_steps_val = len(img_name_val) // batch_size

    # Load the numpy files
    def map_func(img_name, cap):
      img_tensor = np.load(img_name.decode('utf-8')+'.npy')
      return img_tensor, cap

    # Create training dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset_train = dataset_train.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset_train = dataset_train.shuffle(buffer_size).batch(batch_size)
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Create validation dataset
    dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))

    # Use map to load the numpy files in parallel
    dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset_val = dataset_val.shuffle(buffer_size).batch(batch_size)
    dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return (dataset_train, dataset_val, num_steps_train, num_steps_val)
