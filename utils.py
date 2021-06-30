import json
import logging
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.utils import shuffle

import model
import train

logger = logging.getLogger('utils')
logger.setLevel(logging.INFO)


def get_data_path():
    '''Returns the path to the image and annotation data.
    Downloads the data if it doesn't exist.
    '''
    # Download caption annotation files
    annotation_folder = '/data/train_data/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        logger.info('Downloading captions file.')
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                cache_subdir=os.path.abspath('./data/train_data'),
                                                origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                extract = True)
        annotation_file_path = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
        os.remove(annotation_zip)
    else:
        annotation_file_path = os.path.abspath('.') + annotation_folder + 'captions_train2014.json'
        logger.info(f'Captions file already exists here {annotation_file_path}.')

    # Download image files
    image_folder = '/data/train_data/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        logger.info('Downloading image data. This may take a while.')
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('./data/train_data'),
                                            origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                            extract = True)
        image_file_path = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        image_file_path = os.path.abspath('.') + image_folder
        logger.info(f'Image data already exists here {image_file_path}.')

    return image_file_path, annotation_file_path


def get_caption_image_names(annotation_file_path, image_file_path, shuffle_data=True):
    '''Returns a shuffled list of the captions and the corresponding image names.'''

    # Read the json file
    with open(annotation_file_path, 'r') as f:
        annotations = json.load(f)
    logger.info('Loaded the annotations file.')

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = image_file_path + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    if shuffle_data:
        logger.info('Shuffling the data...')
        train_captions, img_name_vector = shuffle(all_captions,
                                                  all_img_name_vector,
                                                  random_state=1)
    else:
        train_captions = all_captions
        img_name_vector = all_img_name_vector

    return train_captions, img_name_vector


def get_top_k(train_captions, img_name_vector, num_examples):
    '''Selects the first k examples from the data.'''
    assert len(train_captions) == len(img_name_vector)
    original_cap_length = len(train_captions)
    if num_examples > original_cap_length:
        logger.warning(f'Desired num examples {num_examples} > actual number examples {original_cap_length}, using whole training set')
        num_examples = original_cap_length

    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]
    logger.info(f'Num train captions: {len(train_captions)}, num all captions: {original_cap_length}')

    return train_captions, img_name_vector


def calc_max_length(tensor):
    """Find the maximum length of any tensor"""
    return max(len(t) for t in tensor)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def plot_loss(loss_data):
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


def save_loss_plot(loss_data, figname, data_label):
    plt.figure(figsize=(10, 10))
    plt.plot(loss_data, label=data_label)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend(loc='upper left')
    plt.savefig(figname)
    plt.close()


def build_model(model_logdir, vocab_size):
    embedding_dim = 256
    units = 512
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    encoder = model.CNN_Encoder(embedding_dim)
    decoder = model.RNN_Decoder(embedding_dim, units, vocab_size)
    # get optim, and checkpoint manager
    optimizer = train.get_optimizer()
    loss_object = train.get_loss_object()
    ckpt_manager, ckpt = train.get_checkpoint_manager(encoder, decoder, optimizer, path=model_logdir)

    # Restore tokenizer
    with open(os.path.join(model_logdir, 'tokenizer.json')) as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

    return encoder, decoder, tokenizer, ckpt_manager, ckpt
