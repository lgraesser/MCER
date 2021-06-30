import gc
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import tensorflow as tf
import shutil

import utils

logger = logging.getLogger('evaluate')
logger.setLevel(logging.INFO)

ADVERTS_IMAGES = "data/eval_data/example"
ATTENTION_FEATURES_SHAPE = 64
NUM_REPEATS = 20
EVERY_K = 40000


def evaluate(image, encoder, decoder, image_features_extract_model, tokenizer, max_length):
    attention_plot = np.zeros((max_length, ATTENTION_FEATURES_SHAPE))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(utils.load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot, save_fig=True, show_fig=False, figname="test"):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(16, 16))

    len_result = len(result)
    num_subplots = max(max((len_result + 1) // 2, len_result // 2), 2)
    for li in range(len_result):
        temp_att = np.resize(attention_plot[li], (8, 8))
        ax = fig.add_subplot(num_subplots, num_subplots, li+1)
        ax.set_title(result[li])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()

    if save_fig:
        plt.savefig(figname)
    if show_fig:
        plt.show()

    # Matplotlib doesn't appear to release the memory. Problems.
    fig.clf()
    plt.close()
    del temp_image
    gc.collect()


def evaluate_and_plot(image_name,
                      encoder,
                      decoder,
                      image_features_extract_model,
                      tokenizer,
                      attention_fig_name,
                      caption_fname,
                      max_length,
                      save_fig=True,
                      show_fig=False):
    caption, attention_plot = evaluate(image_name, encoder, decoder, image_features_extract_model, tokenizer, max_length=max_length)
    plot_attention(image_name, caption, attention_plot, figname=attention_fig_name, save_fig=save_fig, show_fig=show_fig)
    logger.info(f'Saved attention plot to {attention_fig_name}')
    caption_str = ' '.join(caption)
    with open(caption_fname, 'w') as f:
        f.write(caption_str)
    logger.info(f'Saved caption: {caption_str}')


def eval_in_domain(encoder, decoder, tokenizer, image_features_extract_model, logdir, epoch, num_repeats=NUM_REPEATS, max_length=50):
    image_path, annotation_file_path = utils.get_data_path()
    # We don't want to shuffle so that we always get the same images
    captions, img_name_vector = utils.get_caption_image_names(annotation_file_path, image_path, shuffle_data=False)
    # Select the same few images in the dataset to eval
    eval_captions = [x for i, x in enumerate(captions) if i % EVERY_K == 0]
    eval_img_name_vector = [x for i, x in enumerate(img_name_vector) if i % EVERY_K == 0]
    logger.info(f'Evaluating in domain on {len(eval_captions)} examples')

    for i, (im_name, cap) in enumerate(zip(eval_img_name_vector, eval_captions)):
        # Each image gets its own directory
        dir_name = im_name.split('/')[-1]
        dir_name = re.sub('.jpg', '', dir_name).strip()
        # Add root dir
        dir_name = os.path.join(logdir, dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        logger.info(f'Generating eval for {im_name}, saving results to {dir_name}')
        # save real image and caption
        with open(os.path.join(dir_name, 'real_caption.txt'), 'w') as f:
            f.write(cap)
        print(f'Original caption: {cap}')
        shutil.copy(im_name, dir_name)

        # Generate predictions
        logger.info(f'Repeating eval {num_repeats} times.')
        for j in range(num_repeats):
            fig_name = os.path.join(dir_name, f'attention_plot_epoch:{epoch}_repeat:{j}.png')
            caption_name = os.path.join(dir_name, f'model_caption_epoch:{epoch}_repeat:{j}.txt')
            evaluate_and_plot(im_name, encoder, decoder, image_features_extract_model, tokenizer, fig_name, caption_name, max_length=max_length, save_fig=True)


def eval_adverts(encoder, decoder, tokenizer, image_features_extract_model, logdir, epoch, num_repeats=NUM_REPEATS, max_length=50, adverts_images=None):
    # Get list of image file names
    if adverts_images is None:
        adverts_images = ADVERTS_IMAGES
    image_file_names = [os.path.join(adverts_images, x) for x in os.listdir(adverts_images)]

    for i, image_file_name in enumerate(image_file_names):
        # Each image gets its own directory
        dir_name = image_file_name.split('/')[-1]
        dir_name = re.sub('.jpg', '', dir_name).strip()
        dir_name = os.path.join(logdir, dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        logger.info(f'Generating eval for {image_file_name}, saving results to {dir_name}')
        # save real image
        shutil.copy(image_file_name, dir_name)

        # Generate predictions
        for j in range(num_repeats):
            fig_name = os.path.join(dir_name, f'attention_plot_epoch:{epoch}_repeat:{j}.png')
            caption_name = os.path.join(dir_name, f'model_caption_epoch:{epoch}_repeat:{j}.txt')
            evaluate_and_plot(image_file_name, encoder, decoder, image_features_extract_model, tokenizer, fig_name, caption_name, max_length=max_length, save_fig=True)
