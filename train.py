import io
import json
import logging
import math
import os
import sys
import tensorflow as tf
import time

import evaluate
import model
import preprocessing
import utils

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
_SOURCE_IMAGES_ADVERTS_DIRS = ["data/eval_data/example"]


def get_optimizer():
    return tf.keras.optimizers.Adam()


def get_loss_object():
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_checkpoint_manager(encoder, decoder, optimizer, path="."):
    path = path + '/checkpoints/train'
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    # We keep all the checkpoints
    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=None)
    return ckpt_manager, ckpt


@tf.function
def train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, loss_object):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions, loss_object)
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


@tf.function
def eval_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, loss_object):
    """Same as the train step but with no gradients or parameter updates"""
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    features = encoder(img_tensor)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        loss += loss_function(target[:, i], predictions, loss_object)
        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    return loss, total_loss


def eval(epoch, dataset_val, num_steps_val, encoder, decoder, tokenizer, optimizer, loss_object, batch=None):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset_val):
        batch_loss, t_loss = eval_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, loss_object)
        total_loss += t_loss

    logger.info(f'Eval: Epoch {epoch} Batch: {batch} Loss {total_loss / num_steps_val}')
    logger.info(f'Eval: Time taken for 1 epoch {time.time() - start} sec')
    return total_loss / num_steps_val


def train(num_epochs=20,
          num_training_captions=30000,
          vocab_size=5000,
          experiment_name="test",
          eval_in_domain=True,
          eval_adverts=True,
          src_adverts_dirs=None,
          partial_epoch_eval=False,
          num_repeats=20,
          save_eval_dataset=False):

    image_path, annotation_file_path = utils.get_data_path()
    logger.info('Downloaded the dataset.')

    train_captions, img_name_vector = utils.get_caption_image_names(annotation_file_path, image_path)
    logger.info('Extracted the caption and image names.')

    train_captions, img_name_vector = utils.get_top_k(train_captions, img_name_vector, num_training_captions)
    assert len(train_captions) == len(img_name_vector) == num_training_captions
    logger.info(f'Selected {num_training_captions} examples.')

    image_features_extract_model = model.create_im_feat_extract_model()
    preprocessing.preprocess_images(img_name_vector, image_features_extract_model)
    logger.info('Preprocessed the images.')

    caption_vector, tokenizer, train_seqs = preprocessing.preprocess_text(train_captions, vocab_size)
    max_length = utils.calc_max_length(train_seqs)
    logger.info('Preprocessed the text.')

    (dataset_train, dataset_val,
     num_steps_train, num_steps_val) = preprocessing.create_dataset(img_name_vector,
                                                                    caption_vector,
                                                                    train_seqs,
                                                                    test_size=0.2,
                                                                    batch_size=64,
                                                                    buffer_size=1000)
    logger.info('Created the dataset.')

    # Create data folders
    model_logdir = os.path.join("models", experiment_name)
    results_logdir = os.path.join("results", experiment_name)
    train_loss_fig_name = os.path.join(results_logdir, 'train_loss_plot.png')
    train_loss_partial_fig_name = os.path.join(results_logdir, 'train_loss_plot_partial_eval.png')
    eval_loss_fig_name = os.path.join(results_logdir, 'eval_loss_plot.png')
    eval_loss_partial_fig_name = os.path.join(results_logdir, 'eval_loss_plot_partial_eval.png')
    results_eval_logdir = os.path.join(results_logdir, 'eval_results')
    results_eval_in_domain_logdir = os.path.join(results_eval_logdir, 'in_domain')
    results_eval_adverts_logdir = os.path.join(results_eval_logdir, 'adverts')
    results_eval_example_logdir = os.path.join(results_eval_adverts_logdir, 'example')
    results_advert_dirs = [results_eval_example_logdir]
    source_images_adverts_dirs = src_adverts_dirs
    if source_images_adverts_dirs is None:
        source_images_adverts_dirs = _SOURCE_IMAGES_ADVERTS_DIRS

    if not os.path.exists(model_logdir):
        os.makedirs(model_logdir)
        logger.info(f'Created the model logging folder {model_logdir}.')
    else:
        logger.warning(f'Model directory already exists, continue?')
        user_input = input('Y to continue. Any other key to exit: ')
        if user_input.lower() != 'y':
            sys.exit(0)

    if save_eval_dataset:
        dataset_path = os.path.join(model_logdir, 'eval_dataset')
        tf.data.experimental.save(dataset_val, dataset_path)

    if not os.path.exists(results_logdir):
        os.makedirs(results_logdir)
        os.makedirs(results_eval_in_domain_logdir)
        os.makedirs(results_eval_adverts_logdir)
        os.makedirs(results_eval_book1_logdir)
        logger.info(f'Created the results folders with root {results_logdir}.')
    else:
        logger.warning(f'Results directory already exists, continue?')
        user_input = input('Y to continue. Any other key to exit: ')
        if user_input.lower() != 'y':
            sys.exit(0)

    # Log relevant training values
    with open(os.path.join(results_logdir, 'EPOCHS.txt'), 'w') as f:
        f.write(f'Num epochs: {num_epochs}')
    with open(os.path.join(results_logdir, 'NUM_TRAIN_EGS.txt'), 'w') as f:
        f.write(f'Num train captions: {len(train_captions)}')
    with open(os.path.join(results_logdir, 'VOCAB_SIZE.txt'), 'w') as f:
        f.write(f'Vocab size: {vocab_size}')

    # ******** Build models ********
    # We add one more to the vocab to account for '<pad>' token
    vocab_size += 1
    embedding_dim = 256
    units = 512
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    encoder = model.CNN_Encoder(embedding_dim)
    decoder = model.RNN_Decoder(embedding_dim, units, vocab_size)
    # get optim, loss objectm and checkpoint manager
    optimizer = get_optimizer()
    loss_object = get_loss_object()
    ckpt_manager, ckpt = get_checkpoint_manager(encoder, decoder, optimizer, path=model_logdir)

    # *********** Train model ***********
    # Used for partial eval
    eval_every_k_batch = int(num_steps_train / 4.0)
    start_epoch = 0
    train_loss_plot = []
    eval_loss_plot = []
    train_loss_plot_partial_eval = []
    eval_loss_plot_partial_eval = []

    # restore last checkpoint
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logger.info(f'Restored latest checkpoint {ckpt_manager.latest_checkpoint}')
        logger.fatal(f'The training data is likely different which means the tokenizer is different. Exiting as the least confusing / error prone option for now.')

    # Eval and save random model
    logger.info(f'Saving model at epoch {start_epoch} to {model_logdir}/checkpoints/train.')
    ckpt_manager.save()
    # Also save the tokenizer since the vocab is specific to the training data
    tokenizer_json = tokenizer.to_json()
    with io.open(os.path.join(model_logdir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    logger.info(f'Evaluating model before training at epoch {start_epoch - 1}')
    eval_loss = eval(start_epoch - 1, dataset_val, num_steps_val, encoder, decoder, tokenizer, optimizer, loss_object)
    eval_loss_plot.append(eval_loss)
    eval_loss_plot_partial_eval.append(eval_loss)
    utils.save_loss_plot(eval_loss_plot, eval_loss_fig_name, 'eval')
    utils.save_loss_plot(eval_loss_plot_partial_eval, eval_loss_partial_fig_name, 'eval_ptl_eval')

    if eval_in_domain:
        evaluate.eval_in_domain(encoder, decoder, tokenizer, image_features_extract_model, results_eval_in_domain_logdir, start_epoch - 1, max_length=max_length, num_repeats=num_repeats)
    if eval_adverts:
        for image_set, results_ad_logdir in zip(source_images_adverts_dirs, results_advert_dirs):
            evaluate.eval_adverts(encoder, decoder, tokenizer, image_features_extract_model, results_ad_logdir, start_epoch - 1, max_length=max_length, adverts_images=image_set, num_repeats=num_repeats)

    logger.info('Starting training...')
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset_train):
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, loss_object)
            total_loss += t_loss

            if batch % 50 == 0:
                logger.info(f'Epoch {epoch} Batch {batch} Loss {batch_loss.numpy() / int(target.shape[1])}')

            if partial_epoch_eval:
                # Partition the epoch so there are roughly 4 evaluations per epoch
                if batch in [eval_every_k_batch, eval_every_k_batch * 2, eval_every_k_batch * 3]:
                    logger.info(f'Evaluating at partial epoch, Epoch {epoch}, Batch {batch}')
                    eval_loss = eval(epoch, dataset_val, num_steps_val, encoder, decoder, tokenizer, optimizer, loss_object, batch=batch)
                    eval_loss_plot_partial_eval.append(eval_loss)
                    logger.info(f'Partial eval loss: {eval_loss_plot_partial_eval}')
                    utils.save_loss_plot(eval_loss_plot_partial_eval, eval_loss_partial_fig_name, 'eval_ptl_eval')

                    partial_epoch = str(epoch) + '_' + str(batch)
                    if eval_in_domain:
                        evaluate.eval_in_domain(encoder, decoder, tokenizer, image_features_extract_model, results_eval_in_domain_logdir, partial_epoch, max_length=max_length, num_repeats=num_repeats)
                    if eval_adverts:
                        for image_set, results_ad_logdir in zip(source_images_adverts_dirs, results_advert_dirs):
                            evaluate.eval_adverts(encoder, decoder, tokenizer, image_features_extract_model, results_ad_logdir, partial_epoch, max_length=max_length, adverts_images=image_set, num_repeats=num_repeats)

                    # storing the current average loss value to plot later
                    train_loss_plot_partial_eval.append(total_loss / batch)
                    logger.info(f'Partial train loss: {train_loss_plot_partial_eval}')
                    utils.save_loss_plot(train_loss_plot_partial_eval, train_loss_partial_fig_name, 'train_ptl_eval')

                    # Save the partial epoch. This means the total epoch count will be (num_epochs + 1) * 4
                    logger.info(f'Partial epoch cktpt: Saving model at epoch {start_epoch} batch {batch} to {model_logdir}/checkpoints/train.')
                    ckpt_manager.save()

        # storing the epoch end loss value to plot later
        train_loss_plot.append(total_loss / num_steps_train)
        train_loss_plot_partial_eval.append(total_loss / num_steps_train)
        utils.save_loss_plot(train_loss_plot, train_loss_fig_name, 'train')
        utils.save_loss_plot(train_loss_plot_partial_eval, train_loss_partial_fig_name, 'train_ptl_eval')
        logger.info(f'Train loss: {train_loss_plot}')
        logger.info(f'Partial train loss: {train_loss_plot_partial_eval}')

        # Save every epoch
        logger.info(f'Saving model at epoch {start_epoch} to {model_logdir}/checkpoints/train.')
        ckpt_manager.save()

        logger.info(f'Train: Epoch {epoch} Loss {total_loss / num_steps_train}')
        logger.info(f'Train: Time taken for 1 epoch {time.time() - start} sec')

        logger.info(f'Evaluating model at epoch {epoch}')
        eval_loss = eval(epoch, dataset_val, num_steps_val, encoder, decoder, tokenizer, optimizer, loss_object)
        eval_loss_plot.append(eval_loss)
        eval_loss_plot_partial_eval.append(eval_loss)
        utils.save_loss_plot(eval_loss_plot, eval_loss_fig_name, 'eval')
        utils.save_loss_plot(eval_loss_plot_partial_eval, eval_loss_partial_fig_name, 'eval_ptl_eval')
        logger.info(f'Eval loss: {eval_loss_plot}')
        logger.info(f'Partial eval loss: {eval_loss_plot_partial_eval}')

        if eval_in_domain:
            evaluate.eval_in_domain(encoder, decoder, tokenizer, image_features_extract_model, results_eval_in_domain_logdir, epoch, max_length=max_length, num_repeats=num_repeats)
        if eval_adverts:
            for image_set, results_ad_logdir in zip(source_images_adverts_dirs, results_advert_dirs):
                evaluate.eval_adverts(encoder, decoder, tokenizer, image_features_extract_model, results_ad_logdir, epoch, max_length=max_length, adverts_images=image_set, num_repeats=num_repeats)
