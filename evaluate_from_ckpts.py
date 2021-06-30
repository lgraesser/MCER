import json
import logging
import os
import tensorflow as tf

import evaluate
import model
import preprocessing
import train
import utils

logger = logging.getLogger('eval_from_ckpt')
logger.setLevel(logging.INFO)

str_1 = "_1294"
str_2 = "_2588"
str_3 = "_3882"
str_4 = ""
_CKPT_TO_EPOCH_BATCH_414 = {
    1: -1,
    2: str(0) + str_1,
    3: str(0) + str_2,
    4: str(0) + str_3,
    5: str(0) + str_4,
    6: str(1) + str_1,
    7: str(1) + str_2,
    8: str(1) + str_3,
    9: str(1) + str_4,
    10: str(2) + str_1,
    11: str(2) + str_2,
    12: str(2) + str_3,
    13: str(2) + str_4,
    14: str(3) + str_1,
    15: str(3) + str_2,
    16: str(3) + str_3,
    17: str(3) + str_4,
    18: str(4) + str_1,
    19: str(4) + str_2,
    20: str(4) + str_3,
    21: str(4) + str_4,
    22: str(5) + str_1,
    23: str(5) + str_2,
    24: str(5) + str_3,
    25: str(5) + str_4,
    26: str(6) + str_1,
    27: str(6) + str_2,
    28: str(6) + str_3,
    29: str(6) + str_4,
}
_EPOCH_BATCH_414_CKPT = {v: k for k, v in _CKPT_TO_EPOCH_BATCH_414.items()}


def map_ckpts_to_partial_epoch(ckpt_num, map_type="414k"):

    if map_type == "414k":
        epoch_map = _CKPT_TO_EPOCH_BATCH_414
    else:
        raise ValueError(f"{map_type} not implemented yet.")

    logger.info(f"Mapping {ckpt_num} to {epoch_map[ckpt_num]}")
    return epoch_map[ckpt_num]


def map_partial_epoch_to_ckpts(epoch_num=4, batch_num="", map_type="414k"):

    if map_type == "414k":
        epoch_map = _EPOCH_BATCH_414_CKPT
    else:
        raise ValueError(f"{map_type} not implemented yet.")

    logger.info(f"Mapping epoch {epoch_num}, batch {batch_num} to {epoch_map[str(epoch_num) + batch_num]}")
    return epoch_map[str(epoch_num) + batch_num]


def evaluate_from_ckpts(num_training_captions=30000,
                        vocab_size=5000,
                        experiment_name="test",
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
    results_logdir = os.path.join("results_eval_only", experiment_name)
    eval_loss_fig_name = os.path.join(results_logdir, 'eval_loss_plot.png')
    eval_loss_partial_fig_name = os.path.join(results_logdir, 'eval_loss_plot_partial_eval.png')
    results_eval_logdir = os.path.join(results_logdir, 'eval_results')

    # We just eval the example directory right now. To eval more add the corresponding directories here.
    results_eval_adverts_logdir = os.path.join(results_eval_logdir, 'adverts')
    results_eval_example_logdir = os.path.join(results_eval_adverts_logdir, 'example')
    results_advert_dirs = [results_eval_example_logdir]
    source_images_adverts_dirs = ["data/eval_data/example"]

    if not os.path.exists(model_logdir):
        os.makedirs(model_logdir)
        logger.info(f'Created the model logging folder {model_logdir}.')
    else:
        logger.warning(f'Model directory already exists, continue?')
        user_input = input('Y to continue. Any other key to exit: ')
        if user_input.lower() != 'y':
            sys.exit(0)

    if not os.path.exists(results_logdir):
        os.makedirs(results_eval_necklace_logdir)
        logger.info(f'Created the results folders with root {results_logdir}.')
    else:
        logger.warning(f'Results directory already exists, continue?')
        user_input = input('Y to continue. Any other key to exit: ')
        if user_input.lower() != 'y':
            sys.exit(0)

    if save_eval_dataset:
        dataset_path = os.path.join(model_logdir, 'eval_dataset')
        tf.data.experimental.save(dataset_val, dataset_path)

    # ******** Build models ********
    # We add one more to the vocab to account for '<pad>' token
    vocab_size += 1
    encoder, decoder, tokenizer, ckpt_manager, ckpt = utils.build_model(model_logdir, vocab_size)

    # Eval on a random image to ensure all model elements are built.
    random_image = tf.keras.applications.inception_v3.preprocess_input(tf.random.uniform(shape=(299, 299, 3)))
    evaluate.evaluate(random_image, encoder, decoder, image_features_extract_model, tokenizer, max_length)

    # We re-run the eval to check the loss is comparable to the original model
    eval_loss_plot = []
    # Loop through all the checkpoints and evaluate from ckpts
    for current_ckpt in ckpt_manager.checkpoints:
        ckpt.restore(current_ckpt)
        logger.info(f'Restored checkpoint {current_ckpt}')
        ckpt_num = int(current_ckpt.split('-')[-1])
        epoch = map_ckpts_to_partial_epoch(ckpt_num)

        logger.info(f'Evaluating model at epoch {epoch}')
        eval_loss = train.eval(epoch, dataset_val, num_steps_val, encoder, decoder, tokenizer, optimizer, loss_object)
        eval_loss_plot.append(eval_loss)
        utils.save_loss_plot(eval_loss_plot, eval_loss_fig_name, 'eval')
        logger.info(f'Eval loss: {eval_loss_plot}')

        for image_set, results_ad_logdir in zip(source_images_adverts_dirs, results_advert_dirs):
            evaluate.eval_adverts(encoder, decoder, tokenizer, image_features_extract_model, results_ad_logdir, epoch, max_length=max_length, adverts_images=image_set, num_repeats=num_repeats)
