from absl import app
from absl import flags
import logging
import evaluate_from_ckpts
import train

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 7, 'How many epochs to train for.')
flags.DEFINE_integer('num_training_captions', 30000, 'How many training captions to use.')
flags.DEFINE_integer('vocab_size', 5000, 'Size of the vocab.')
flags.DEFINE_string('experiment_name', 'test', 'Name of the experiment.')
flags.DEFINE_boolean('eval_in_domain', True, 'Whether to run in domain eval.')
flags.DEFINE_boolean('eval_adverts', True, 'Whether to run adverts eval.')
flags.DEFINE_boolean('partial_epoch_eval', False, 'Whether to run evals part the way through the epoch.')
flags.DEFINE_integer('num_repeats', 20, 'How many times to caption the same image during eval.')
flags.DEFINE_boolean('eval_only', False, 'Whether to just run an evaluation on existing checkpoints.')
flags.DEFINE_boolean('save_eval_dataset', False, 'Whether to save the eval dataset for sense checking models later.')


def main(_):
    logging.basicConfig(filename='log.log', level=logging.INFO)
    if FLAGS.eval_only:
        logging.info(f'Evaluating an existing model ckpted here: models/{FLAGS.experiment_name}/checkpoints.')
        evaluate_from_ckpts.evaluate_from_ckpts(
        num_training_captions=FLAGS.num_training_captions,
        vocab_size=FLAGS.vocab_size,
        experiment_name=FLAGS.experiment_name,
        save_eval_dataset=FLAGS.save_eval_dataset)
    else:
        logging.info(f'Training a new model and ckpting it here: models/{FLAGS.experiment_name}/checkpoints.')
        train.train(num_epochs=FLAGS.num_epochs,
                    num_training_captions=FLAGS.num_training_captions,
                    vocab_size=FLAGS.vocab_size,
                    experiment_name=FLAGS.experiment_name,
                    eval_in_domain=FLAGS.eval_in_domain,
                    eval_adverts=FLAGS.eval_adverts,
                    partial_epoch_eval=FLAGS.partial_epoch_eval,
                    num_repeats=FLAGS.num_repeats,
                    save_eval_dataset=FLAGS.save_eval_dataset)


if __name__ == "__main__":
    app.run(main)
