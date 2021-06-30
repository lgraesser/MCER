# Model Caption Epoch Repeat

This codebase is for training models which can caption images.

The code was used to train MCER, a model which wrote the text in Michael Iveson's artwork *model_caption_epoch_repeat* (2020-ongoing).

## Setup

```
git clone https://github.com/lgraesser/MCER.git
cd MCER
conda env create -f environment.yml
conda activate imcap
```

## Training a model

```
python main.py --experiment_name 'test-model' --num_training_captions 10000 --vocab_size 5000 --num_epochs 7 --num_repeats 20 --partial_epoch_eval True
```

> **_NOTE:_**: This will download and extract the entire MSCOCO dataset which requires >15GB space.

## Training MCER

The command below trains a model with the same settings as MCER.

```
python main.py --experiment_name 'test-mcer-model' --num_training_captions 414113 --vocab_size 20000 --num_epochs 7 --num_repeats 20 --partial_epoch_eval True
```


## Training a model inside a Docker container

```
# This starts a container with a GPU and mounts the repository root directory so that data, checkpoints, and results will be available automatically outside the container.
docker run -u 1000:1000 --mount type=bind,source="$(pwd)",target=/image-captioning --rm --gpus all -it imcap:v1.0.0

# Train a model.
python main.py --experiment_name 'test-model' --num_training_captions 10000 --vocab_size 5000 --num_epochs 7 --num_repeats 20 --partial_epoch_eval True
```

> **_NOTE:_** This will download and extract the entire MSCOCO dataset which requires >15GB space.


## Acknowledgements

Thank you to the [Tensorflow (Abadi et al., 2015)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf) team, the authors of [Show, Attend And Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044), and the creators of the [MSCOCO dataset (Lin et al., 2014)](https://arxiv.org/abs/1405.0312), without which this would not have been possible.

The code is based on the [Tensorflow image captioning tutorial](https://www.tensorflow.org/tutorials/text/image_captioning), however it involves substantial changes to support model training on a remote server, and more extensive evaluation and captioning options. The original tutorial is also included with minor modifications.

The model architecture is similar to [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).
