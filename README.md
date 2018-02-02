# TensorFlow AlphaZero Connect Four 

A simple implementation of DeepMind's [AlphaZero](https://arxiv.org/abs/1712.01815)/[AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) for Connect Four. Uses Tensorflow.

To generate training data (master process, required): `python main.py`
To generate training data in additional processes (optional): `python main.py --assist`
To train: `python main.py --train`
To play a game against the model: `python main.py --play`

After a few hours of training on a GTX 1060, it learns to play the center column of the board.