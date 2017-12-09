# Tensorflow training

Repo created to practice with tensorflow, first problemas are based on [openwebminars course](https://openwebinars.net/cursos/machine-learning-tensorflow/).

## Installing tensorflow from source

In this case I need it compiled for **python 3.6** with cpu **sse4.1 / sse4.2** instructions.

See: <https://github.com/tensorflow/tensorflow/issues/8037>

- Follow [installing from sources guide](https://www.tensorflow.org/install/install_sources)
- Use `-mavx -msse4.1 -msse4.2` when propmted during configuration process

Already compiled version in [tensorflow-builds](https://github.com/sigilioso/tensorflow-build/raw/master/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl) repo.

The compiled version mentioned above is used in the [Pipfile](./Pipfile).


## DITS-classification problem

Classification of traffic signals problem.

- In order to prepare data, download and unzip it from <http://www.dis.uniroma1.it/~bloisi/ds/dits.html>


### TODO

- Auto-adjust learning rate
- Commands to train / make predictions
