# drnn

Code for paper: ["Tree structured-decodin with doubly-recurrent neural networks"](http://people.csail.mit.edu/davidam/docs/ICLR2017_Tree.pdf)


### Dependencies


#### Software

* Torch7 (`luarocks install ...`)
* penlight
* nn
* nngraph
* optim
* rnn
* OpenNMT (only needed for MT task)


#### Data

Only data for synthetic task is provided. MT and IFTTT have to be downloaded separately.

### Basic Usage

./demo_vanilla.sh [TASK] (where task is one of synth, BABYMT, IFTTT)

Trains, evaluates and saves predictions.
