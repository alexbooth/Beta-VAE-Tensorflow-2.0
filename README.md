# Beta Variational Autoencoder in Tensorflow 2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Demo of a Beta-VAE with eager execution in TF2. 

## Usage
Begin training the model with ```train.py```  
```
--learning_rate    n   (optional) Float: learning rate
--epochs           n   (optional) Integer: number of passes over the dataset
--batch_size       n   (optional) Integer: mini-batch size during training
UNSUPPORTED --logdir          dir  (optional) String: log file directory
UNSUPPORTED --keep_training        (optional) loads the most recently saved weights and continues training
UNSUPPORTED --keep_best            (optional) save model only if it has the best training loss so far
--help
```
Track training by starting Tensorboard and then navigate to ```localhost:6006``` in browser
```
tensorboard --logdir ./tmp/log/
```

## References
Understanding disentangling in β-VAE (Burgess et al. 2018)  
https://arxiv.org/abs/1804.03599

From Autoencoder to Beta-VAE (Lilian Weng)  
https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html 

Auto-Encoding Variational Bayes (Kingma & Welling 2013)  
https://arxiv.org/abs/1312.6114
