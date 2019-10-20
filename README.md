# Beta Variational Autoencoder in Tensorflow 2.0

Demo of a Beta-VAE with eager execution in TF2. 

## Usage
Begin training the model with ```train.py```  
```
--learning_rate    n   (optional) Float: learning rate
--epochs           n   (optional) Integer: number of passes over the dataset
--batch_size       n   (optional) Integer: mini-batch size during training
UNSUPPORTED --logdir          dir  (optional) String: log file directory
UNSUPPORTED --keep_training        (optional) loads the most recently saved weights and continues training
UNSUPPORTED --keep_best            (optional) save model only if it has the best loss so far
UNSUPPORTED --use_noise       n    (optional) adds salt and pepper noise to input features with a probability of n percent
--help
```
Track training by starting Tensorboard and then navigate to ```localhost:6006``` in browser
```
tensorboard --logdir ./tmp/log/
```
Sample the training model with ```sample.py```  
Note: Do not run sample.py and train.py at the same time, tensorflow will crash.
```
UNSUPPORTED --sample_size     n    (optional) Integer: number of samples to test
UNSUPPORTED --model       filepath (required) String: path to a trained model (.h5 file)
UNSUPPORTED --use_noise       n    (optional) adds salt and pepper noise to input features with a probability of n percent
--help
```

## References
Understanding disentangling in β-VAE (Burgess et al.)  
https://arxiv.org/pdf/1804.03599.pdf 

From Autoencoder to Beta-VAE (Lilian Weng)  
https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html 
