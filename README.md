# machine-learning-vae
Machine learning exercise

Variational auto-encoder to generate even digits as images using pytorch.

The main part of this project is 'main.py' . This script trains a variational auto-encoder to generate even digits as images. The script will read data from even_mnist.csv . This dataset is created from the MNIST dataset by fitlering out odd number. This leaves only 0, 2, 4, 6, and 8 to classify. The images are reduced from 28x28 to 14x14. Each row in the data file is a flattened 14x14 image and the correct label. The script also reads relevant parameters (e.g. learning rate, number of training epochs).

The script trains the network for a specified number of training epochs using the variational auto-encoder class 'vae_class.py' . The script will track the KL-distance. After training, the script will output a specified number of images containing simulated handwritten digits.

To run `main.py`, use
```
python main.py [-o result] [-n 100] [--param param.json] [-v]
```
You must specify the output folder with `-o` and the number of images to output with `-n`. Other arguments are optional.

The argument `-o` will determine the folder where the output images are saved. The script will create the folder if it does not already exist.

The argument `-n` will determine the number of sample images to be created after the network has finished training. Images will be saved in the output folder as 'i.pdf' from 0 to n. A plot of the training and test loss will also be saved here as 'loss.pdf'

To use parameters from a json file, add the optional argument `--param param.json`. Otherwise, the progam will use the default values for these parameters.

To run in verbose mode, add the optional argument `[-v]`. This will output train and test loss occasionally.

For help, use
```
python main.py -h
```
This will explain how to use the program and list default values of the parameters.

In addition to the main script, the variational auto-encoder is specified in the class 'vae_class.py' . This network consists of an encoder and a decoder. The encoder is made up of two convolutional layers, a fully connected layer, and two more fully connected layers for the mean and variance of the Gaussian distribution. The decoder consists of two fully connected layers and two deconvolutional layers. The user can specify the number of channels in the convolutional layers and the size of each layer. The user can also specify the size of the multi-dimensional space we wish to explore containing our input data.


