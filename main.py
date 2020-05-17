
#import relevant modules
import json, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as func
from vae_class import VAE

torch.set_default_tensor_type('torch.DoubleTensor')

#command line arguments
parser = argparse.ArgumentParser(description='Homework 3',
                                 
                                 epilog='hyper parameters from file \n' +                       
                                        '(parameter) \t (type) \t (default) \t (description) \n' +
                                        'learning rate \t float \t\t 0.001 \t\t learning rate of optimizer \n' +
                                        'momentum \t float \t\t 0.95 \t\t momentum of optimizer \n' +
                                        'num epoch \t int \t\t 200 \t\t number of training epochs \n' +
                                        'display epoch \t int \t\t 10 \t\t how often to display \n' +
                                        'test size \t int \t\t 3000 \t\t size of test set \n'  +
                                        'in path \t string \t \"data/\"\t input path to data',
                                        
                                 formatter_class=argparse.RawTextHelpFormatter)

#parser arguments
parser.add_argument('-o', metavar='result',
                    help='path to results directory')
parser.add_argument('-n', metavar=100,
                    help='number of digits to write')
parser.add_argument('--param', metavar='param.json',
                    help='parameter file name')
parser.add_argument('-v',
                       '--verbose',
                       action='store_true',
                       help='verbosity (optional)')
args = parser.parse_args()

#read in hyper parameters
if args.param is not None:
    paramfile = open(args.param)
    param = json.load(paramfile)
    learning_rate = param['learning rate']
    momentum = param['momentum']
    num_epoch = param['num epoch']
    display_epoch = param['display epoch']
    test_size = param['test size']
    in_path = param['in path']
else: #if no param file, use defaults
    learning_rate = 0.001
    momentum = 0.95
    num_epoch = 200
    display_epoch = 10
    test_size = 3000
    in_path = 'data/'

#make results directory if it does not exist
if not os.path.exists(args.o):
    os.mkdir(args.o)

#set seed for testing
torch.manual_seed(1)

#read in data from csv file
csv_path = in_path + 'even_mnist.csv'
mnist_input = np.genfromtxt(csv_path, delimiter=' ')
xs = mnist_input[:,:-1] #extract 14x14 images
ys_column = mnist_input[:,-1] #extract digital values of images
ys = ys_column/2 #convert to indices

    
#training data set
x_train = torch.from_numpy( xs[test_size:].reshape(len(xs)-test_size,1,14,14) )
y_train = torch.from_numpy( ys[test_size:] )

#test data set
x_test = torch.from_numpy( xs[:test_size].reshape(test_size,1,14,14) )
y_test = torch.from_numpy( ys[:test_size] )

#initialize VAE model
model = VAE()

#define an optimizer and the loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
y_train = y_train.long()
y_test  = y_test.long()


train_vals=[]
test_vals=[]
for epoch in range(1, num_epoch + 1): #loop over epochs
        
    #train model and append to list
    train_val = model.backprop(x_train, optimizer)
    train_vals.append(train_val)
    
    #test model and append to list
    test_val = model.test(x_test)
    test_vals.append(test_val)

    #if verbose mode and if this is a display epoch
    if args.verbose and epoch % display_epoch == 0:
        #print loss
        print('epoch: '+str(epoch)+'/'+str(num_epoch)+\
                      '\tTraining Loss: '+'{:.4g}'.format(train_val)+\
                      '\tTest Loss: '+'{:.4g}'.format(test_val))

#print final loss
print('Final training loss: '+'{:.4f}'.format(train_vals[-1]))
print('Final test loss: '+'{:.4f}'.format(test_vals[-1]))


#create images from decoder
for i in range(int(args.n)):
    sample = model.decode(torch.randn(20))
    sample = sample.detach().numpy()
    sample = np.reshape(sample, (14,14))
    print(sample)
    plt.pcolormesh(sample, cmap='Greys')
    plt.savefig(args.o + '/' + str(i+1) + '.pdf', bbox_inches='tight')
    plt.clf()




'''
train_vals=[]
test_vals=[]
for epoch in range(1, num_epoch + 1): #loop over epochs
    
    #train model and append to list
    train_val = model.backprop(x_train, y_train, loss, optimizer)
    train_vals.append(train_val)
    
    #test model and append to list
    test_val = model.test(x_test, y_test, loss)
    test_vals.append(test_val)
    
    #if this is a display epoch
    if epoch % display_epoch == 0:
        
        #calculate number that are correct
        with torch.no_grad():
            correct = 0 #initialize sum
            for i in range(len(x_test)): #loop over test data
                output = model.forward(x_test[i].reshape(1,1,14,14))
                target = y_test[i].reshape(-1)
                correct_index = torch.exp(output).max(1)[1] #index of max weight
                correct += (target==correct_index).item() #true if agrees with target
    
        #print loss and correct percentage
        print('epoch: '+str(epoch)+'/'+str(num_epoch)+\
                      '\tTraining Loss: '+'{:.4f}'.format(train_val)+\
                      '\tTest Loss: '+'{:.4f}'.format(test_val)+\
                      '\tCorrect '+str(correct)+'/'+str(len(x_test))+' ('+\
                      '{:.2f}'.format(100.*correct/len(x_test))+'%)')

#print final loss
print('Final training loss: '+'{:.4f}'.format(train_vals[-1]))
print('Final test loss: '+'{:.4f}'.format(test_vals[-1]))


#output
out_array = np.array([train_vals, test_vals]).T
with open('output.csv', 'wb') as f:
    f.write(b'train,test\n')
    np.savetxt(f, out_array, delimiter=',')
'''

