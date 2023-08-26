# And now it's time to take all the puzzle pieces and put them together for a final example.
# A neural network to recognize handwritten numbers, written in plain Pythom code.

# The first thing to do is specify the numbers of neurons and layers.
# And here it is. 
# The input layer has 784 neurons that represent 28 by 28 pixels of the images.
# The hidden layer has an arbitrary number of 20 neurons.
# Perhaps I need to change this later, but we'll come to that if the network is not performing well.
# The 10 output neurons represent the digits zero to nine.

# Let's start by defining the network in code.

import mnistreader_4 as reader
import random

epochs = 3
batch_size = 400 # I will set the batch size to 400. I choose to batch size arbitrarily, just like the following learning rate.
learning_rate =0.4

# Now I create the dimensions for the network accordingly to the diagram.
input_count, hidden_count, output_count = 784, 20, 10

# Now I can initialize the weights and biases.
w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]

# The weights and biases are initialized.
# Let's inspect their dimensions.

print(len(w_i_h)) # should be 20
print(len(w_i_h[0])) # should be 784
print(len(w_h_o)) # should be 10
print(len(w_h_o[0])) # should be 20
print(len(b_i_h)) # should be 20
print(len(b_h_o)) # should be 10

# That seems to work.
# Now I will create the softmax function.


