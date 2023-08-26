import flowersdata_1 as data

weights = [0.1, 0.2], [0.15, 0.25], [0.18, 0.1]
biases = [0.3, 0.4, 0.35]
epochs = 1
learning_rate = .1

# train the network
for epoch in range(epochs):
    pred = [[sum([w * i for w, i in zip(we, inp)]) + 
             bi for we, bi in zip(weights, biases)] for inp in data.inputs]
    print(pred)
    print("row count:", len(pred))
    print("column count:", len(pred[0]))

# What can we do witj this matrix?
# The next step is to take the predictions and turn them into a category. For this, we are going to use the softmax activation function.
# The softmax activation function takes the three predictions and creates a probability distribution.
# Notice that the values in the distribution add up to a total of one.
# At this point, the most active neuron determines the category. 
# To train the network we first need to know the cost of the network. But how do we compare the result of the softmax function to the target? 
# What is needed is to compare all the values from the probability to a target. In order to do that, the targets also need to be defined as a list of numbers.
# Only one of the values in this list is one, and all the others are zero. This is called one hot encoding.
# So, now we have a probability and a known target, these can be compared to calculate the cost.
# With one hot encoding, the colors are converted and turned into what we might call certainties. Now the probabilities and the certainities can be compared and used to calculate the cost.