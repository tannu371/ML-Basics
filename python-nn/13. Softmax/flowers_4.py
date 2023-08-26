import flowersdata_2 as data
import math

weights = [0.1, 0.2], [0.15, 0.25], [0.18, 0.1]
biases = [0.3, 0.4, 0.35]
epochs = 1
learning_rate = .1

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p-m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
    return sum(losses)

# train the network
for epoch in range(epochs):
    pred = [[sum([w * i for w, i in zip(we, inp)]) + 
             bi for we, bi in zip(weights, biases)] for inp in data.inputs]
    act = [softmax(p) for p in pred]
    cost = sum([log_loss(ac, ta) for ac, ta in zip(act, data.targets)]) / len(act)
    print(f"ep:{epoch}, c:{cost:.4f}")

# THe log loss function for the softmax activation takes the activated neurons for one training sample.
# They are compared to the target and the sum of the errors is returned.
# Use log loss function to calculate the cost over all training samples. 
# Notice that the log loss function calculates the ;osses fro eacch neuron and returns their total. The cost is then calculated by summing the losses for all training samples and dividing them by the number of training samples.
# So there are two summing operations happening, but the result is only divided by number of trainingsamplles.
# Only repetition helps you to understand the extent of this information.
# Why is the maxx of all the predictions calculated?
# The short answer is to prevent overflow and underflow errors when predictions get very low or very high.

# So now that we have training targets, the softmax function and the cost, we can adjust the weighta and biases.
# TO do so we'll take the following steps:
# Calculate error derivatives 
# Calculate the gradient
# Update weights and biases

# Calculate the error derivatives for all neurons in all training targets.