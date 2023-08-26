# Now we will have a look at what impact random weights have on the network.
# Until now, we have always typed in weights and biases manually.
# We choose arbitrary values, usually somewhere between minus 0.5 and plus 0.5.

# The activation of a neuron is the weighted sum of inputs. In the last example we saw that this can lead to negative predictions and a deactivated neuron.
# When the back propagation is not capable of reviving a neuron, we speak of a dead neuron.
# There are different ways to deal with this problem.
# First of all, we can increase the amount of neurons. As long as enough neurons survive the network is still capable of making good predictions.
# A second option is to replace ReLU with leaky ReLU. Leaky ReLU allows for a small change when the prediction is negative.
# A third option is to choose the weights and biases in a way that it is unlikely that the predictions are negative. 

# But how do we choose good random weights and biases?
# There are different strategies to initialize the weights and biases.
# One of them is to choose the values from a normal distribution. What this means is that the range of values is between minus three and three, and most values will be around zero.
# But there is a much simpler strategy. We will use Python's random function to choose a value between minus 0.5 and 0.5. It's not perfect, but it will be good enough for our network.
# And what about bias? We will set it to zero.

# We will delete the code that indicates the bad prediction.
