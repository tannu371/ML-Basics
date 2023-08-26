# I have learned how to add a hidden layer to the network and why this helps when dealing with non-linear data.
# By now, you have seen a network that can find the relationship between X and Y values.
# Can predict car maintenance costs.
# Can predict whether to keep or sell cars.
# And finally was able to predict flower colors in linear and nonlinear data.

# Look at all the steps required to train a neural network with hidden layers.
# Feed Forward:
    # Calculate the predictions for the hidden layer
    # Activate the hidden layer with ReLU
    # Calculate the predictions for the output layer
    # Activate the output layer with Softmax
# Cost: Calculate the cost over all the training data.
# Back Propagation:
    # Calculate the error derivatives for the output layer
    # Calculate the gradient for the weights and biases from hidden to output
    # Calculate the error derivatives for the hidden layer
    # Calculate the gradient for the weights and biases from input to hidden
    # Update the weights and bias

# It is amazing that just a few lines of code are capable of learning from very complex data.
# And if we would use a library like Numpy, we can even shrink this code to half its size.
# But the goal here is not to write compact code or fast code.
# The goal is to learn how a neural networks works.
# And with this plain Python code, it is also very easy to port code to another language.
# There's no reson that the code should not run on a Commodore 64.

# You have seen all building blocks now. From here we only need to make small variations to get to the ultimate goal.
# Recognizing handwritten numbers.

# There might be a question on your mind after learning about hidden layers.
# For example, how many hidden layers do we need?
# The answer to this question is just try it out. Different problems ask for different solutions.
# You can see the amount of hidden layers as an extra parameter in the network.
# Notice that adding extra hidden layers will add extra calculation time.
# Another question could be : How many neurons does a hidden layer have? 4, 8, 80?
# Again, find a good balance between extra training time and the qualiy of the predictions.

# What happens when we let Python pick random values for the weights and biases?