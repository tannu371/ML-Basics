# Defining the slope. A common variable name for the slope is 'w'.
# The computer needs to calculate the slope for us, but we need to start with an arbitrary value.
w = 0.1

# Create a function that takes an input and returns slope times input.
def predict(i):
    return w*i

# You may not believe it, but this is already a neural network. I admit it won't be able to predict very well, but nonetheless we are well on our way.
# Let's make a prediction and feed the function with the value 2. 

print(predict(2))

# The target value for 2 should be 4.
# Let's execute the code.

# As expected the predictions is far off.