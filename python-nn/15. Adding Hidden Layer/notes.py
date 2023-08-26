# The network has problems classifying the data when the regression lines are curved.
# The problem is that the network, as we created it, cannot handle nonlinear data.
# We need to introduce nonlinearity by adding an extra layer.
# Where does the extra layer go?

# A network always has an input layer and an output layer. 
# Any additional layer comes in between. We call these added layers hidden layers.

# The flower network has two inputs and three outputs. But why are there four neurons in the hidden layer?
# The answer is that the number of neurons in the hidden layer are chosen arbitrarily. We'll find out later whether this number is appropriate.
# Notice there are now two weight matrices. One weight matrix from input to a hidden. One weight matrix from hidden to output.

# The question is what does the hidden layer do?
# Some explain hidden layers by saying... it can pick up certain patterns in the data, just like our brain picks up patterns from the overwhelming visual input it gets from our eyes. 
# A human brain is a master at recognizing patterns like shapes, colors and movement.
# When the brain sees a certain shape or a certain color, it does not need to analyze each pixel. It recognizes in an instant.
# We want to achieve some kind of pattern recognition in our network, too. That's why hidden layers are sometimes compared to the pattern recognition that happens in our brains.

# Is this comparision accurate?

# When data is non-linear, hidden layers allow us to divide the data by curved lines.
# So how do we achieve a divison by curved lines? 
# Adding a hidden layer is not the whole story. Adding a hidden layer would just amplify activations from previous layers.
# The calculation would stay linear. We need to apply some nonlinearity to the hidden layer.

# Activation functions add non linearity to a network. Which ones have you seen?
# Linear activation function, sigmois activation function and softmax activation function.
# Linear activation does not help us here. If the hidden layer would just perform a weight times inputs, plus bias, it would just amplify previous activations but stay linear.
# We use the sigmoid for binary classification. This could be a candidate to create nonlinearity in the hidden layer.
# The softmax is used for multiclass classification and is not suitable for hidden layers.

# So shall we choose the sigmoid? No, although the sigmoid was very popular it fell out of favor over last years, and it has something to do with its derivative. 
# The derivative is used to calculate the gradient. But, its value is very low when the input is either small or large.
# When a neuron is not activated correctly, it could be that the gradient to change this will never get a value to make a significant change to the weigths and biases.
# This problem is called the vanishing gradient problem and gets more serious once we add more hidden layers.

# It turns out there is another activation function that is more suitable for hidden layers. It is called the rectified linear unit, ReLU, in short
# It has some advantages over the sigmoid activation function.
# Any input under zero is cut off at zero. Any inout above zero is passed.
# To implement this function, we can use Python's max function. 'p' is the prediction and the output will be the activation.
# The function looks linear, but in fact id non-linear to update the weights, an activation function needs to be differentiable.
# The ReLU is technically not differential at input zero, but in reality this is not a problem.
# What does the ReLU derivative look like? Anything under zero is zero. And anything above is one. 

# We have to change a lot. The good news is that once we are done, we have built a network thwt is so flexible that we can scale it up to recognise handwritten numbers with little extra effort.

# The feed forward will now have four steps.
# One: calculated predictions of the hidden layer.
# Two: apply the ReLU to activate the predictions.
# Three: calculate the predictions of the third layer.
# Four: apply the softmax to activate the predictions.

# The cost will be calculated
# Then the error derivatives of the outout layers are calculated.
# From the error derivative, the gradient is calculated. The gradient will have the desired changes to the weigths from hidden to output, but the weigths will not be updated yet.
# Notice that the weights matrix from hidden to output hidden to outout is called w_h_o. This is the name that will also be used in the code.

# Until here, everything works as before, but now a step is introduced that for a long time is very confusing.
# We know we need to calculate the gradient to change the w_i_h weights matrix, but to do that, we need to calculate the errors, for hidden layer.
# How do we do that? There is no target value to subtract here.
# Actually , this is the most complicated calculation in the whole network.
# Once we have error derivative, the w_i_h gradient can be calculated. Finally, all the weigths and biases in the network are updated. Sounds complicated. It is.

# Although it sounds logical enough, getting it all in your head is not a trivial task.
# The good news is that once we are done, the network will be so flexible that with minor changes, it will also be able to support recoganizing handwritten digits.
# To move from a network with two layers to a network with three layers, we could change the existing code.
# But in fact, we needd to change so many things. so it's better to start with blank code. 
# The only thing we will keep is the training and test data, the softmax function and log loss function.
