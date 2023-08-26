inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2,
    3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

# I define a weight, bias, epochs and learning rate.

w = 0.1
b = 0.3
epochs = 200
learning_rate = 0.01

# I chose the values arbitrarily and will adjust them when needed.
# I create the predict function.

def predict(i):
    return w * i + b

# And finally the training loop.

# train the network
for epoch in range(epochs):
    
    # First, I make the predictions for all training samples.
    
    preds = [predict(i) for i in inputs]
    
    # Then calculate and print the cost.
    
    cost = sum([(p - t) ** 2 for p, t in zip(preds, targets)])/len(targets)
    print(f"w:{w:.2f}, b:{b:.2f}, c:{cost:.2f}")
    
    # Now the back propagation starts. I calculate the derivatives of the errors.

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    
    # From the error derivatives, I calculate the weight and bias deltas with respect to their inputs.
    
    weigth_d = [e * i for e, i in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]
    
    # I'd like to repeat that the weight deltas are calculated from the error derivatives with respect to inputs.
    # And biased deltas are calculated from the error derivatives with respect to the bias input, which is one.
    # Finally, the weight and bias are updated.

    w -= learning_rate * sum(weigth_d) / len(weigth_d)
    b -= learning_rate * sum(bias_d) / len(bias_d)
   
# Let me execute the code.
# Look at the cost.
# It is still decreasing at the end.
# I'll increse the learning rate.