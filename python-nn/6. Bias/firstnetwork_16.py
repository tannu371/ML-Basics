# I add a variable for the bias. 

inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]

w = 0.1
b = 0.3
# I chose the value arbitrarily. 
# Later, we will let Python chose a random value.
# The reason I am not doing this now is because I want to get the same training results each time the code runs.

learning_rate = 0.1
epochs = 40

# I will use the bias in the predict function.

def predict(i):
    return w * i + b

# train the network
for _ in range (epochs):
    preds = [predict(i) for i in inputs]
    errors = [(p - t) ** 2 for  p, t in zip(preds, targets)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f},Bias: {b:.2f}, Cost: {cost:.2f}")
    
    # With all the preparation we have done to update the weight, it is now surprisingly easy to update the bias.
    # Look at weight_d. It is list of error derivatives with respect to the training inputs.
    # The bias deltas are calculated in similar way.

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]
    
    # You might wonder why I create a list where each element is an error derivative times one.
    # If you obsreve this code, you see that bias_d will just be a copy of errors_d.
    # You are right when you are thinking: bias_d equals errors_d, but for this example, I'd like to create the biased deltas in the explicit manner.
    # You know that the weight deltas are calculated with respect to the input. The same applied to the bias.
    # And since the bias input is always one, the bias deltas are calculated with respect to number one.
    # Let's use bias_d to update to bias.
    # I will also print the bias.
    
    w -= learning_rate * sum(weight_d)/len(weight_d)
    b -= learning_rate * sum(bias_d)/len(bias_d)

# test the network
test_inputs = [5, 6]
test_targets = [20, 22]
test_preds = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_preds):
    print(f"input: {i}, target: {t}, pred: {p:.4f}")
    
# let's see how well the network does with the bias added. 

# You can see the cost going down.
# But we also see that the bias is not nearing 10.
# Let's add some epochs and see what happens.