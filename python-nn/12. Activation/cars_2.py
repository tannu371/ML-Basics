import math

inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] # 0 = keep, 1 = sell

weights = [0.1, 0.2]
b = 0.3
epochs = 4000
learning_rate = 0.1

def predict(inputs):
    return sum([w * i for w, i in zip(weights, inputs)]) + b # weigthed sum of inputs

def activate(x):
    return 1 / (1 + math.exp(-x))

# The cost function for binary classification is called log loss.
# Since it is a biit more code than the mean squared error cost function , I'll create it as a separate function.

def log_loss(act, target):
    return -target * math.log(act) - (1 - target) * math.log(1 - act)

# train the network
for epoch in range(epochs):
    preds = [predict(inp) for inp in inputs]
    acts = [activate(p) for p in preds]
    
    # I use the log loss function to calculate the cost. The log loss function is called for each activation and target pair.
    # The result is averaged by summing and dividing by the numbers of activations. 
    # Now the error derivatives can be calculated.

    cost = sum([log_loss(a, t) for a, t in zip(acts, targets)]) / len(acts)
    print(f"ep:{epoch}, c:{cost:.2f}")
    
    # I subtract the targets from the activations.

    errors_d = [(a - t) for a, t in zip(acts, targets)]
    weights_d = [[err * i for i in inp] for err, inp in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]
    weights_d_T =list(zip(*weights_d))  # transpose weight_d
    
    # The calculation of the gradient and updating the weights stays the same.

    for i in range(len(weights)):
        weights[i] -= learning_rate * sum(weights_d_T[i]) / len(weights_d)   
    b -= learning_rate * sum(bias_d) / len(bias_d)
    
    
# test the network
test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [0, 0, 1, 0, 0] # 0 = keep, 1 = sell

# Use the activations when testing the network.

test_preds = [predict(inp) for inp in test_inputs]
test_acts = [activate(p) for p in test_preds]
for a, t in zip(test_acts, test_targets):
    print(f"target:{t}, predicted:{a:.0f}")


# I'll execute the code.
# Wow. That worked. The test cars are all predicted correctly. Notice how the cost did not change at the end.
# Let's try decreasing the epochs.

