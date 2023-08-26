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

    errors_d = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act, data.targets)]
    inputs_T = list(zip(*data.inputs))  # transpose training inputs
    errors_d_T = list(zip(*errors_d))  # transpose error derivatives
    weights_d = [[sum([e * i for e, i in zip(er, inp)]) for er in errors_d_T] for inp in inputs_T]
    biases_d = [sum([e for e in errors]) for errors in errors_d_T]

    print(weights_d)

# Use the error derivatives to calculate the weight deltas.
# This will be a matrix multiplication where the error derivative matrix is multiplied by the inputs matrix.
# Variables postfixed with a capital T are transposed variations of the original matrices.

# You have learned that weights_d and biases_d together form the gradient to update the weights and biases.
# So what shape must weights_d and biases_d have?
# Are they matrices?
# Are they lists?
# It would be great if weigths_d has the same shape as the weights matrix