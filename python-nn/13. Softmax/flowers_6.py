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

    print(biases_d)

# Check if bias_d has the same shape as the biases.

# Transpose weights_d, update the weigths and update the biases.