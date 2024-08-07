# The flowers here in the training and test data could be divided by straight lines. This data is very linear, but what happens when the data is non-linear?
# Neural network is a function approximator. It tries to emulate a function by learning from training data.
# Our network has two layers. Input and Output. And although the softmax function introduces some nonlinearity to categorize the result. The rest is all linear.

# How does this network deal with non-linear data?
# Some flowers are very close together and still have a different color. OU]ur network might have some trouble classifying the flowers.

import flowersdata_1 as data
import math

weights = [0.1, 0.2], [0.15, 0.25], [0.18, 0.1]
biases = [0.3, 0.4, 0.35]
epochs = 100
learning_rate = .2

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

    weights_d_T = list(zip(*weights_d))  # transpose weigths_deltas
    for y in range(len(weights_d_T)):
        for x in range(len(weights_d_T[0])):
            weights[y][x] -= learning_rate * weights_d_T[y][x] / len(data.inputs)
        biases[y] -= learning_rate * biases_d[y] / len(data.inputs)

# test the network
test_preds = [[sum([w * i for w, i in zip(we, inp)]) + 
              bi for we, bi in zip(weights, biases)] for inp in data.test_inputs]
act = [softmax(p) for p in test_preds]
correct = 0  # keep a counter with correct predictions.
for a, t in zip(act, data.test_targets):
    if a.index(max(a)) == t.index(max(t)):
        correct += 1
print(f"Correct: {correct}/{len(act)} ({correct / len(act):%})")

# Only 18 out 30 samples are predicted correctly. What can we do at this point?
# Adjust the epochs and learning rate.
# Increase the epochs