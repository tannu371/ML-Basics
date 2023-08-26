import flowersdata_3 as data
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


# You might wonder why we're not creating helper functions for transposing and multiplying matrices.
# One reason is that so that you are able to follow the code without having to jump to functions all the time.
# Another reason is that once you understand how it works, you will create your own helper functions or switch to using Numpy.
# Numpy has built- in functions to transpose and multiply matrices.

# We keep expanding the network and went from a single neuron with a single input to a network with two inputs all the way to a network with two input neuron and three output neurons.
# As you are noticing, the complexity lies in the matrices of inputs and weights.
# We keep learning new tools that will help us with our ultimate goal: hand written number recognition.

# Compare the most active neuron of the prediction with the most active neuron of the target.

# Let's increase the epochs and try again.