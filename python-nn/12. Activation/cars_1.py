# The first thing I do is change the targets into classification zero or one.
# Remember that zero means keep the car. And one means sell the car.
# I do  the same with test targets.

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

# The next thing to do is add the sigmoid activation function. And use it.
# The feed forward now has two steps. Calculate the predictions. Each prediction is a scalar value. 
# Then the predictions are turned into activations. Each activation has a value between zero and one.

def activate(x):
    return 1 / (1 + math.exp(-x))

# train the network
for epoch in range(epochs):
    preds = [predict(inp) for inp in inputs]
    acts = [activate(p) for p in preds]
    print(acts)
    exit()
    cost = sum([(p - t) ** 2 for p, t in zip(preds, targets)])/len(targets)
    print(f"ep:{epoch}, c:{cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    weights_d = [[err * i for i in inp] for err, inp in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]
    weights_d_T =list(zip(*weights_d))  # transpose weight_d

    for i in range(len(weights)):
        weights[i] -= learning_rate * sum(weights_d_T[i]) / len(weights_d)   
    b -= learning_rate * sum(bias_d) / len(bias_d)

# test the network
test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [0, 0, 1, 0, 0] # 0 = keep, 1 = sell

test_preds = [predict(inp) for inp in test_inputs]
for p, t in zip(test_preds, test_targets):
    print(f"target:${t}, predicted:${p:.0f}")

# Notice that all values are squished between zero and one. This is the result of feeding the predictions to the sigmoid function.
# Let's have a look at the cost function.