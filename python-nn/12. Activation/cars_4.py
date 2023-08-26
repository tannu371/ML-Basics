import math

inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] # 0 = keep, 1 = sell

weights = [0.1, 0.2]
b = 0.3
epochs = 400
learning_rate = 0.5

def predict(inputs):
    return sum([w * i for w, i in zip(weights, inputs)]) + b # weigthed sum of inputs

def activate(x):
    return 1 / (1 + math.exp(-x))

def log_loss(act, target):
    return -target * math.log(act) - (1 - target) * math.log(1 - act)

# train the network
for epoch in range(epochs):
    preds = [predict(inp) for inp in inputs]
    acts = [activate(p) for p in preds]

    cost = sum([log_loss(a, t) for a, t in zip(acts, targets)]) / len(acts)
    print(f"ep:{epoch}, c:{cost:.2f}")

    errors_d = [(a - t) for a, t in zip(acts, targets)]
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
test_acts = [activate(p) for p in test_preds]
for a, t in zip(test_acts, test_targets):
    print(f"target:{t}, predicted:{a:.0f}")


# Yes, the predictions are correct again and look at the cost. The cost is still changing, but at a much slower rate at the end.
# I'd say this network predicts pretty accurate.
