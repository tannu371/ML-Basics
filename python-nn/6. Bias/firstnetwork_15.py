inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]

w = 0.1
learning_rate = 0.1
epochs = 40

def predict(i):
    return w*i

# train the network
for _ in range (epochs):
    preds = [predict(i) for i in inputs]
    errors = [(p - t) ** 2 for  p, t in zip(preds, targets)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    w -= learning_rate * sum(weight_d)/len(weight_d)

# test the network
test_inputs = [5, 6]
test_targets = [20, 22]
test_preds = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_preds):
    print(f"input: {i}, target: {t}, pred: {p:.4f}")

# After a few epochs, the weight and cost do not change anymore. 
# This is the best the network can do, so we need a new strategy. 
