inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
learning_rate = 0.1

def predict(i):
    return w*i

# train the network
for _ in range(25):
    preds = [predict(i) for i in inputs]
    errors = [(p - t) ** 2 for p, t in zip(preds, targets)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")

# There is only one thing we need to do before executing the code.
# Look at the weight deltas. They are negative. 
# The error function derivative calculates values that maximise the cost.
# Of course, we want to minimise the cost, and to do this, we subtract the deltas from the weight.

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    w -= learning_rate * sum(weight_d)/len(weight_d)

# test the network
test_inputs = [5, 6]
test_targets = [10, 12]
test_pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")

# I execute the code.

# Look at that result. After a few iterations, the network found the optimum weight. 
# I'll decrease the iterations, 10 should be enough.
# By the way, these iterations are called epochs.
# I will create a variable for this.