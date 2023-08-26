inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
learning_rate = 0.1

def predict(i):
    return w * i

# train the network
for _ in range(25):
    preds = [predict(i) for i in inputs]
    errors = [(p - t) ** 2 for p, t in zip(preds, targets)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    weigth_d = [e * i for e, i in zip(errors_d, inputs)]  # weight_delta contains the changes that each training sample wants to make to the weight
    print(weigth_d)
    exit()
    w += learning_rate

# test the network
test_inputs = [5, 6]
test_targets = [10, 12]
test_preds = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_preds):
    print(f"Weigth: {i}, target:{t}, pred: {p:.4f}")
    
# I excecute the code.

# weight_d now contains the changes that each training sample wants to make to the weight.

# Now, the average of weight_d will be used to update the weight.
# Let us delete the print and exit statement.