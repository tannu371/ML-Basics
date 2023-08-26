# I am going to move to points to up. I'll increase each target value by 10.
# I also update the test target values. 5 times 2 plus 10 is 20. 6 times 2 plus 10 is 22.

inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]

w = 0.1
learning_rate = 0.1
epochs = 10

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

# Let's see how well the network is learning.

# You can already see that the cost remains high. And also, the predictions are way off. 
# Even increasing the epochs won't help us here.