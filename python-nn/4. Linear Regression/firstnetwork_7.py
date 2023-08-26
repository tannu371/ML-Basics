inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1 
learning_rate = 0.1

def predict(i):
    return w*i

# train the network
for _ in range(25):
    preds = [predict(i) for i in inputs]
    errors = [t - p for t, p in zip(targets, preds)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    w += learning_rate * cost

# Very nice

# Let us plot the cost for each iteration. This time, the error is gradually going down. 
# Since weight adjustments depend on the cost the updates are getting smaller as the cost gets nearer to zero.