inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w =0.1
learning_rate = 0.01

def predict(i):
    return w*i

# train the network
for _ in range(20):
    preds = [predict(i) for i in inputs]
    errors = [t - p for t, p in zip(targets, preds)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    w += learning_rate * cost

# The cost keeps going down, but it is not close to zero. 
# I'll make the learning rate 10 times larger.