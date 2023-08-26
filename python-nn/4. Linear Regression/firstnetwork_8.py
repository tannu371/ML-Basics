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

# test the network
test_inputs = [5, 6]  
test_targets = [10, 12]

# I make a list of test predictions. 
test_preds = [predict(i) for i in test_inputs]  
# And compare them with the test data.
for i , t, p in zip(test_inputs, test_targets, test_preds):
    print(f"input: {i}, target: {t}, pred: {p:.4f}")
    
# I execute the code.

# That is very accurate. 