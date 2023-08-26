inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
# I will set the learning rate to 0.01.
# This is just an arbitrary value to see what happens.
learning_rate =0.01  

def predict(i):
    return w*i

# train the network
for _ in range(10):
    preds = [predict(i) for i in inputs]
    errors = [t - p for t, p in zip(targets, preds)]
    cost = sum(errors)/len(targets)
    w += learning_rate * cost
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    
# I execute the code.
# Yes, that looks better.

# By taking small correction steps, the cost gets closer to zero. 
# Now we need to find a good balance between the learning rate and iterations.

# I'll start by changing the iteratioins to 20.

