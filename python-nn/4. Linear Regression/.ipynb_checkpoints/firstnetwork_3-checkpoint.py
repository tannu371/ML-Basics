inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w =0.1

def predict(i):
    return w*i

# train the network
for _ in range(10):
    preds = [predict (i) for i in inputs]
    errors = [t - p for t, p in zip(targets, preds)]
    cost = sum(errors)/len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    w += cost
# I add the cost to the weight. 
# Neural network learn by repeatedly processing the training data.

# Let's put the whole thing in a loop to see what happens to the cost after 10 iterations.
# When I execute the code, the cost should go to zero.
# As you can see, the cost values are all over the place.
# Let us plot them in a diagram. cost vs iteration

# The diagram shows an oscillating pattern. 
# And instead of decreasing the cost, it gets bigger with each iteration. 
# Apparently, the adjustments are too big.
# To reduce the adjustments, we use something called a learning rate.
# Let us add it to the code.