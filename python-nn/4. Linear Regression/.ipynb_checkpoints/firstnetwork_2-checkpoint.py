inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# The data points are added and know it's time to train the network.

w =0.1

def predict(i):
    return w*i

# train the network

# I will now make a prediction for all the inputs and put the results in variable 'pred'.
preds = [predict(i) for i in inputs] 
# 'pred' now holds a list of predictions.
# The network now needs a way to express how accurate the predictions are. 
# Each target value can be compared with a predicton. The differece between the two is known as the error of a single training sample.
# I'll create a list with errors for all samples.
errors = [t - p for t, p in zip(targets, preds)] 
# Then we calculate the average over all the errors.
cost = sum(errors)/len(targets) 
# This average is called the cost. It is a single number that indicates how well the network is doing.
# Let's print the slope and the cost.
print(f"Weight: {w:.2f}, Cost: {cost:.2f}" ) 
# Notice I am printing slope w and mark it as weight. For our purposes, weight and slope are the same thing.

# Let me execute the code.

# When we use weight 0.1, the cost is 4.75.
# You will get a good understanding of the meaning of this value soon.
# For now, let's just say the goal is to get the cost as close to zero as possible.

# That leaves us with a question, how can we make the cost go to zero?
# It turns out that this is not a trivial question, which we will see once our network will progress.
# For now, however, let's use the cost value to adjust the only paramter in the network that we control the weight.