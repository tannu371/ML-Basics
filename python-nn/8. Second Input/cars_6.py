# I start by adding the mileage to the training data.
# I'll pase a new training input set.
# Each input sample is now a tuple of age and mileage.

inputs = [(0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000),
    (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2, 40000), (3.3, 21000), (3.5, 45000),
    (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

# I will replace w with w1 and w2.

w1 = 0.1
w2 = 0.2

# I chose the values arbitrarily.
# Until now, the predict function calculated with one input and one weight.
# I will now this to support two inputs and two weights.

b = 0.3
epochs =400 
learning_rate = 0.05

def predict(i1, i2):
    return w1 *i1 + w2 * i2 + b

# Now I change the training loop to handle two inputs and two weights

# train the network
for epoch in range(epochs):
    preds = [predict(i1, i2) for i1, i2 in inputs]
    cost = sum([(p - t) ** 2 for p, t in zip(preds, targets)])/len(targets)
    print(f"ep:{epoch}, c:{cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    
    # Instead of a single weight_d, I need two lists. One for each weight deltas.
    # Weight1_d will have the changes with respect to the first input.   
    
    weigth1_d = [e * i[0] for e, i in zip(errors_d, inputs)]
    
    # Weight2_d will have the changes with respect to the second input.
     
    weight2_d = [e * i[1] for e, i in zip(errors_d, inputs)]
    
    # The  question is, what are we going to print?
    # I honestly don't know if printing weights and biases make sense anymore, so I'll just delete them. 
    # I'll print the epoch instead.
    # Ok. Almost there.
    # The only thing I need to do is update both weights.
    
    bias_d = [e * 1 for e in errors_d]

    w1 -= learning_rate * sum(weigth1_d) / len(weigth1_d)
    w2 -= learning_rate * sum(weight2_d) / len(weight2_d)
    b -= learning_rate * sum(bias_d) / len(bias_d)
    
# I delete the prediction at the end.
# That was a major change to the code.
# Let's see what happens.

# Python gives an overflow error.
# The first thiing I do is set epochs to 4 to see how the cost evolves from the start.