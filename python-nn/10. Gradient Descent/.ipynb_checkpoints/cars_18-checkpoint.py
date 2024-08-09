inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

weights = [0.1, 0.2]

# I replaced the weight variables with a weight list. The values are choosen arbitrarily.

b = 0.3
epochs = 4000
learning_rate = 0.1

# The next thing to do is to allow a list of inputs in the predict function.
# I need some way to loop through all the inputs and weights and sum their product. I use list comprehension for this.

def predict(inputs):
    return sum([w * i for w, i in zip(weights, inputs)]) + b

# As you can see, I have zipped the weights and inputs, then multiplied each weight times the input into a new list of products. And finally summed the resulting list.
# This is also know as the weighted sum of inputs.

# train the network
for epoch in range(epochs):
    
    # I can now pass all the input elements at once.
    
    preds = [predict(inp) for inp in inputs]
    cost = sum([(p - t) ** 2 for p, t in zip(preds, targets)])/len(targets)
    print(f"ep:{epoch}, c:{cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    
    # You probably have already noticed the duplicate code to calculate the weight updates. I'll make this more generic.
    
    weights_d = [[err * i for i in inp] for err, inp in zip(errors_d, inputs)]
    
    # At this point, the code gets a bit more complex. Let me show you why that is.
    
    bias_d = [e * 1 for e in errors_d]
    
    # I will replace the code lines that update the individual weights.
    
    weights_d_T =list(zip(*weights_d))  # transpose weight_d
    
    # First, I transposed the matrix so I can calculate the average easier.
    # I used puthon's zip function and pass and unpacked weights_d list. weights_d_T is now list of two rows with 16 elements each.
    # And now I look through the weights and update them with the transposed weight deltas.

    for i in range(len(weights)):
        weights[i] -= learning_rate * sum(weights_d_T[i]) / len(weights_d)   
    b -= learning_rate * sum(bias_d) / len(bias_d)
    
    # OK. The code should now be able to work with any number of inputs. 

# test the network
test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [500, 850, 1650, 950, 1375]

test_preds = [predict(inp) for inp in test_inputs]
for p, t in zip(test_preds, test_targets):
    print(f"target:${t}, predicted:${p:.0f}")
    
# Let's see if everything still works
# An error occur if you forgot to change the code for the test data.
# Try it again. 

# Everything works again.
# We have successfully refactored the code to work with any number of inputs.
# And it is at this point where we can explain gradient descent.

