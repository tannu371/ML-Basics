# I start by removing the test predictions.
# I paste the normalised training inputs.

inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

w1 = 0.1
w2 = 0.2
b = 0.3
epochs = 4000
learning_rate = 0.000000000005

def predict(i1, i2):
    return w1 *i1 + w2 * i2 + b

# train the network
for epoch in range(epochs):
    preds = [predict(i1, i2) for i1, i2 in inputs]
    cost = sum([(p - t) ** 2 for p, t in zip(preds, targets)])/len(targets)
    print(f"ep:{epoch}, c:{cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(preds, targets)]
    weigth1_d = [e * i[0] for e, i in zip(errors_d, inputs)]
    weight2_d = [e * i[1] for e, i in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]

    w1 -= learning_rate * sum(weigth1_d) / len(weigth1_d)
    w2 -= learning_rate * sum(weight2_d) / len(weight2_d)
    b -= learning_rate * sum(bias_d) / len(bias_d)

print(f"w1:{w1:.2f}, w2:{w2:.2f}, b:{b:.4f}")

# Let's execute the code.

# Nothing much has changed.
# But that is probably because the learning rate is so low.
# I'll set the learning rate to 0.1.
# This should bring the weights and bias up.

# The weights and bias went up. But what does that mean?
# We normalised the inputs, and the relationship between input and output is now difficult to see.