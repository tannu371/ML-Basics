inputs = [(0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000),
    (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2, 40000), (3.3, 21000), (3.5, 45000),
    (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

w1 = 0.1
w2 = 0.2
b = 0.3
epochs = 4
learning_rate = 0.05

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
    
# You can see that the cost gets out of control rapidly.
# The changes to the weights and biases are too big.
# One way to control this is to decrease the learning rate. I'll add a few zeros after decimal.