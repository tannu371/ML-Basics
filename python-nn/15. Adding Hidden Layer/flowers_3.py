import flowersdata_1 as data
import math

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
    return sum(losses)

epochs = 1
learning_rate = .1

w_i_h = [[0.1, -0.2], [-0.3, 0.25], [0.12, 0.23], [-0.11, -0.22]]  # 4 hidden neurons
w_h_o = [[0.2, 0.17, 0.3, -0.11], [0.3, -0.4, 0.5, -0.22], [0.12, 0.23, 0.15, 0.33]]
b_i_h = [0.2, 0.34, 0.21, 0.44]  # 4 hidden neurons
b_h_o = [0.3, 0.29, 0.37]  # 3 output neuroms

for epoch in range(epochs):
    pred_h = [[sum([w * a for w, a in zip(weights, inp)]) + 
               bias for weights, bias in zip(w_i_h, b_i_h)] for inp in data.inputs]
    act_h = [[max(0, p) for p in pred] for pred in pred_h] # apply ReLU
    print(act_h)

# All neurons seem to have a value over zero now. Let's proceed with the prediction of the output layer.