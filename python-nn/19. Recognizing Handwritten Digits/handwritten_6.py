import mnistreader_4 as reader
import random
import math

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
    return sum(losses)

# Now we are ready to start the training loop.

epochs = 3
batch_size = 400 
learning_rate =0.4

input_count, hidden_count, output_count = 784, 20, 10

w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]

# train the network
for epoch in range(epochs):
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        pred_h = [[sum([w * a for w, a in zip(weights, inp)]) +
                   bias for weights, bias in zip(w_i_h, b_i_h)] for inp in inputs]
        act_h = [[max(0,p) for p in pred] for pred in pred_h]
        pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + 
                   bias for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
        act_o = [softmax(predictions) for predictions in pred_o]
        
        # The training loop through the epochs. And then through the batches.
        # The hidden layer is activated by the ReLU activation function. And the output layer is activated by the softmax activation function.
        # I'll calculate and print the cost.
        
        cost = sum([log_loss(a, t) for a, t in zip(act_o, targets)]) / len(act_o)
        print(f"epoch:{epoch} cost:{cost:.4f}")
        
# The cost has been calculated and printed 150 times. 3 epochs times 50 batches of 400 samples.
# Each batch showed a different cost. But the cost did not go down because there is no back propagation yet.


