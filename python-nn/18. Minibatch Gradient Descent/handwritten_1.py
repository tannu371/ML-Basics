# Let's test if I can retrieve a batch.

import mnistreader_1 as reader


epochs = 1
batch_size = 2000

for epoch in range(epochs):
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        print(len(inputs))