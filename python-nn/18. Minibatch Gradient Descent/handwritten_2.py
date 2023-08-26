# Let's see what's in the data. I'll print the first sample of each batch.
import mnistreader_1 as reader


epochs = 1
batch_size = 2000

for epoch in range(epochs):
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        print(inputs[0])

# Those are a lot of numbers.        
        