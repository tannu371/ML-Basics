import mnistreader_3 as reader


epochs = 1
batch_size = 2000

for epoch in range(epochs):
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        print(labels[0])
        reader.plot_number(inputs[0])
        print()
        
# What does this code do? 
# watch this.
# We see the first digit of each batch printed with ASCII characters.
# So now we have batched training data.

# We also need to retrieve the test data.
# For this, I will write a function that is similarnto the get_training_samples function.
