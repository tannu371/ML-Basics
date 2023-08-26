import mnistreader_4 as reader


epochs = 1
batch_size = 2000

# Let's print the first 10 test samples     
labels, targets, inputs = reader.get_test_samples()
for v, i in zip(labels[:10], inputs[:10]):
    print(v)
    reader.plot_number(i)
    print()
    
# Very nice. 
# We have prepared the data. We can retrieve batched training samples and a list of all the test samples.
# And we are able to print the numbers in the console.
# You can see that the handwritten numbers will be quite a challenge to recognize.
# How will our network do?
