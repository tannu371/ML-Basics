# How do we prepare our training set for mini batch gradient descent ?

# The first thing we need is training and test data.
# The first number is the target digit or the label. I added one hot encoding to the data. For this, we need 10 extra numbers. 
# The rest of the numbers is the image pixel data numbered between zero and one.
# Let’s create a script that can read and show this data.
# I copied the CSV file to the project.
# Now I create a script that reads the training data.

import random 

def get_training_samples(batch_size):
    with open("/Users/tannu/Projects/python-nn/18. Minibatch Gradient Descent/train.csv") as file:
        text = file.read()
    textlines = text.strip().split("\n") # At this point, the complete file is in memory and split on new lines.
    # I shuffle the data.
    random.shuffle(textlines) 
    
    # Now I loop through the texlines and whenever the batch size  is reached, I yield a new batch.
    start = 0
    while start < len(textlines):
        labels = []
        targets = []
        inputs = []
        end = start + batch_size
        for textline in textlines [start:end]:
            cells = textline.split(",")
            labels.append(int(cells[0]))
            targets.append([float(c) for c in cells[1:11]])
            inputs.append([float(c) for c in cells[11:]])
        yield labels, targets, inputs
        start += batch_size
        