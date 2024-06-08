from termcolor import colored
from keras import layers

# Compare the two inputs
def comparator(learner, instructor):
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
        
    print(colored("All tests passed!", "green"))

# Extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]

        if (type(layer) == layers.Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)

        if (type(layer) == layers.MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)

        if (type(layer) == layers.Dropout):
            descriptors.append(layer.rate)

        if (type(layer) == layers.ZeroPadding2D):
            descriptors.append(layer.padding)

        if (type(layer) == layers.Dense):
            descriptors.append(layer.activation.__name__)

        result.append(descriptors)

    return result