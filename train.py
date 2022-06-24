import torch
import getopt, sys
from models import simplemlp, siam

learning_rate = 0.001
batch_size = 32
epochs = 100

argumentList = sys.argv[1:]
options = "hm:"
long_options = "help, model:"

try:
    # Parsing argument
    args, vals = getopt.getopt(argumentList, options, long_options)

    for arg, val in args:
        if arg in ("-m", "--model"):
            if val == "mlp":
                model = simplemlp.SimpleMlp()
            elif val == "siam":
                model = siam.SiamNN()
            else:
                print("Model type not known. Valid models are: 'mlp', 'siam'.")
                raise ValueError
        elif arg in ("-e", "--epochs"):
            epochs = int(val)
        elif arg in ("-b", "--batch-size"):
            batch_size = int(val)
        elif arg in ("-l", "--learning-rate"):
            learning_rate = float(val)
        else:
            print("Valid arguments: help, model")
            exit(1)

except getopt.error as err:
    print(str(err))

    
