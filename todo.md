# Workflow
1. read in a dataset with variation in classes
    1. we are going to use the urban sound classification dataset with 10 classes and ~8000 files evenly spread
2. use basic feature extraction to create mel spectrograms
3. try a basic `ANN architecture` to classify sounds from the wave file timeseries with the following\
   layout:
    1. model: torch.NeuralNet
    2. Layers: 3 + 1 output
    3. Neurons: L1=1024, L2=512, L3=128
    4. Loss function: Cross Entropy Loss
    5. optimizer: Adam
    6. Epochs: 10
    7. Batch size: 32
4. to improve on the basic timeseries wave data, train a CNN on the data
    1. use extracted mel frequency features as the input to the CNN
    2. `CNN architecture`:
        1. model: torch.ConvNet
        2. Layers: 3 + 1 output
        3. Neuron structure: 
            1. L1= Input shape (1, 128, 173) ->  Output shape (8, 62, 84) 
            2. L2= Input shape (8, 62, 84) -> Output shape (16, 30, 41)
            3. L3= Input shape (16, 30, 41) -> Output shape (64, 10, 15)
        4. Loss function: Cross Entropy Loss
        5. optimizer: Adam
        6. Epochs: 10, but dynamic depending on observed loss (3 epochs where loss <= 0.001)
        7. Batch size: 32
5. verify the usefulness of using a transfer learning approach instead of a train -> validate approach because:
    1. the dataset we have is much smaller (~40 samples/class vs ~800 samples/class)
    2. pre-trained architectures like ResNet34 can be more effective at audio classification through a transfer learning approach
    3. `ResNet architecture`:
        1. model: torch.resnet34
        2. Layers: *pre-trained*
        3. Neuron structure: *pre-trained*
        4. Loss function: Cross Entropy Loss
        5. optimizer: Adam
        6. Epochs: 5, but dynamic depending on observed loss (3 epochs where loss <= 0.001)
        7. Batch size: 32
6. now use the home audio dataset to run these established workflows, and report on results

# Future work
- VGGNet transfer learning model
- Find:
    - Precision
    - Recall 
    - F1 Score