## Source tutorial: https://github.com/smitkiri/urban-sound-classification?tab=readme-ov-file
# %% Libs
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152

from lib import utils
from lib.misc_fns import (
    get_melspectrogram_db,
    spec_to_image,
    train_val_split,
    load_data,
    evaluate,
    get_spec_loader,
)


# %% Cuda Device info: check if acceleration is available
torch_mem_info = torch.cuda.mem_get_info()
# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Globally available:", round(torch_mem_info[0] / 1024**3, 1), "GB")
    print("Total:   ", round(torch_mem_info[1] / 1024**3, 1), "GB")

# Check GPU compatibility with bfloat16 (pre turing GPUs probably won't be able to use it)
compute_dtype = getattr(torch, "float16")
if compute_dtype == torch.float16 and True:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        fp16_support = True
        #! Enable tensor core operations for fp16 and TF32 matmul
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True
        print("=" * 80)
    else:
        fp16_support = False

# %% Loading the train and test csv files
df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
df.head()

# %% Playing a sample of drilling sound excerpt
ipd.Audio("./data/Train/2.wav")

# %% Loading the sample using librosa
ex_data, ex_sr = librosa.load("./data/Train/2.wav")
ex_data.shape, ex_sr

# %% Seeing how the sound looks like on a wave plot
fig, ax = plt.subplots(figsize=(16, 4))
librosa.display.waveshow(ex_data, sr=ex_sr)
plt.ylabel("Amplitude")
plt.title("Waveform for a drilling sound")
plt.show()

# %% Getting a random audio file for each class
np.random.seed(0)
random_class_df = pd.DataFrame(
    df.groupby("Class")["ID"].apply(np.random.choice).reset_index()
)
random_class_df


# %% Reading data for the random audio files selected
random_class_data = []

for idx in random_class_df.index:
    file_name = str(random_class_df["ID"][idx]) + ".wav"
    wav, sr = librosa.load("./data/Train/" + file_name)

    random_class_data.append(wav)


# %%Plotting the waveforms for each class
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 30))
for i in range(5):
    librosa.display.waveshow(random_class_data[2 * i], sr=sr, ax=ax[i][0])
    ax[i][0].set_title(random_class_df["Class"][2 * i])

    librosa.display.waveshow(random_class_data[2 * i + 1], sr=sr, ax=ax[i][1])
    ax[i][1].set_title(random_class_df["Class"][2 * i + 1])

plt.show()


# %% Getting the spectrogram and the image of spectrogram from time-series audio
random_class_spec = [get_melspectrogram_db(wav, sr=sr) for wav in random_class_data]
random_class_spec_img = [spec_to_image(spec) for spec in random_class_spec]


# %% Plot mel spectrogram
ax = utils.plot_images(
    random_class_spec_img, ncols=2, figsize=(16, 25), bgr2rgb=False, axis_style="off"
)
for i in range(5):
    ax[i][0].set_title(random_class_df["Class"][2 * i])
    ax[i][1].set_title(random_class_df["Class"][2 * i + 1])

plt.show()


# %% Find the existing ratios of classes in the data
df.groupby("Class")["ID"].agg("count") / len(df)


# %% Converting classes into numeric format
df["numeric_class"] = df["Class"].astype("category").cat.codes
df


# %% Get the dictionary of classes based on their numeric value
classes = dict(df.groupby("numeric_class").agg("max")["Class"])
classes

# %% Splitting the data with 20% validation set
train_df, val_df = train_val_split(df, "Class", split_size=0.2, seed=0)
train_df.shape, val_df.shape


# %% Using an ANN on the TS data
# Load the data
BATCH_SIZE = 32
train_time_series, train_sr, train_labels = load_data(
    train_df, "ID", "numeric_class", "./data/Train"
)

val_time_series, val_sr, val_labels = load_data(
    val_df, "ID", "numeric_class", "./data/Train"
)
train_time_series.shape, val_time_series.shape

# %% Convert numpy arrays to torch tensors
train_time_series = torch.from_numpy(train_time_series)
train_labels = torch.from_numpy(train_labels).long()

val_time_series = torch.from_numpy(val_time_series)
val_labels = torch.from_numpy(val_labels).long()

# Create data loaders
train_time_series = data_utils.TensorDataset(train_time_series, train_labels)
train_loader = data_utils.DataLoader(
    train_time_series, batch_size=BATCH_SIZE, shuffle=True
)

val_time_series = data_utils.TensorDataset(val_time_series, val_labels)
val_loader = data_utils.DataLoader(val_time_series, batch_size=BATCH_SIZE, shuffle=True)


# %% Defining training parameters
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = len(classes)
N_FEATURES = train_time_series[0][0].shape[0]


# %% # Defining our neural network architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Layer 1 with 1024 neurons
        self.fc1 = nn.Linear(in_features=N_FEATURES, out_features=1024)

        # Layer 2 with 512 neurons
        self.fc2 = nn.Linear(in_features=1024, out_features=512)

        # Layer 3 with 128 neurons
        self.fc3 = nn.Linear(in_features=512, out_features=128)

        # Layer 4, output layer
        self.fc4 = nn.Linear(in_features=128, out_features=NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# %% # Defining loss and optimizer
net = NeuralNet()

#! If GPU is available send model to it
if device.type == "cuda":
    net.to(device)
    print("model sent to GPU")
else:
    print("model sent to CPU")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# %% Train the ANN model on the testing dataset
num_train_batches = len(train_loader)
loss_hist = []
acc_hist = []

for epoch in range(EPOCHS):
    print("Epoch " + str(epoch + 1) + ":")

    for i, batch in enumerate(train_loader):
        # batch is a tuple of input data and labels
        inputs, labels = batch
        #! Send to the GPU if possible
        if device.type == "cuda":
            inputs, labels = inputs.to(device), labels.to(device)
        #! If CUDA is avaiable, use mixed precision for FP16 training using ⭐️ ⭐️ Autocasting
        if (device.type == "cuda") & (fp16_support == True):
            print("FP16 available and in use for mixed precision training")
            with torch.cuda.amp.autocast():
                # Running forward pass
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        #! Otherwise fall back to FP32 training
        else:
            # Running forward pass
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measuring Accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        # Printing progress
        utils.drawProgressBar(
            (i + 1),
            num_train_batches,
            "\t loss: {:.4f} \t acc: {:.4f}".format(
                round(loss.item(), 4), round(accuracy, 4)
            ),
        )

    print("\n\n")
    acc_hist.append(accuracy)
    loss_hist.append(loss.item())


# %% Plotting the losses and accuracies on the testing dataset
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

ax[0].plot(range(1, len(loss_hist) + 1), loss_hist, c="orange")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_title("Loss progress through training")

ax[1].plot(range(1, len(acc_hist) + 1), acc_hist, c="green")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_title("Accuracy progress through training")

plt.show()


# %% Evaluate performance on the validation set
val_acc, val_loss = evaluate(net, val_loader, device)

print("\n\nValidation accuracy: {:.4f}".format(round(val_acc, 4)))
print("Validation loss: {:.4f}".format(round(val_loss, 4)))


# %% Now we try training a CNN on the same dataset to avoid overfitting
# --------------------
# Checking the sample rate of the audio files in train and validation sets
set(train_sr), set(val_sr)

# %% set frequency to 22050
train_sr = 22050
val_sr = 22050


# %% Getting the spectrogram image for each audio in train set
train_loader = get_spec_loader(train_time_series, train_sr, BATCH_SIZE, shuffle=True)

# Getting the spectrogram image for each audio in validation set
val_loader = get_spec_loader(val_time_series, val_sr, BATCH_SIZE)


# %% Define the ConvNet class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Layer 1, Input shape (1, 128, 173) ->  Output shape (8, 62, 84)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 6)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Layer 2, Input shape (8, 62, 84) -> Output shape (16, 30, 41)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Layer 3, Input shape (16, 30, 41) -> Output shape (64, 10, 15)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 7)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(6, 6)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Fully Connected layer 1, Input features 64 * 10 * 15 -> Output features 512
        self.fc1 = nn.Linear(in_features=64 * 10 * 15, out_features=512)

        # Fully Connected layer 2, Input features 512 -> Output features 256
        self.fc2 = nn.Linear(in_features=512, out_features=256)

        # Fully Connected layer 3, Input features 256 -> Output features 128
        self.fc3 = nn.Linear(in_features=256, out_features=128)

        # Fully Connected layer 4, Input features 128 -> Output features 10
        self.fc4 = nn.Linear(in_features=128, out_features=NUM_CLASSES)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        n_features = 1
        for s in size:
            n_features = n_features * s

        return n_features


# %% Defining loss and optimizer
model = ConvNet()

#! If GPU is available send model to it
if device.type == "cuda":
    model.to(device)
    print("model sent to GPU")
else:
    print("model sent to CPU")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %% Train the CNN model on the testing dataset
THRESHOLD = 0.001  # Threshold for early stopping
THRESH_EPOCHS = 3  # Number of epochs loss does not decrease before early stopping
PATH = "./models/ConvNet.pth.tar"  # Path to save the best model

loss_hist = []
acc_hist = []
num_train_batches = len(train_loader)

early_stop_epoch = 0
lowest_loss = np.inf

for epoch in range(EPOCHS):
    print("Epoch " + str(epoch + 1) + ":")

    for i, batch in enumerate(train_loader):
        # batch is a tuple of data and labels
        data, labels = batch
        #! Send to the GPU if possible
        if device.type == "cuda":
            data, labels = data.to(device), labels.to(device)

        #! If CUDA is avaiable, use mixed precision for FP16 training using ⭐️ ⭐️ Autocasting
        if (device.type == "cuda") & (fp16_support == True):
            print("FP16 available and in use for mixed precision training")
            with torch.cuda.amp.autocast():
                # Running forward pass
                outputs = model(data)
                loss = loss_fn(outputs, labels)
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        #! Otherwise fall back to FP32 training
        else:
            # Running forward pass
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measuring accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        # Printing progress
        utils.drawProgressBar(
            (i + 1),
            num_train_batches,
            "\t loss: {:.4f} \t acc: {:.4f}".format(
                round(loss.item(), 4), round(accuracy, 4)
            ),
        )

    print("\n")

    if abs(lowest_loss - loss.item()) < THRESHOLD:
        early_stop_epoch += 1
        print("Loss did not decrease from " + str(lowest_loss))

    else:
        print(
            "Loss decreased from {:.4f} to {:.4f}, saving model to {}".format(
                round(lowest_loss, 4), round(loss.item(), 4), PATH
            )
        )

        lowest_loss = loss.item()
        early_stop_epoch = 0
        torch.save({"state_dict": model.state_dict()}, PATH)

    acc_hist.append(accuracy)
    loss_hist.append(loss.item())
    print("\n\n")

# %% Plotting the losses and accuracies
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

ax[0].plot(range(1, len(loss_hist) + 1), loss_hist, c="orange")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_title("Loss progress through training")

ax[1].plot(range(1, len(acc_hist) + 1), acc_hist, c="green")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_title("Accuracy progress through training")

plt.show()


# %% Loading the best model
model = ConvNet()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["state_dict"])


# %% Getting the validation accuracy and loss
val_acc, val_loss = evaluate(model, val_loader, device)

print("\n\nValidation accuracy: {:.4f}".format(round(val_acc, 4)))
print("Validation loss: {:.4f}".format(round(val_loss, 4)))


# %% Now try using a pretrained ResNet34 model to do the same thing
# Need to change the input and output layer
resnet = resnet34(pretrained=True)

#! If GPU is available send model to it
if device.type == "cuda":
    resnet.to(device)
    print("model sent to GPU")
else:
    print("model sent to CPU")

resnet.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)
#! If GPU is available send weights to it
if device.type == "cuda":
    resnet.conv1.to(device)
resnet.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES)
#! If GPU is available send weights to it
if device.type == "cuda":
    resnet.fc.to(device)

# %% Model parameters
EPOCHS = 10
LEARNING_RATE = 0.0001
THRESHOLD = 0.001
THRESH_EPOCHS = 3
PATH = "./models/ResNet.pth.tar"


# %% Defining loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=LEARNING_RATE)


# %% Train the ResNet34 model on the testing dataset
loss_hist = []
acc_hist = []
num_train_batches = len(train_loader)

early_stop_epoch = 0
lowest_loss = np.inf

for epoch in range(EPOCHS):
    print("Epoch " + str(epoch + 1) + ":")

    for i, batch in enumerate(train_loader):
        # batch is a tuple of data and labels
        data, labels = batch
        #! Send to the GPU if possible
        if device.type == "cuda":
            print("sending to cuda")
            data, labels = data.to(device), labels.to(device)

        #! If CUDA is avaiable, use mixed precision for FP16 training using ⭐️ ⭐️ Autocasting
        if (device.type == "cuda") & (fp16_support == True):
            print("FP16 available and in use for mixed precision training")
            with torch.cuda.amp.autocast():
                # Running forward pass
                outputs = resnet(data)
                loss = loss_fn(outputs, labels)
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        #! Otherwise fall back to FP32 training
        else:
            # Running forward pass
            outputs = resnet(data)
            loss = loss_fn(outputs, labels)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measuring accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        # Printing progress
        utils.drawProgressBar(
            (i + 1),
            num_train_batches,
            "\t loss: {:.4f} \t acc: {:.4f}".format(
                round(loss.item(), 4), round(accuracy, 4)
            ),
        )

    print("\n")

    # Checking if loss has decreased
    if lowest_loss - loss.item() < THRESHOLD:
        early_stop_epoch += 1
        print("Loss did not decrease from {:.4f}".format(round(lowest_loss, 4)))

    else:
        print(
            "Loss decreased from {:.4f} to {:.4f}, saving model to {}".format(
                round(lowest_loss, 4), round(loss.item(), 4), PATH
            )
        )

        lowest_loss = loss.item()
        early_stop_epoch = 0
        torch.save({"state_dict": resnet.state_dict()}, PATH)

    acc_hist.append(accuracy)
    loss_hist.append(loss.item())
    print("\n\n")

# %% Plotting the losses and accuracies
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

ax[0].plot(range(1, len(loss_hist) + 1), loss_hist, c="orange")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_title("Loss progress through training")

ax[1].plot(range(1, len(acc_hist) + 1), acc_hist, c="green")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_title("Accuracy progress through training")

plt.show()


# %% Loading the best model
model = resnet
resnet_checkpoint = torch.load(PATH)
model.load_state_dict(resnet_checkpoint["state_dict"])


# %% Getting the validation accuracy and loss
val_acc, val_loss = evaluate(model, val_loader, device)

print("\n\nValidation accuracy: {:.4f}".format(round(val_acc, 4)))
print("Validation loss: {:.4f}".format(round(val_loss, 4)))

# %%
