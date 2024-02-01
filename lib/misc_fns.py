import numpy as np
import librosa
import pandas as pd
import utils
import torch
import torch.nn as nn
import torch.utils.data as data_utils


def get_fixed_audio_len(wav, sr, audio_len):
    """
    Converts a time-series audio to a fixed length either by padding or trimming

    Parameters
    -------------
    wav: Audio time-series

    sr: Sample rate

    audio_len: The fixed audio length needed in seconds
    """
    if wav.shape[0] < audio_len * sr:
        wav = np.pad(
            wav, int(np.ceil((audio_len * sr - wav.shape[0]) / 2)), mode="reflect"
        )
    wav = wav[: audio_len * sr]

    return wav


def get_melspectrogram_db(
    wav,
    sr,
    audio_len=4,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8300,
    top_db=80,
):
    """
    Decomposes the audio sample into different frequencies using fourier transform
    and converts frequencies to mel scale and amplitude to decibel scale.

    Parameters
    -------------------
    wav: Audio time-series

    sr: Sample rate

    audio_len: The fixed length of audio in seconds

    n_fft: Length of the Fast Fourier Transform window

    hop_length: Number of samples between successive frames

    n_mels: Number of mel filters, which make the height of spectrogram image

    fmin: Lowest frequency

    fmax: Heighest frequency

    top_db: Threashold of the decibel scale output
    """
    wav = get_fixed_audio_len(wav, sr, audio_len)

    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    spec = librosa.power_to_db(spec, top_db=top_db)

    return spec


def spec_to_image(spec):
    """
    Converts the spectrogram to an image

    Parameters
    -------------
    spec: Spectrogram
    """
    eps = 1e-6

    # Z-score normalization
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()

    # Min-max scaling
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)

    return spec_scaled


def train_val_split(df, target_col, split_size=0.33, seed=0):
    """
    Splits a dataframe into training and validation dataframes with
    approximately equal proportions of target column classes.

    Parameters
    --------------
    df: Dataframe to split

    target_col: The column to consider for equal split ratio

    split_size: The ratio of validation dataframe

    seed: Seed for random sample
    """
    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)

    for grp in set(df[target_col]):
        curr_grp_data = df[df[target_col] == grp]

        # Shuffle the dataframe
        curr_grp_data = curr_grp_data.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )

        split_idx = int(split_size * curr_grp_data.shape[0])
        val_df = pd.concat([val_df, curr_grp_data[:split_idx]])
        train_df = pd.concat([train_df, curr_grp_data[split_idx:]])

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df


def load_data(df, id_col, label_col=None, data_path="./", audio_len=4):
    """
    Loads the audio time-series data

    Parameters
    -------------
    df: The dataframe that contains the file name and corresponding label

    id_col: The column name that contains the file name

    label_col: The column name that contains the label
    """
    audio_time_series = []
    sample_rates = []
    labels = []

    tot = len(df)
    curr = 0

    for idx in df.index:
        try:
            file_name = str(df[id_col][idx]) + ".wav"
            wav, sr = librosa.load(data_path + "/" + file_name)

            wav = get_fixed_audio_len(wav, sr, audio_len)

            audio_time_series.append(wav)
            sample_rates.append(sr)

            if label_col is not None:
                labels.append(df[label_col][idx])

            curr += 1
            utils.drawProgressBar(curr, tot, barLen=40)

        except KeyboardInterrupt:
            print("KeyBoardInterrupt")
            break

        except Exception:
            print("Couldn't read file", df[id_col][idx])
            curr += 1

    print("\n")

    return np.stack(audio_time_series, axis=0), np.array(sample_rates), np.array(labels)


def evaluate(model, test_loader):
    """
    Returns the accuracy and loss of a model

    Parameters
    --------------
    model: A PyTorch neural network

    test_loader: The test dataset in the form of torch DataLoader
    """
    model.eval()
    num_test_batches = len(test_loader)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Printing progress
            utils.drawProgressBar((i + 1), num_test_batches)

        accuracy = correct / total
        test_loss = total_loss / num_test_batches

    return accuracy, test_loss


def get_spec_loader(audio_time_series, sr, batch_size, shuffle=False):
    """
    Returns data loader of spectrogram images

    Parameters
    ------------
    audio_time_series: Tensor Dataset with wav, label iterables

    sr: Sample rate

    batch_size: The batch size of data loader
    """
    audio_spec_img = []
    labels = []
    curr = 0
    tot = len(audio_time_series)

    for wav, label in audio_time_series:
        spec_img = spec_to_image(get_melspectrogram_db(wav.numpy(), sr))
        spec_img = np.expand_dims(spec_img, axis=0)
        audio_spec_img.append(spec_img)
        labels.append(label)

        curr += 1
        utils.drawProgressBar(curr, tot, barLen=40)

    audio_spec_img = torch.Tensor(audio_spec_img)
    audio_spec_img = audio_spec_img / 255

    labels = torch.Tensor(labels).long()

    audio_spec_img = data_utils.TensorDataset(audio_spec_img, labels)
    audio_loader = data_utils.DataLoader(
        audio_spec_img, batch_size=batch_size, shuffle=shuffle
    )

    return audio_loader
