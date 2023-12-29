import pandas as pd
from tqdm import tqdm
import torchaudio
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import re
from transformers import RobertaTokenizer
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
from tqdm.contrib.concurrent import process_map


class MulDataset:
    def parallel_processing(self):
        results = process_map(
            self.to_memory, range(len(self)), max_workers=16, chunksize=16
        )
        return results


class CoronavirusDataset(Dataset, MulDataset):
    def __init__(self, train=True, load=False):
        super().__init__()
        if not load:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.tokenizer.model_max_length = 1000

        if train:
            self.df = self.preprocess("dataset/Coronavirus tweets/Corona_NLP_train.csv")
        else:
            self.df = self.preprocess("dataset/Coronavirus tweets/Corona_NLP_test.csv")

        self.labels = self.df["Sentiment"].values

        if load:
            print("load!")
            self.data = torch.load(f"dataset/Coronavirus tweets/{train}.pt")
        else:
            print("process!")
            self.data = []
            for i in tqdm(range(len(self))):
                self.data.append(self.to_memory(i))
            torch.save(self.data, f"dataset/Coronavirus tweets/{train}.pt")

    def to_memory(self, idx):
        def remove_links(text):
            to_remove = ["\r", "\n", ",", ";", ":", "."]
            out = re.sub(r"http\S+", "", text)
            for token in to_remove:
                out = out.replace(token, "")
            return re.sub(" +", " ", out.lower())

        text = remove_links(self.df["OriginalTweet"].values[idx])
        text = self.tokenizer.encode(text=text, padding="max_length")
        return torch.tensor(text), torch.tensor(self.labels[idx])

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess(self, path):
        df = pd.read_csv(path, encoding="latin1").sample(frac=1)

        encoder = LabelEncoder()
        df["Sentiment"] = encoder.fit_transform(df["Sentiment"])

        return df

    def __len__(self):
        return len(self.labels)


class BirdClefDataset(Dataset, MulDataset):
    def __init__(self, train=True, load=True):
        df = self.preprocess()

        if train == True:
            df = df[df["train"] == 1]
        else:
            df = df[df["train"] == 0]

        self.audio_paths = df["filename"].values
        self.labels = df["primary_label"].values

        self.transform_audio = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=512, hop_length=512, n_mels=64
        )
        self.target_sample_rate = 32000
        self.num_samples = self.target_sample_rate * 7  # sample_rate * duration
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if load:
            print("load!")
            self.data = torch.load(f"dataset/birdclef-2022/{train}.pt")
        else:
            print("process!")
            self.data = []
            for i in tqdm(range(len(self))):
                self.data.append(self.to_memory(i))
            torch.save(self.data, f"dataset/birdclef-2022/{train}.pt")

    def preprocess(self):
        df = pd.read_csv("dataset/birdclef-2022/train_metadata.csv")

        encoder = LabelEncoder()
        df["primary_label"] = encoder.fit_transform(df["primary_label"])

        df["train"] = 1
        df.loc[[i for i in range(0, len(df), 5)], "train"] = 0

        return df

    def __len__(self):
        return len(self.audio_paths)

    def to_memory(self, idx):
        audio_path = f"dataset/birdclef-2022/train_audio/{self.audio_paths[idx]}"
        signal, sr = torchaudio.load(audio_path)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0, keepdim=True)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        mel = self.transform_audio(signal)
        image = torch.cat([mel, mel, mel])
        image = image / torch.abs(image).max()
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label

    def __getitem__(self, idx):
        return self.data[idx]


class ButterflyDataset(Dataset, MulDataset):
    def __init__(self, train=True, load=True):
        df = self.preprocess()
        if train:
            df = df[df["train"] == 1]
        else:
            df = df[df["train"] == 0]

        self.image_paths = df["filename"].values
        self.labels = df["label"].values
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if load:
            print("load!")
            self.data = torch.load(f"dataset/Butterfly Image Classification/{train}.pt")
        else:
            print("process!")
            self.data = []
            for i in tqdm(range(len(self))):
                self.data.append(self.to_memory(i))
            torch.save(self.data, f"dataset/Butterfly Image Classification/{train}.pt")

    def preprocess(self):
        df = pd.read_csv("dataset/Butterfly Image Classification/Training_set.csv")
        encoder = LabelEncoder()
        df["label"] = encoder.fit_transform(df["label"])

        df["train"] = 1
        df.loc[[i for i in range(0, len(df), 2)], "train"] = 0
        return df

    def to_memory(self, idx):
        image_path = (
            f"dataset/Butterfly Image Classification/train/{self.image_paths[idx]}"
        )
        img = Image.open(image_path)

        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img.clone(), torch.tensor(label)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.image_paths)


class CIFAR100(Dataset):
    def __init__(self, train=True, load=True):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        if train:
            self.cifar100 = datasets.CIFAR100(
                "dataset/cifar100", train=True, download=True, transform=self.transform
            )
        else:
            self.cifar100 = datasets.CIFAR100(
                "dataset/cifar100", train=False, download=True, transform=self.transform
            )

        if load:
            print("load!")
            self.data = torch.load(f"dataset/cifar100/{train}.pt")
        else:
            print("process!")
            self.data = []
            for i in tqdm(range(len(self))):
                self.data.append(self.to_memory(i))
            # torch.save(self.data, f"dataset/cifar100/{train}.pt")

    def to_memory(self, idx):
        image, label = self.cifar100[idx * 10]
        return image, label

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.cifar100) // 10


if __name__ == "__main__":
    dataset = BirdClefDataset()
