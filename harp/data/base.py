import torch
from torch.utils.data import Dataset
import torchaudio

class BaseDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = self.get_metadata()

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self, split):
        all_tracks = list(range(len(self.meta)))
        split_ratio = 0.9
        if split == "train":
            tracks = all_tracks[:int(split_ratio * len(all_tracks))]
        elif split == "val":
            tracks = all_tracks[int(split_ratio * len(all_tracks)):]
        elif split == "all":
            tracks = all_tracks
        return tracks
        
    def __getitem__(self, index):
        raise NotImplementedError("Item retrieval is not defined.")
    
    def get_metadata(self):
        raise NotImplementedError("Metadata retrieval is not defined.")
    
    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)

        return audio

class AudioDataset(BaseDataset):
    """Dataset class for the audio dataset."""

    def __init__(self, split, data_configs):
        self.dataset_root = data_configs["dataset_root"]
        self.split = split
        super().__init__()
        self.sample_rate = data_configs["sample_rate"]
        self.segment_length = data_configs["segment_length"]
        self.tracks = self.get_tracks(split)
    
    def __getitem__(self, index):
        track_index = self.tracks[index]
        audio_path = self.get_audio_path(track_index)
        audio = self.load_audio(audio_path)
    
        # Random segment if segment_length is specified
        if self.segment_length is not None:
            audio = self._get_random_segment(audio)
        
        return audio
    
    def _get_random_segment(self, audio):
        segment_samples = int(self.segment_length * self.sample_rate)
        # Use torch
        # Randomly select a segment
        if audio.shape[1] > segment_samples:
            start_idx = torch.randint(0, audio.shape[1] - segment_samples + 1, (1,)).item()
            audio = audio[:, start_idx:start_idx + segment_samples]
        else:
            pad_width = segment_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_width))
        
        return audio
    
    def get_audio_path(self, track_index):
        raise NotImplementedError