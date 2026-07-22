from .base import AudioDataset
import os
import pandas as pd
from glob import glob


class MUSDB18(AudioDataset):
    def __init__(self, split, data_configs):
        super().__init__(split, data_configs)

    def get_tracks(self, split):
        """MUSDB18 splits are directory-based (train/test), return all tracks."""
        return list(range(len(self.meta)))

    def get_metadata(self):
        """
        Walks through MUSDB18-HQ directory structure to build metadata.
        Structure: musdb18hq/{split}/{Artist - Track}/mixture.wav
        """
        split_path = os.path.join(self.dataset_root, self.split)
        
        records = []
        
        # Find all mixture.wav files in the split directory
        track_dirs = sorted(glob(os.path.join(split_path, "*")))
        
        for track_dir in track_dirs:
            if not os.path.isdir(track_dir):
                continue
            
            mixture_path = os.path.join(track_dir, "mixture.wav")
            if not os.path.exists(mixture_path):
                continue
            
            track_name = os.path.basename(track_dir)
            rel_path = os.path.relpath(mixture_path, self.dataset_root)
            
            records.append({
                "track_name": track_name,
                "path": rel_path,
            })
        
        meta = pd.DataFrame(records)
        return meta if len(meta) > 0 else pd.DataFrame()

    def get_audio_path(self, track_index):
        rel_path = self.meta.iloc[track_index]["path"]
        audio_path = os.path.join(self.dataset_root, rel_path)
        return audio_path