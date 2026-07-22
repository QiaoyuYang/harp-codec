from .base import AudioDataset

import os
import pandas as pd

class Jamendo(AudioDataset):

    def __init__(self, split, data_configs):
        super().__init__(split, data_configs)
    
    def get_metadata(self):
        metadata_path = os.path.join(self.dataset_root, "raw_30s_cleantags.tsv")
        
        # Read with flexible number of columns to handle multiple tags
        meta = pd.read_csv(
            metadata_path, 
            sep="\t",
            header=0,
            names=["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"],
            usecols=[0, 1, 2, 3, 4, 5],
            on_bad_lines='warn'
        )
        return meta if meta is not None else {}
    
    def get_audio_path(self, track_index):
        rel_path = self.meta.iloc[track_index]["PATH"]
        audio_path = os.path.join(self.dataset_root, rel_path)    
        return audio_path