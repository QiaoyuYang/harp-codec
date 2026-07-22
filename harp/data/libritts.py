from .base import AudioDataset
import os
import pandas as pd
from glob import glob


class LibriTTS(AudioDataset):
    def __init__(self, split, data_configs):
        super().__init__(split, data_configs)

    def get_tracks(self, split):
        """LibriTTS splits are directory-based, so return all tracks."""
        return list(range(len(self.meta)))
    
    def get_metadata(self):
        """
        Walks through LibriTTS directory structure to build metadata.
        Structure: LibriTTS/{split}/{speaker_id}/{chapter_id}/{speaker}_{chapter}_{utterance_id}_{segment}.wav
        """
        split_path = os.path.join(self.dataset_root, self.split)
        
        records = []
        
        # Find all wav files in the split directory
        wav_files = glob(os.path.join(split_path, "**", "*.wav"), recursive=True)
        
        for wav_path in wav_files:
            # Parse filename: {speaker_id}_{chapter_id}_{utterance_id}_{segment}.wav
            filename = os.path.basename(wav_path)
            basename = filename.replace(".wav", "")
            parts = basename.split("_")
            
            if len(parts) >= 4:
                speaker_id = parts[0]
                chapter_id = parts[1]
                utterance_id = parts[2]
                segment_id = parts[3]
            else:
                continue
            
            # Get relative path from dataset root
            rel_path = os.path.relpath(wav_path, self.dataset_root)
            
            # Read normalized transcript if available
            normalized_txt_path = wav_path.replace(".wav", ".normalized.txt")
            transcript = ""
            if os.path.exists(normalized_txt_path):
                with open(normalized_txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
            
            records.append({
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
                "segment_id": segment_id,
                "path": rel_path,
                "transcript": transcript
            })
        
        meta = pd.DataFrame(records)
        return meta if len(meta) > 0 else pd.DataFrame()

    def get_audio_path(self, track_index):
        rel_path = self.meta.iloc[track_index]["path"]
        audio_path = os.path.join(self.dataset_root, rel_path)
        return audio_path