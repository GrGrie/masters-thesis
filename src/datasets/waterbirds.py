import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


SPLIT_TO_ID = {
    "train": 0,
    "val": 1,
    "test": 2,
}


class WaterbirdsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Waterbirds dataset for testing robustness to spurious correlations.
        
        Args:
            root_dir (str): Path to the dataset directory containing images and metadata.csv.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split not in SPLIT_TO_ID:
            valid = ", ".join(SPLIT_TO_ID)
            raise ValueError(f"Unknown Waterbirds split '{split}'. Expected one of: {valid}")

        metadata_path = os.path.join(root_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Waterbirds metadata not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        required_columns = {"img_filename", "y", "place", "split"}
        missing_columns = required_columns.difference(metadata.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Waterbirds metadata is missing required column(s): {missing}")

        self.metadata = metadata[metadata['split'] == SPLIT_TO_ID[split]].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx]['img_filename'])
        image = Image.open(img_path).convert('RGB')

        target = int(self.metadata.iloc[idx]['y'])
        spurious = int(self.metadata.iloc[idx]['place'])
        group = target * 2 + spurious
        
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "target": target,
            "spurious": spurious,
            "group": group,
            "index": int(idx),
            "path": img_path,
            # Dataset-specific aliases kept for interpretability and compatibility.
            "place": spurious,
            "background": spurious,
        }
