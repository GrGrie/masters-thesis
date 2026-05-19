import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        self.transform = transform
        
        # Load metadata
        metadata_path = os.path.join(root_dir, 'metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            
            # Map splits if necessary (e.g., Waterbirds uses 0: train, 1: val, 2: test)
            split_dict = {'train': 0, 'val': 1, 'test': 2}
            self.metadata = self.metadata[self.metadata['split'] == split_dict[split]]
        else:
            self.metadata = pd.DataFrame()
            # In a real scenario, this would raise an error or download the dataset.
            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.metadata.empty:
            return None, None, None

        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx]['img_filename'])
        image = Image.open(img_path).convert('RGB')
        
        # 'y' is the object label (e.g., 0 for landbird, 1 for waterbird)
        # 'place' is the background label (e.g., 0 for land, 1 for water)
        target = self.metadata.iloc[idx]['y']
        background = self.metadata.iloc[idx]['place']
        
        if self.transform:
            image = self.transform(image)
            
        return image, target, background
