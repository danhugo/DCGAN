import os
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image
import numpy as np
class LSUNDataset(Dataset):
    """# LSUN Dataset

    Arguments
    ---------
    - root (string): Directory to LSUN images folder.
    - transform (callable, optional): Optional transform to be applied on a sample.

    Attributes
    ----------
    - data (list): list[str] of image paths obtained from root.

    """

    def __init__(
            self, 
            root: str,
            nsample: int = None,
            transform: Optional[Callable] = None) -> None:
        """
        Arguments
        ---------
        - root (string): Directory to LSUN images folder.
        - transform (callable, optional): Optional transform to be applied on a sample.

        Attributes
        ----------
        - data (list): list[str] of image paths obtained from root.

        """
        self.root = root
        self.transform = transform
        if os.path.exists(self.root):
            self.data = os.listdir(self.root)
        else:
            raise NotADirectoryError(f"{self.root} does not exist or is not a directory.")
        
        if isinstance(nsample, int) and nsample is not None:
            if nsample <= len(self.data):
                self.data = self.data[:nsample]
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        """Get item by index. 

        Arguments
        ---------
        - index (int): Index
        
        Returns
        -------
        - (Image.Image or torch.tensor): A PIL Image or PyTorch tensor of images. If `to_tensor` is True, returns a
            tensor, otherwise returns an PIL Image.
        """
        img = self.load_data(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def load_data(self, path: str = None) -> np.ndarray:
        img = Image.open(os.path.join(self.root,path))
        return img
