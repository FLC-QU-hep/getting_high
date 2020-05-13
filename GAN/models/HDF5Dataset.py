
import h5py
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, path, name):
        self.file_path = path
        self.dataset = None
        self.name = name
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["{}/energy".format(name)])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["{}".format(self.name)]

        return {'energy' : self.dataset['energy'][index],
                 'shower' : self.dataset['layers'][index]}

    def __len__(self):
        return self.dataset_len
