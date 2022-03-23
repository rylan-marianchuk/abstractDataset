from abstract_dataset import AbstractDataset
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import torch
from ecg_dataset_utils import augment_to_12_leads

class EcgDataset(AbstractDataset):

    def __init__(self, xml_dataset_path, data_path, transform=None):
        """

        :param xml_dataset_path: (str) path to the standardized xml file outlining the entire dataset
        :param data_path: (str) path to the directory containing self.filenames for direct reads
        :param transform: (TYPE?) the by-lead transform to apply in __getitem__()
        """
        super(EcgDataset, self).__init__(xml_dataset_path, data_path)
        self.modality = "ECG"
        self.n_leads: int
        self.parse_assign_metadata()


    def get_observation_tensor(self, id):
        """
        Return the pytorch tensor of the given dataset id
        :param id: (int) identification long int
        :return: (tensor) shape=(n_leads, 5000) each row a lead of the ecg
        """
        f = h5py.File(self.data_path + self.filenames[id])
        np_ecg = np.array(f[self.modality])
        ecg = torch.from_numpy(np_ecg).view(8, 5000)
        if self.n_leads == 12:
            return augment_to_12_leads(ecg)
        return ecg


    def single_load(self, id):
        """
        The structure pulled out of this class by the DataLoader
        i.e.
        for X, y in loader:
            ...
        this method should return observation_tensor, target
        :param id: (int)
        :return: ecg, target, id
        """
        return self.get_observation_tensor(id), self.targets[id], self.ids[id]

    def parse_assign_metadata(self):
        tree = ET.parse(self.xml_dataset_path)
        self.n_leads = int(tree.find(".//n_leads").text)
