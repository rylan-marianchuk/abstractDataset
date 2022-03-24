from abstract_dataset import AbstractDataset
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import torch
from ecg_dataset_utils import get_tensor_from_filename, viewECG

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


    def single_load(self, id):
        """
        The structure pulled out of this class by the DataLoader
        i.e.
        for X, y, ids in loader:
            ...
        this method should return observation_tensor, target, id
        :param id: (int)
        :return: ecg, target, id
        """
        return get_tensor_from_filename(self.filenames[id], self.data_path, self.n_leads), self.targets[id], self.ids[id]


    def view_encounter(self, id, lead_id=None, save_to_disk=False):
        """

        :param id:
        :param save_to_disk:
        :return:
        """
        print("Viewing ECG with filename: " + self.filenames[id])
        viewECG(self.filenames[id], self.n_leads, lead_id, save_to_disk)


    def parse_assign_metadata(self):
        tree = ET.parse(self.xml_dataset_path)
        self.n_leads = int(tree.find(".//n_leads").text)
