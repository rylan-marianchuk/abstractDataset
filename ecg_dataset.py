from abstract_dataset import AbstractDataset
import xml.etree.ElementTree as ET
from ecg_dataset_utils import get_tensor_from_filename, viewECG
import os

class EcgDataset(AbstractDataset):

    def __init__(self, xml_dataset, data_path=None,  slice=None, transform=None):
        """
        :param xml_dataset: (str) xml file outlining the entire dataset, generated according to standards TODO  (HERE)
                            if no path & only filename, uses DATASETS_PATH system environment variable as
                            if path to the file, uses it and ignores system env var
        :param data_path: (str) path to directory containing all observations, named by filename_uid in the xml_dataset
                          if None, uses ECGS_PATH system environment variable
        :param slice: (slice obj) if not None, slices the list of observations in xml_dataset
        :param transform: (callable object) the by-lead transform to apply in __getitem__()
        """
        # Use parameter as the path if it is a path, otherwise get the environment variable holding the path
        if xml_dataset is not None:
            if "/" in xml_dataset: xml_dataset = xml_dataset
            else: xml_dataset = os.getenv("DATASETS_PATH") + xml_dataset
        if data_path is None: data_path = os.getenv("ECGS_PATH")

        super(EcgDataset, self).__init__(xml_dataset, data_path, slice)

        self.modality = "ECG"
        self.n_leads: int
        self.parse_assign_metadata()


    def __getitem__(self, idx):
        """
        PyTorch executes this method in a multithread fashion
        The structure pulled out of this class by the DataLoader
        i.e.
        for X, y, ids in loader:
            ...
        this method should return observation_tensor, target, id
        :param idx: (int)
        :return: ecg, target, id
        """
        return get_tensor_from_filename(self.filenames[idx], n_leads=self.n_leads, given_data_path=self.data_path), self.targets[idx], self.ids[idx]


    def view_encounter(self, id, lead_id=None, save_to_disk=False):
        """
        Implementation of the abstract declaration
        :param id: (int) id number in this dataset to view
        :param lead_id: (int or str) if viewing only a single lead is desired,
        :param save_to_disk: (bool) whether the viewable object should be saved as a file and not temporarily viewed in browser
        :return: None
        """
        print("Viewing ECG with filename: " + self.filenames[id])
        viewECG(self.filenames[id], target=self.targets[id], n_leads=self.n_leads, lead_id=lead_id, given_data_path=self.data_path, save_to_disk=save_to_disk)


    def parse_assign_metadata(self):
        """
        Parse the standardized xml dataset file and assign class variables from its extraction
        :return: None
        """
        if self.xml_dataset_path is None:
            self.n_leads = 8
            return
        tree = ET.parse(self.xml_dataset_path)
        self.n_leads = int(tree.find(".//n_leads").text)
