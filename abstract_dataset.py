import os
from abc import abstractmethod
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch
import plotly.graph_objs as go

class AbstractDataset(Dataset):

    def __init__(self, xml_dataset_path, data_path, slice):
        """
        Populate filenames, targets, and extract the dataset metadata by parsing the given standardized xml dataset
        :param xml_dataset: (str) path to xml file outlining the entire dataset, generated according to standards TODO  (HERE)
        :param data_path: (str) path to directory containing all observations, named by filename_uid in the xml_dataset
        :param slice: (slice obj) if not None, slices the list of observations in xml_dataset
        """
        self.slice = slice
        self.xml_dataset_path = xml_dataset_path
        self.data_path = data_path
        self.desc: str
        self.filenames: list = []
        self.targets = torch.Tensor([])
        self.n_obs: int
        self.continuous_targets: bool
        self.parse_assign_metadata_abstract()
        self.ids = torch.LongTensor(range(self.n_obs))


    @abstractmethod
    def view_encounter(self, id, save_to_disk=False):
        pass

    def __len__(self):
        return self.n_obs

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def parse_assign_metadata_abstract(self):
        """

        :return:
        """
        if self.xml_dataset_path is None:
            self.desc = "Zero-Vector-Targets"
            self.continuous_targets = False
            self.filenames = os.listdir(self.data_path)
            self.targets = torch.zeros(len(self.filenames))
        else:
            tree = ET.parse(self.xml_dataset_path)
            self.desc = tree.find(".//desc").text
            obs_in_xml = int(tree.find(".//n_obs").text)
            self.targets = torch.zeros(obs_in_xml, dtype=torch.float32)
            self.continuous_targets = bool(tree.find(".//continuous_target").text)
            all_obs = tree.find(".//dataset").findall("obs")
            for i,obs in enumerate(all_obs):
                self.filenames.append(obs.find(".//filename_uid").text)
                self.targets[i] = float(obs.find(".//target").text)

        if self.slice is not None:
            self.filenames = self.filenames[self.slice]
            self.targets = self.targets[self.slice]
        self.n_obs = len(self.filenames)
        return

    def view_by_parameter(self, callables, names, cap_points=5000, color_by_target=True, save_to_disk=False):
        """

        :param callables:
        :param names:
        :param cap_points:
        :param color_by_target:
        :param save_to_disk:
        :return:
        """
        if isinstance(callables, list):
            if len(callables) > 3: raise Exception("Cannot visualize more than 3 parameters at a time")
        else: callables = (callables)

        dims = len(callables)
        features = torch.zeros(dims, len(self))
        for i in range(len(self)):
            observation = self.get_observation_tensor(i)
            for j in range(dims):
                features[j,i] = callables[j](observation)



    def view_target_distribution(self, save_to_disk=False):
        """

        :param save_to_disk:
        :return:
        """
        if self.continuous_targets:
            # Violin plot
            fig = go.Figure(data=go.Violin(y=self.targets, box_visible=True, line_color='black',
                                           meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                                           x0='Target Distribution of ' + self.desc))
            fig.update_layout(title="Violin Plot of Target: " + self.desc)
        else:
            # Categorical frequency plot
            categories = sorted(set(self.targets.numpy()))

        if save_to_disk: fig.write_html("TargetDistribution.html")
        else: fig.show()
