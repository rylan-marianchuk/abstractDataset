from abc import abstractmethod
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch
import plotly.graph_objs as go

class AbstractDataset(Dataset):

    def __init__(self, xml_dataset_path, data_path):
        self.xml_dataset_path = xml_dataset_path
        self.data_path = data_path
        self.desc: str
        self.filenames: list
        self.targets = torch.Tensor([])
        self.size: int
        self.continuous_targets: bool
        self.parse_assign_metadata_abstract()
        self.ids = torch.LongTensor(range(self.size))


    @abstractmethod
    def view_encounter(self, id, save_to_disk=False):
        pass

    @abstractmethod
    def single_load(self, id):
        pass

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.single_load(idx)
        else:
            return self.numerous_load(idx)

    def numerous_load(self, idx):
        print("Hello from numerous load")
        for id in idx:
            return self.single_load(id)

    def parse_assign_metadata_abstract(self):
        tree = ET.parse(self.xml_dataset_path)


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
