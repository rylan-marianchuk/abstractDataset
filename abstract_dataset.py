import os
from abc import abstractmethod
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch
import plotly.graph_objs as go
import plotly.figure_factory as ff

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
            self.continuous_targets = tree.find(".//continuous_target").text == "True"
            all_obs = tree.find(".//dataset").findall("obs")
            for i,obs in enumerate(all_obs):
                self.filenames.append(obs.find(".//filename_uid").text)
                self.targets[i] = float(obs.find(".//target").text)

        if self.slice is not None:
            self.filenames = self.filenames[self.slice]
            self.targets = self.targets[self.slice]
        self.n_obs = len(self.filenames)
        return


    def view_by_parameter(self, callables, names, cap_points=1500, color_by_target=True, save_to_disk=False):
        """
        A general viewing utility of extractable features from the observations of this dataset

        :param callables: (tuple of callables) invoking callable[i] on an observation tensor from self.__getitem__()
                          should return a float
        :param names: (tuple of str) names[i] describes the feature of callables[i]
        :param cap_points: (int) max amount of observations to extract features from. If not None, a random subset of
                           this dataset is selected for viewing
        :param color_by_target: (bool) whether the figure should color its points according to a colorscale of the target
                                distribution
        :param save_to_disk: (bool) whether the figure should be immediately viewed or saved permanently to disk
        :return: (tensor) features shape=(cap_points, len(callables)) dtype=torch.float32
        """
        if isinstance(callables, tuple):
            if len(callables) > 3: raise Exception("Cannot visualize more than 3 parameters at a time")
        else:
            raise Exception("callables parameter should be a tuple")

        assert len(callables) == len(names)

        if cap_points is not None and cap_points < len(self):
            # Select a random subset
            ids = torch.randperm(len(self))[:cap_points]
        else:
            ids = torch.arange(0, len(self))

        # Extract all the features by invoking the callables
        dims = len(callables)
        features = torch.zeros(ids.shape[0], dims)
        filenames = []
        for i, id in enumerate(ids):
            observation = self[id][0]
            filenames.append(self.filenames[id])
            for j in range(dims):
                features[i, j] = callables[j](observation)

        # Display the figures according to provided dims and color of target
        if color_by_target:
            colors = self.targets[ids]
        else:
            colors = torch.zeros(ids.shape[0])

        title = self.desc + "-" + "-".join(names) + "-scatter"

        hovertemplate = "%{text}"
        for var, dim_name in zip(('x', 'y', 'z')[:dims], names):
            hovertemplate += '<br>' + dim_name + ': %{' + var + ':.4f}'

        if color_by_target:
            texts = ['id: {idval}<br>filename: {file}<br>target: {target}'.format(idval=id, file=self.filenames[id],
                                                                                  target=self.targets[id]) for id in ids]
        else:
            texts = ['id: {idval}<br>filename: {file}'.format(idval=id, file=self.filenames[id]) for id in ids]

        if dims == 3:
            fig = go.Figure(go.Scatter3d(x=features[:, 0], y=features[:, 1], z=features[:2], mode='markers',
                                         hovertemplate=hovertemplate, text=texts,
                                         marker=dict(color=colors, colorscale='Viridis', showscale=True)))
            fig.update_layout(title=title, zaxis_title=names[2], yaxis_title=names[1], xaxis_title=names[0],
                              hoverlabel_align='right')
        else:
            if dims == 1:
                y = torch.rand(ids.shape[0])
                ytitle = "Random Dispersion"
            else:
                y = features[:, 1]
                ytitle = names[1]
            fig = go.Figure(go.Scatter(x=features[:, 0], y=y, mode='markers',
                                         hovertemplate=hovertemplate, text=texts,
                                         marker=dict(color=colors, colorscale='Viridis', showscale=True)))
            fig.update_layout(title=title, yaxis_title=ytitle, xaxis_title=names[0], hoverlabel_align='right')

        if save_to_disk:
            fig.write_html(title + ".html")
        else:
            fig.show()



    def view_target_distribution(self, save_to_disk=False):
        """

        :param save_to_disk:
        :return:
        """
        if self.continuous_targets:
            # Violin plot
            fig_violin = go.Figure(data=go.Violin(y=self.targets, box_visible=True, line_color='black',
                                           meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                                           x0='Target Distribution of ' + self.desc))
            fig_violin.update_layout(title="Violin Plot of Target: " + self.desc)

            fig_hist_density = ff.create_distplot([self.targets.numpy()], [self.desc])
            fig_violin.update_layout(title="Histogram Density of Target: " + self.desc)

            if save_to_disk:
                fig_hist_density.write_html("TargetDistributionHistDensity.html")
                fig_violin.write_html("TargetDistributionViolin.html")
            else:
                fig_violin.show()
                fig_hist_density.show()

        else:
            # Categorical frequency plot
            values, counts = torch.unique(self.targets, return_counts=True)
            fig = go.Figure(go.Bar(x=values, y=counts))
            if save_to_disk:
                fig.write_html(".html")
            else:
                fig.show()

