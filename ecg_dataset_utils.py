import torch
import h5py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

def augment_to_12_leads(ecg_8_lead):
    """
    Compute the 4 remaining leads in the 12 leads Electrocardiogram
    Reference for linear combination correctness: Malmivuo, Jaakko & Plonsey, Robert. (1975). Bioelectromagnetism. 15. 12-Lead ECG System.
    :param ecg_8_lead: (tensor) shape=(8, 5000) eight lead Electrocardiogram (8 leads acquired)
    :return: (tensor) shape=(12, 5000)
    """

    # Compute the final leads
    # III
    lead_III = ecg_8_lead[1] - ecg_8_lead[0]
    # aVL
    lead_aVL = (ecg_8_lead[0] - lead_III) / 2
    # aVR
    lead_aVR = -(ecg_8_lead[0] + ecg_8_lead[1]) / 2
    # aVF
    lead_aVF = (lead_III + ecg_8_lead[1]) / 2

    ecg12 = torch.vstack((ecg_8_lead, lead_III, lead_aVL, lead_aVR, lead_aVF))
    return ecg12


def get_tensor_from_filename(euid_filename, n_leads=8, median=False, given_data_path=None):
    """
    Return the ecg as a PyTorch tensor given a filename
    :param euid_filename: (str) of the h5 filename NOT a path
    :param n_leads: (int) if 12, the four remaining leads are computeda
    :param median: (bool) whether to view the median QRS, or if False, the entire 10 second rhythm
    :param given_data_path: (str) path where to read the filename from
    :return: (tensor) shape=(n_leads, 5000) where each row is a lead of the ecg in the following order:
             "I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "III", "aVL", "aVR", "aVF"
    """
    path2file = os.getenv("ECGS_PATH")
    if not median:
        path2file += "rhythm/"
        shape = 5000
    else:
        path2file += "median/"
        shape = 600

    if given_data_path is not None:
        path2file = given_data_path

    path2file += euid_filename

    f = h5py.File(path2file)
    np_ecg = np.array(f["ECG"])
    ecg = torch.from_numpy(np_ecg).view(8, shape)
    if n_leads == 12:
        return augment_to_12_leads(ecg)
    return ecg


def viewECG(filename, target=None, n_leads=8, lead_id=None, median=False, given_data_path=None, save_to_disk=False):
    """
    Utility for easily viewing an Electrocardiogram
    :param filename: (str) of the filename NOT a path
    :param target: (numeric) the target of the ECG
    :param n_leads: (int) 8 or 12
    :param lead_id: (int or str) if viewing only a single lead is desired, str must be in lead_names defined below
    :param median: (bool) whether to view the median QRS, or if False, the entire 10 second rhythm
    :param given_data_path: (str) path where to read the filename from
    :param save_to_disk: (bool) whether the viewable object should be saved as a file and not temporarily viewed in browser
    :return: None
    """

    lead_names = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6") if n_leads == 8 \
        else ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "III", "aVL", "aVR", "aVF")

    ecg = get_tensor_from_filename(filename, n_leads, median, given_data_path)

    if lead_id is not None:
        if isinstance(lead_id, str):
            lead_id = lead_names.index(lead_id)
        signal = ecg[lead_id]
        fig = go.Figure(go.Scatter(y=signal, mode='markers', marker=dict(color='red')))

        title = "Electrocardiogram " + "<br>File:  " + filename + "<br>Lead " + str(lead_id) + " :  " + lead_names[lead_id] + "<br>Target:  " + str(target)

        fig.update_layout(
            title=title,
            yaxis_title="Amplitude (mV)",
            xaxis_title="Sample Number"
        )

        if save_to_disk:
            fig.write_html("ECG-lead-" + lead_names[lead_id] + "-" + filename.split(".")[0] + ".html")
        else:
            fig.show()
        return

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=lead_names)

    for i, rc in enumerate(((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2))):
        r, c = rc
        fig.add_trace(go.Scatter(y=ecg[i], mode='markers', marker=dict(color='red')), row=r, col=c)

    master_title = "8 Lead Electrocardiogram " + "<br>File:  " + filename  + "<br>Target:  " + str(target)

    fig.update_layout(title_text=master_title)

    if save_to_disk:
        fig.write_html("8-LeadECG-" + filename.split(".")[0] + ".html")
    else:
        fig.show()

    if n_leads == 12:
        fig_computed_leads = make_subplots(
            rows=2, cols=2,
            subplot_titles=lead_names[-4:])

        for i, rc in enumerate(((1, 1), (1, 2), (2, 1), (2, 2))):
            r, c = rc
            fig_computed_leads.add_trace(go.Scatter(y=ecg[i+8], mode='markers', marker=dict(color='red')), row=r, col=c)

        master_title = "Computed Leads (12 Lead ECG) " + "<br>File:  " + filename

        fig_computed_leads.update_layout(title_text=master_title)

        if save_to_disk:
            fig_computed_leads.write_html("ComputedLeads-" + filename.split(".")[0] + ".html")
        else:
            fig_computed_leads.show()
    return