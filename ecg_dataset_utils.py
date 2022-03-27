import torch
import h5py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

def augment_to_12_leads(ecg_8_lead):
    """
    Compute the 4 remaining leads in the 12 leads Electrocardiogram
    Reference for linear combination correctness:
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


def get_tensor_from_filename(filename, n_leads=8, give_path=None):
    """

    :return:
    """
    if give_path is not None:
        f = h5py.File(give_path + filename)
    else:
        f = h5py.File(os.getenv("ECG_DIR") + filename)
    np_ecg = np.array(f["ECG"])
    ecg = torch.from_numpy(np_ecg).view(8, 5000)
    if n_leads == 12:
        return augment_to_12_leads(ecg)
    return ecg


def viewECG(filename, n_leads=8, lead_id=None, give_path=None, save_to_disk=False):
    """

    :param filename:
    :param save_to_disk:
    :return:
    """
    lead_names = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6") if n_leads == 8 \
        else ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "III", "aVL", "aVR", "aVF")

    ecg = get_tensor_from_filename(filename, n_leads, give_path)

    if lead_id is not None:
        if isinstance(lead_id, str):
            lead_id = lead_names.index(lead_id)
        signal = ecg[lead_id]
        fig = go.Figure(go.Scatter(y=signal, mode='markers', marker=dict(color='red')))

        title = "Electrocardiogram " + "<br>File:  " + filename + "<br>Lead " + str(lead_id) + " :  " + lead_names[lead_id]

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

    master_title = "8 Lead Electrocardiogram " + "<br>File:  " + filename

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