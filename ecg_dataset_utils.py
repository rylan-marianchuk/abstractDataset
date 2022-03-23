import torch

def augment_to_12_leads(ecg_8_lead):
    """
    Compute the 4 remaining leads in the 12 leads Electrocardiogram
    Reference for linear combination correctness:
    :param ecg_8_lead: (tensor) shape=(8, 5000) eight lead Electrocardiogram (8 leads acquired)
    :return: (tensor) shape=(12, 5000)
    """
    #TODO verify assumption that leads are ordered I, II, V1, ..., V6

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