VIRTUAL_CH_KEY = {
    "F4-M1": [
        "F4-M1","EEG", "F4-A1", "F4-(M1+M2)"
    ], 
    "F3-M2": [
        "F3-M2", "F3-A2", "EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)", "F3-(M2+M1)"
    ], 
    "C4-M1": [
        "C4-M1","EEG", "C4-A1", "C4-(M1+M2)"
    ], 
    "C3-M2": [
        "C3-M2","EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)", "C3-A2", "C3-(M2+M1)"
    ], 
    "O2-M1": [
        "O2-M1", "EEG", "O2-A1", "O2-(M1+M2)"
    ], 
    "O1-M2": [
        "O1-M2", "EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)", "O1-A2", "O1-(M2+M1)"
    ], 
    "LOC"  : [
        "LOC", "EOG(L)", "LE", "E1", 'LCO'
    ], 
    "ROC"  : [
        "ROC", "ROC-0", "ROC-1", "EOG(R)", "RE", "E2"
    ], 
    "EMG"  : [
        "Chin", "ChinL", "Lower.Left-Uppe", "EMG", "Lower.Left-Upper", "EMG-Submental", "Bipolar 5"
    ]
}

SLEEPSTAGE_TO_LABEL = {
    "SLEEP-U":-1,
    "SLEEP-W":0, 
    "SLEEP-R":1,
    "SLEEP-1":2, 
    "SLEEP-2":3, 
    "SLEEP-3":4,
}

PRESET = {
    "STAGENET": ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "LOC", "ROC", "EMG"], 
    "SHHS": ["O1-M2", "O2-M1", "LOC", "ROC", "EMG"],
}


REPLACE_KEY = {
    "F4-M1": [
        "F3-M2", "C4-M1", "O2-M1", "C3-M2", "O1-M2"
    ], 
    "F3-M2": [
        "F4-M1", "C3-M2", "O1-M2", "C4-M1", "O2-M1"
    ], 
    "C4-M1": [
        "C3-M2", "F4-M1", "O2-M1", "F3-M2", "O1-M2"
    ], 
    "C3-M2": [
        "C4-M1", "F3-M2", "O1-M2", "F4-M1", "O2-M1"
    ], 
    "O2-M1": [
        "O1-M2", "C4-M1", "F4-M1", "C3-M2", "F3-M2"
    ], 
    "O1-M2": [
        "O2-M1", "C3-M2", "F3-M2", "C4-M1", "F4-M1"
    ], 
}
