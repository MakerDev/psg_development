
VIRTUAL_CH_KEY = {
    "F4-M1": [
        "F4-M1","EEG", "F4-A1"
    ], 
    "F3-M2": [
        "F3-M2", "F3-A2", "EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)"
    ], 
    "C4-M1": [
        "C4-M1", "EEG", "C4-A1"
    ], 
    "C3-M2": [
        "C3-M2","EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)", "C3-A2"
    ], 
    "O2-M1": [
        "O2-M1", "EEG", "O2-A1"
    ], 
    "O1-M2": [
        "O1-M2", "EEG(sec)", "EEG2", "EEG sec", "EEG 2", "EEG(SEC)", "O1-A2"
    ], 
    "LOC"  : [
        "LOC", "EOG(L)", "LE", "E1", 'LCO'
    ], 
    "ROC"  : [
        "ROC", "ROC-0", "ROC-1", "EOG(R)", "RE", "E2", 'RCO'
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
    "SLEEP-WAKE":0,
    "SLEEP-REM":1,
    "SLEEP-N1":2,
    "SLEEP-N2":3,
    "SLEEP-N3":4,
}

PRESET = {
    "STAGENET": ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "LOC", "ROC", "EMG"], 
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


EVENT_TABLE = {
    "STAGENET":{
        "SLEEP-W": [
            'Wake|0', "SLEEP-S0", "SLEEP-W"
        ],
        "SLEEP-1": [
            'Stage 1 sleep|1', "SLEEP-S1", "SLEEP-1"
        ],
        "SLEEP-2": [
            'Stage 2 sleep|2', "SLEEP-S2", "SLEEP-2"
        ],
        "SLEEP-3": [
            'Stage 3 sleep|3', 'Stage 4 sleep|4', "SLEEP-S3", "SLEEP-S4", "SLEEP-3", "SLEEP-4"
        ], 
        "SLEEP-R": [
            'REM sleep|5', "SLEEP-REM", "SLEEP-R"
        ]
    }
}

EVENT_PRESET = {
    "STAGENET":["SLEEP-1", "SLEEP-2", "SLEEP-3", "SLEEP-R", "SLEEP-W"]
}

EVENT_MAP = {
    "STAGENET":{
        "SLEEP-W":0, 
        "SLEEP-R":1,
        "SLEEP-1":2, 
        "SLEEP-2":3, 
        "SLEEP-3":4}
}