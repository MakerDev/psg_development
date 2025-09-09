import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from .iofiles.config import *

from datetime import datetime, timedelta


DICT_PatientInfo_SubElements = {
    "GUID":"string", "ID": "string", "FirstName": "string", 
    "MiddleName":"string", "LastName":"string", "Prefix":"string", 
    "Street":"string", "City":"string", "State":"string", "ZipCode":"string", 
    "DateOfBirth":"datetime", "Gender":"i4", "HomePhone":"string", 
    "OtherPhone":"string", "EMail":"string", "Notes":"string", 
    "Height":"r8", "Weight":"r8", "CustomProperties":False}

LIST_EventTypes = [
    "ACTIVITY-MOVE", "ANALYSIS-START", "ANALYSIS-STOP", "APNEA", 
    "APNEA-CENTRAL", "APNEA-MIXED", "APNEA-OBSTRUCTIVE", "AROUSAL", 
    "AROUSAL-APNEA", "AROUSAL-DESAT", "AROUSAL-HYPOPNEA", "AROUSAL-LM", 
    "AROUSAL-PLM", "AROUSAL-RERA", "AROUSAL-RESP", "AROUSAL-SNORE", 
    "AROUSAL-SPONT", "AS-APNEA", "AS-BREATH", "AS-HYPOPNEA", "AS-SETTINGS", 
    "AUDIO-DICTATION", "AUTONOMIC", "AUTONOMIC-DESAT", "AUTONOMIC-LM", 
    "AUTONOMIC-RESP", "AUTONOMIC-RESPDESAT", "AUTONOMIC-SPONT", "AV-CLIP", 
    "AV-MARK-IN", "AV-MARK-OUT", "BATTERY_CRITICAL", "BATTERY_LOW", 
    "BIOCAL-EYES-BLINK", "BIOCAL-EYES-CLOSED", "BIOCAL-EYES-OPEN", 
    "BIOCAL-FLOW-DEEP", "BIOCAL-FLOW-HOLD", "BIOCAL-FLOW-NASAL", 
    "BIOCAL-FLOW-NORMAL", "BIOCAL-FLOW-ORAL", "BIOCAL-LOOK-DOWN", 
    "BIOCAL-LOOK-LEFT", "BIOCAL-LOOK-RIGHT", "BIOCAL-LOOK-UP", 
    "BIOCAL-SNORE-LOUD", "BIOCAL-SNORE-MODERATE", "BIOCAL-SNORE-SOFT", 
    "BIOCAL-TEETH-GRIND", "BIOCAL-TOES-LEFT-DOWN", "BIOCAL-TOES-RIGHT-DOWN", 
    "BUTTON", "BUTTON-BU", "Bruxism", "CAP-A1", "CAP-A2", "CAP-A3", "CAP-B", 
    "CAP-NCAP", "DESAT", "DEVICE-CALIBRATION-OFF", "DEVICE-CALIBRATION-ON", 
    "DEVICE-CONNECT", "DEVICE-CONNECT-BU", "DEVICE-CONNECT-PU", 
    "DEVICE-DISCONNECT", "DEVICE-DISCONNECT-BU", "DEVICE-DISCONNECT-PU", 
    "DISK_FULL", "DISPLAY-CHANGE-RECORDING", "DISPLAY-CHANGE-SCORING", 
    "EEG-EYES-CLOSED", "EEG-EYES-OPEN", "EEG-HYPERVENTILATION-START", 
    "EEG-HYPERVENTILATION-STOP", "EEG-MOVING", "EEG-PHOTICSIMULATION-START", 
    "EEG-PHOTICSIMULATION-STOP", "EEG-POSTHYPERVENTILATION-START", 
    "EEG-POSTHYPERVENTILATION-STOP", "EEG-SWALLOW", "EEG-TALKING", 
    "EKG-AFIBRILLATION", "EKG-ARRHYTHMIA", "EKG-ASYTOLE", "EKG-BRADYCARDIA", 
    "EKG-NCTACHYCARDIA", "EKG-STACHYCARDIA", "EKG-TACHYCARDIA", 
    "EKG-WCTACHYCARDIA", "EMG-ALMA", "EMG-EFM", "EMG-HFT", "EMG-PhasicBrux", 
    "EMG-PhasicRBD", "EMG-RMD", "EMG-TonicBrux", "EMG-TonicRBD", 
    "EQUIPMENT-AUTORESUME", "EQUIPMENT-INTERRUPT", "EQUIPMENT-NOSTORAGE", 
    "EQUIPMENT-OUTOFSTORAGE", "EXPORT-START", "EXPORT-STOP", "FLOWGENERATOR-APAP", 
    "FLOWGENERATOR-ASV", "FLOWGENERATOR-CPAP", "FLOWGENERATOR-OPTIMAL-START", 
    "FLOWGENERATOR-OPTIMAL-STOP", "FLOWGENERATOR-START", "FLOWGENERATOR-STOP", 
    "FLOWGENERATOR-VAUTO", "FLOWGENERATOR-VPAP", "HEARTRATE-INCREASE", "HYPOPNEA", 
    "HYPOPNEA-CENTRAL", "HYPOPNEA-MIXED", "HYPOPNEA-OBSTRUCTIVE", 
    "IMPEDANCE-ERROR", "IMPEDANCE-FAILED", "IMPEDANCE-PASSED", 
    "IMPEDANCE-REFERENCE", "IMPEDANCE-START", "LIGHTS-OFF", "LIGHTS-ON", 
    "MACHINECAL-INVALID", "MACHINECAL-VALID", "MCAP-A1", "MCAP-A2", "MCAP-A3", 
    "OXYGEN-FLOW", "PEDIATRIC-RESP-BASELINE", "PEDSLEEP-AS", "PEDSLEEP-IS", 
    "PEDSLEEP-QS", "PEDSLEEP-W", "PHOTIC-FLASH", "PHOTIC-SEQUENCE-START", "PLM", 
    "PLM-LM", "PLM-RRLM", "POSITION-LEFT", "POSITION-PRONE", "POSITION-RIGHT", 
    "POSITION-SUPINE", "POSITION-UNKNOWN", "POSITION-UPRIGHT", "POWER-LOSS", 
    "PRINT-START", "PRINT-STOP", "RESERVED-SELECTION", "RESP-BREATH-APNEIC", 
    "RESP-BREATH-CHEYNESTOKES", "RESP-BREATH-CRESC", "RESP-BREATH-CSCYCLE", 
    "RESP-BREATH-DECRESC", "RESP-BREATH-E", "RESP-BREATH-FL", 
    "RESP-BREATH-HYPERPNEIC", "RESP-BREATH-HYPOVENTILATION", "RESP-BREATH-I", 
    "RESP-BREATH-PERIODICBREATHING", "RESP-BREATH-SIGH", "RESP-FLI", 
    "RESP-MOVEMENT-PARADOXICAL", "RESP-MOVEMENT-STOP", "RESP-RERA", "RESP-RMI", 
    "ROOMAIR-START", "ROOMAIR-STOP", "SENSOR-ALARM", "SIGNAL-ARTIFACT", 
    "SIGNAL-BASELINE", "SIGNAL-IMPEDANCE", "SIGNAL-QUALITY-LOW", "SLEEP-KCOMPLEX", 
    "SLEEP-MT", "SLEEP-N", "SLEEP-REM", "SLEEP-RM", "SLEEP-S0", "SLEEP-S1", 
    "SLEEP-S2", "SLEEP-S3", "SLEEP-S4", "SLEEP-SM", "SLEEP-SPINDLE", 
    "SLEEP-SPINDLE-BLOCK", "SLEEP-SPINDLE-SYNC", "SLEEP-TNREM", "SLEEP-TREM", 
    "SLEEP-UNSCORED", "SNORE", "SNORE-SINGLE", "STICKYNOTE", "TEXT", "UARS", 
    "VIDEO-CUT-START", "VIDEO-CUT-STOP"]

def gen_event(Events, st, stage_idx, loc='Pos.Angle-Gravity'):

    stagemap ={ 
        0:'SLEEP-S0', 1:'SLEEP-REM', 
        2:'SLEEP-S1', 3:'SLEEP-S2', 4:'SLEEP-S3' }

    Event = ET.SubElement(Events, 'Event')

    Type       = ET.SubElement(Event, 'Type')
    Type.set('dt:dt', 'string')
    Type.text = stagemap[stage_idx]

    Location   = ET.SubElement(Event, 'Location')
    Location.set('dt:dt', 'string')
    Location.text = loc

    StartTime  = ET.SubElement(Event, 'StartTime')
    StartTime.set('dt:dt', 'string')
    StartTime.text = st.strftime("%Y-%m-%dT%H:%M:%S.%f")

    StopTime   = ET.SubElement(Event, 'StopTime')
    StopTime.set('dt:dt', 'string')
    StopTime.text = (st+timedelta(seconds=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f")

    Parameters0 = ET.SubElement(Event, 'Parameters')
    Parameter0 = ET.SubElement(Parameters0, 'Parameter')
    Key0 = ET.SubElement(Parameter0, 'Key')
    Key0.set('dt:dt', 'string')
    Key0.text = '1'

    Val0 = ET.SubElement(Parameter0, 'Value')
    Parameters1 = ET.SubElement(Val0, 'Parameters')

    Parameter1 = ET.SubElement(Parameters1, 'Parameter')
    Key1 = ET.SubElement(Parameter1, 'Key')
    Key1.set('dt:dt', 'string')
    Key1.text = 'is manual'
    Val1 = ET.SubElement(Parameter1, 'Value')
    Val1.set('dt:dt', 'boolean')
    Val1.text = '1'
    Parameter2 = ET.SubElement(Parameters1, 'Parameter')
    Key2 = ET.SubElement(Parameter2, 'Key')
    Key2.set('dt:dt', 'string')
    Key2.text = 'supertype'
    Val2 = ET.SubElement(Parameter2, 'Value')
    Val2.set('dt:dt', 'string')
    Val2.text = 'SLEEP'

    Parameter3 = ET.SubElement(Parameters1, 'Parameter')
    Key3 = ET.SubElement(Parameter3, 'Key')
    Key3.set('dt:dt', 'string')
    Key3.text = 'type'
    Val3 = ET.SubElement(Parameter3, 'Value')
    Val3.set('dt:dt', 'string')
    Val3.text = stagemap[stage_idx]

def record_sn_results_into_xml(basetime:datetime, y_pred:np.array):
    """[summary]
    Record StageNet result into xml using ElementTree.
    The format of the xml file was defined by Embla.

    Arguments:
        basetime {datetime.datetime} -- Datetime when recording started.
        y_pred   {numpy.array} -- Results numpy array generated by StageNet. 
    """

    time_format = "%Y-%m-%dT%H:%M:%S.%f"

    "ROOT"
    EventExport = ET.Element('EventExport')
    EventExport.set('xmlns:dt', 'urn:schemas-microsoft-com:datatypes')

    "PatientInfo"
    PatientInfo = ET.SubElement(EventExport, 'PatientInfo')


    PatientInfo_SubElements = {
        key: ET.SubElement(PatientInfo, key)
        for key, dt in DICT_PatientInfo_SubElements.items() }

    for key, dt in DICT_PatientInfo_SubElements.items():
        if dt: PatientInfo_SubElements[key].set('dt:dt', str(dt))
    PatientInfo_SubElements['GUID'].text = 'd5c2d2a3-66bf-4366-ade6-267d84cd6973'

    "ExportDate"
    ExportDate  = ET.SubElement(EventExport, 'ExportDate')
    ExportDate.set('dt:dt', 'datetime')
    ExportDate.text = datetime.now().strftime(time_format)[:-3]

    "EventTypes"
    EventTypes  = ET.SubElement(EventExport, 'EventTypes')
    for event_type in LIST_EventTypes:
        e_type = ET.SubElement(EventTypes, 'EventType')
        e_type.set('dt:dt', 'string')
        e_type.text = event_type

    Events = ET.SubElement(EventExport, 'Events')
    for i, y_idx in enumerate(y_pred):
        gen_event(Events, basetime+timedelta(seconds=30*i), y_idx)

    return ET.tostring(EventExport)


def export_to_xml(initial_timestamp, predictions, output_path):
    pred_to_desc = {value: key for key, value in EVENT_MAP['STAGENET'].items()}

    annotationlist = ET.Element("annotationlist")
    
    current_timestamp = initial_timestamp
    
    for pred in predictions:
        annotation = ET.SubElement(annotationlist, "annotation")
        
        onset = ET.SubElement(annotation, "onset")
        onset.text = current_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
        
        duration = ET.SubElement(annotation, "duration")
        duration.text = "30.000000"
        
        desc = ET.SubElement(annotation, "description")
        desc.text = pred_to_desc[int(pred)]
        
        current_timestamp += timedelta(seconds=30)
    
    rough_string = ET.tostring(annotationlist, encoding="utf-8")
    parsed = parseString(rough_string)
    beautified_xml = parsed.toprettyxml(indent="  ")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(beautified_xml)


def to_xml_legacy(path_xml:str, basetime:datetime, y_pred:np.array):
    """[summary]
    - Record StageNet results in XML file and Save the recorded XML file.

    Arguments:
        path_xml {str}   -- Result XML file path.
        basetime {datetime.datetime} -- Datetime when recording started.
        y_pred   {numpy.array} -- Results numpy array generated by StageNet. 
    """

    "Record StageNet resutls into XML file"
    recoreded_xml = record_sn_results_into_xml(basetime, y_pred)

    "Save the recoreded xml file."
    with open(path_xml, 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>')
        f.write(recoreded_xml)

def to_xml(path_xml:str, basetime:datetime, y_pred:np.array):
    """[summary]
    - Record StageNet results in XML file and Save the recorded XML file.

    Arguments:
        path_xml {str}   -- Result XML file path.
        basetime {datetime.datetime} -- Datetime when recording started.
        y_pred   {numpy.array} -- Results numpy array generated by StageNet. 
    """

    "Record StageNet resutls into XML file"
    export_to_xml(basetime, y_pred, path_xml)
