import xml.etree.ElementTree as ET

from os import listdir
from os.path import join
from datetime import datetime
import datetime as dt

from .iofiles import edf as edf_io
from .iofiles import xml as xml_io
from .iofiles.config import *

class DataSet:
    def __init__(self, dir_x, dir_y, ext_x, ext_y, 
        missing_ch   = 'raise',
        multi_buffer = False):

        assert missing_ch in ['raise', 0, 1, 2], \
            'missing_ch can be raise or handling not {}'.format(missing_ch)
            
        self.multi_buffer = multi_buffer
        self.missing_ch   = missing_ch
        self._dir_x       = dir_x
        self._dir_y       = dir_y
        self._ext_x       = ext_x
        self._ext_y       = ext_y

        self._keys        = self.__build_keys()
        
        self._buffer_x    = dict() 
        self._buffer_y    = dict()



    def __build_keys(self):

        set_x = set(self.__find_keys(self._dir_x, self._ext_x))
        ext_y = set(self.__find_keys(self._dir_y, self._ext_y))

        return list( set_x & ext_y )

    def _check_key(self, key):

        if key not in self._keys: 
            raise ValueError("Invalid key: {}".format(key))

    def load_edf(self, key, 
        preload=False, resample=False, preset=False, exclude=True):
        
        self._check_key(key)
        if key in self._buffer_x.keys():
            return self._buffer_x[key]

        path = join(self._dir_x, '{}{}'.format(key, self._ext_x))

        edf, n_missing_ch = edf_io.load(
                                    path       = path, 
                                    preload    = preload, 
                                    resample   = resample, 
                                    preset     = preset, 
                                    exclude    = exclude,
                                    missing_ch = self.missing_ch,)

        self.n_missing_ch = n_missing_ch

        if self.multi_buffer:
            self._buffer_x[key] = edf
        else: 
            self._buffer_x = {key: edf}

        return edf

    def load_xml(self, key):
        """Load the XML file and parse it into a usable format."""
        xml_path = f"{self._dir_y}/{key}{self._ext_y}"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        events = []
        for annotation in root.findall("annotation"):
            onset = annotation.find("onset").text
            duration = annotation.find("duration").text
            description = annotation.find("description").text

            if '-U' in description:
                location = "EEG-F4"
            else:
                location = annotation.find("location").text
            # location = "EEG-F4"

            events.append({
                "onset": self.conv_time(onset),
                "duration": float(duration),
                "description": description,
                "location": location
            })
        
        self._buffer_y[key] = events

    def keys(self):

        return self._keys 

    @staticmethod
    def __find_keys(dir_, ext):

        return [ file_.replace(f'{ext}','') 
            for file_ in listdir(dir_) 
            if file_.endswith(f'{ext}') ]
    
    @staticmethod
    def conv_time(str_time):
        return datetime.strptime(str_time,"%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def fill_na(events):
        n_filled_events = 0
        new_events = [events[0]]
        prev_e = events[0]

        for curr_e in events[1:]:

            if prev_e['e_sec'] != curr_e['s_sec']:
                distance = int((curr_e['s_sec'] - prev_e['e_sec'])//30)
                for i in range(distance):
                    new_events.append({
                        'annotation': prev_e['annotation'], 
                        's_sec': prev_e['e_sec']+(i*30), 
                        'e_sec': prev_e['e_sec']+(i*30)+30})
                    
                    n_filled_events += 1
            
            new_events.append(curr_e)
            prev_e = curr_e 
        
        return new_events, n_filled_events

class SCH(DataSet):

    def __init__(self, 
        root_edf     = '', 
        root_xml     = '', 
        ext_x        = '.edf',
        ext_y        = '.xml',
        missing_ch   = 'raise',
        multi_buffer = False):

        super(SCH, self).__init__(
            dir_x        = root_edf, 
            dir_y        = root_xml,
            ext_x        = ext_x,
            ext_y        = ext_y, 
            missing_ch   = missing_ch,
            multi_buffer = multi_buffer 
        )

    def load_events(self, key, preset, fill_na=True):
        
        self._check_key(key)

        if key not in self._buffer_x.keys():
            self.load_edf(key, preload=False)

        # meas_date = datetime.utcfromtimestamp(
        #     self._buffer_x[key].info['meas_date'][0])
        meas_date = self._buffer_x[key].info['meas_date'].replace(tzinfo=None)

        if key not in self._buffer_y.keys():
            self.load_xml(key)
        
        doc = self._buffer_y[key]

        eventm = self.event_map(doc, preset)
        events = self.extract_event(eventm, doc)
        events = self.datetime_to_sec_events(events, meas_date)
        if fill_na:
            events, n_filled = self.fill_na(events)
            self.n_filled = n_filled
        return events
    
    @staticmethod
    def search_event(keyword, doc):
        return list(set([ 
            event['description']
            for event in doc
            if keyword in event['description']
        ]))

    @staticmethod
    def extract_event(eventm, doc):
        return [{
            "annotation": eventm[event['description']],
            "onset": event['onset'],
            "duration": event['onset'] + dt.timedelta(seconds=event['duration'])}
            for event in doc if event['description'] in eventm.keys()]

    @staticmethod
    def event_map(doc, preset):
        events = list(set([event['description'] for event in doc]))
        eventm = {
            event: key
            for event in events
            for key, candidates in EVENT_TABLE[preset].items()
            if event in candidates and key in EVENT_PRESET[preset]
        }

        return eventm

    @staticmethod
    def datetime_to_sec_events(events, meas_date):
        meas_date = meas_date.replace(tzinfo=None)
        result = []
        for event in events:
            s_sec = (event['onset'] - meas_date).total_seconds()
            e_sec = (event['duration'] - meas_date).total_seconds()
            annotation = event['annotation']
            result.append(event)


        return [{
            "annotation": event["annotation"],
            "s_sec": (event["onset"] - meas_date).total_seconds(),
            "e_sec": (event["duration"] - meas_date).total_seconds()}
            for event in events]
