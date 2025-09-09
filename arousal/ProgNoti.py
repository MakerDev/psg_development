from abc import ABC, abstractmethod

"""
HWP | LWP    HLP | LLP
 0           LP : Total  => PROG_INIT : 전체 스텝 수 전달
 1
"""

class ProgNoti(ABC):

    def __init__(self, target, debug:bool=False):

        self._target = target
        self._analId = 0
        self._evtId  = 0

        self._curStep   = 0
        self._subStep   = 0
        self._totalStep = 0

        self._debug:bool = debug
    #--INIT

    @property
    def devId(self):
        return self._target
    @property
    def analId(self)->int:
        return self._analId
    @property
    def eventId(self)->int:
        return self._evtId

    @property
    def totalStep(self)->int:
        return self._totalStep
    @property
    def subStep(self)->int:
        return self._subStep
    @property
    def curStep(self)->int:
        return self._curStep
    #--DEF


    @abstractmethod
    def stepInit(self, analId:int, totalStep:int, msg:str=None):
        self._analId = analId

        self._curStep = 0
        self._subStep = totalStep
        self._totalStep = totalStep
    #--DEF

    @abstractmethod
    def stepEnd(self, msg:str=None):
        self._curStep = 0
        self._subStep   = 0
    #--DEF

    @abstractmethod
    def changeStep(self, analId:int, eventId:int, subStep:int, msg:str=None):
        self._analId = analId
        self._evtId  = eventId

        self._curStep = 0
        self._subStep = subStep
    #--DEF

    @abstractmethod
    def stepForward(self, msg:str=None):
        self._curStep += 1
    #--DEF

#--CLASS:ProgNoti
