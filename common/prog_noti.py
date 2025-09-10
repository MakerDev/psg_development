from abc import ABC, abstractmethod


class ProgNoti(ABC):
    """Progress notifier abstract base class."""

    def __init__(self, target, debug: bool = False):
        self._target = target
        self._analId = 0
        self._evtId = 0

        self._curStep = 0
        self._subStep = 0
        self._totalStep = 0

        self._debug: bool = debug

    @property
    def devId(self):
        return self._target

    @property
    def analId(self) -> int:
        return self._analId

    @property
    def eventId(self) -> int:
        return self._evtId

    @property
    def totalStep(self) -> int:
        return self._totalStep

    @property
    def subStep(self) -> int:
        return self._subStep

    @property
    def curStep(self) -> int:
        return self._curStep

    @abstractmethod
    def stepInit(self, analId: int, totalStep: int, msg: str = None):
        self._analId = analId
        self._curStep = 0
        self._subStep = totalStep
        self._totalStep = totalStep

    @abstractmethod
    def stepEnd(self, msg: str = None):
        self._curStep = 0
        self._subStep = 0

    @abstractmethod
    def changeStep(self, analId: int, eventId: int, subStep: int, msg: str = None):
        self._analId = analId
        self._evtId = eventId
        self._curStep = 0
        self._subStep = subStep

    @abstractmethod
    def stepForward(self, msg: str = None):
        self._curStep += 1
