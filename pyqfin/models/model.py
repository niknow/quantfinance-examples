from abc import ABC


class ParametersBase(ABC):
    pass


class AnalyticBase(ABC):

    def __init__(self, params) -> None:
        self.params = params
        self._add_getters()
        super().__init__()

    def _add_getters(self):
        for k, v in vars(self.params).items():
            setattr(self, k, lambda v=v: v)


class SimulationBase(ABC):
    pass
