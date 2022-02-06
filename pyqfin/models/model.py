from abc import ABC


class ParametersBase(ABC):
    pass


class ParameterDependant(ABC):

    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self._add_getters()

    def _add_getters(self):
        for k, v in vars(self.params).items():
            setattr(self, k, lambda v=v: v)


class AnalyticBase(ParameterDependant):

    def __init__(self, params) -> None:
        super().__init__(params)


class SimulationBase(ParameterDependant):
    def __init__(self, params) -> None:
        super().__init__(params)
