import abc
import torch


class Serializable:
    """
    Base method implementing saving and restoring.

    This works by providing two methods: One produces a state_dict, the other consumes a state
    dict to recreate the state of the class when it was saved.
    """
    @abc.abstractmethod
    def state_dict(self) -> dict:
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass


class EmptySerializable(Serializable):
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict):
        pass


class ChildrenSerializable(Serializable):
    """
    Base method implementing saving and restoring.

    This works by providing two methods: One produces a state_dict, the other consumes a state
    dict to recreate the state of the class when it was saved.
    """
    def key_to_component(self):     # pylint: disable=no-self-use
        return {}

    def own_state_dict(self):       # pylint: disable=no-self-use
        return {}

    def load_own_state_dict(self, state_dict):
        pass

    def state_dict(self) -> dict:
        key_to_comp = self.key_to_component()
        state_dict = {}
        for k, component in key_to_comp.items():
            state_dict[k] = component.state_dict()
        state_dict["state_dict"] = self.own_state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        key_to_comp = self.key_to_component()
        for k, component in key_to_comp.items():
            component.load_state_dict(state_dict[k])
        self.load_own_state_dict(state_dict["state_dict"])


def cur_rng_state():
    rng_dict = {
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            rng_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
        except (RuntimeError, AssertionError):
            rng_dict["cuda_rng_state"] = None
    else:
        rng_dict["cuda_rng_state"] = None
    return rng_dict


def load_rng_state(state_dict):
    torch.set_rng_state(state_dict["rng_state"])
    if state_dict["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
