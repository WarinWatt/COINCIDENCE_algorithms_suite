__all__ = ["ExperimentConfiguration", "ExperimentResult", "ExperimentRunner"]


def __getattr__(name):
    """Keep optional pymoo dependencies lazy for lightweight problem APIs."""
    if name == "ExperimentConfiguration":
        from .configuration import ExperimentConfiguration
        return ExperimentConfiguration
    if name in ("ExperimentResult", "ExperimentRunner"):
        from .runner import ExperimentResult, ExperimentRunner
        return {"ExperimentResult": ExperimentResult, "ExperimentRunner": ExperimentRunner}[name]
    raise AttributeError(name)
