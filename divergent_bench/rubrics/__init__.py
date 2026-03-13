def __getattr__(name):
    if name == "DATRubric":
        from .dat_rubric import DATRubric
        return DATRubric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DATRubric"]
