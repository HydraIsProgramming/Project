"""3D GUI package.

Exports are lazy so the interactive app can run without optional training
dependencies (for example stable_baselines3).
"""

__all__ = ["ArmControllerGUI3D", "TrainingGUI3D", "main", "app_main", "training_main"]


def __getattr__(name):
    if name in {"ArmControllerGUI3D", "main", "app_main"}:
        from .app_3d import ArmControllerGUI3D, main as app_main

        if name == "ArmControllerGUI3D":
            return ArmControllerGUI3D
        if name == "main":
            return app_main
        return app_main

    if name in {"TrainingGUI3D", "training_main"}:
        from .training_gui import TrainingGUI3D, main as training_main

        if name == "TrainingGUI3D":
            return TrainingGUI3D
        return training_main

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
