from GUI.screens.results_screen import ResultsScreen
from GUI.screens.splash_screens import LoadingScreen, SavingScreen
from GUI.screens.model_screen import ModelScreen
from GUI.screens.process_screen import ProcessScreen
from GUI.screens.training_screen import TrainingScreen

"""
Dictionary containing the high order references to screens in the GUI
"""
Screens = {
    "model": ModelScreen,
    "training": TrainingScreen,
    "process": ProcessScreen,
    "loading": LoadingScreen,
    "saving": SavingScreen,
    "result": ResultsScreen
}
