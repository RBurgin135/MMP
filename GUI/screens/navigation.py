from GUI.screens.result_screen import ResultScreen
from GUI.screens.splash_screens import LoadingScreen, SavingScreen
from GUI.screens.model_screen import ModelScreen
from GUI.screens.process_screen import ProcessScreen
from GUI.screens.training_screen import TrainingScreen


Screens = {
    "model": ModelScreen,
    "training": TrainingScreen,
    "process": ProcessScreen,
    "loading": LoadingScreen,
    "saving": SavingScreen,
    "result": ResultScreen
}
