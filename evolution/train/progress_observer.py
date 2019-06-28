from abc import ABC, abstractmethod


class ProgressObserver(ABC):

    @abstractmethod
    def on_progress(self, name: str, cv_idx: int, epoch_idx: int,
                    total_cv: int) -> None:
        pass
