from abc import ABC, abstractmethod

class PersonalRM(ABC):
    
    @abstractmethod
    def fit(self, data: list) -> None:
        """
        Fit the model to the data.  The data should be a list of user preferences, with each row containing a chosen and rejected completion.
        The data should be in the format {"context": List[dict], "chosen": dict, "rejected": dict}.
        
        Parameters:
        data (list): List of user preferences of the form {"context": List[dict], "chosen": dict, "rejected": dict}.
        """
        pass
    
    @abstractmethod
    def predict(self, context: list, completion: dict) -> float:
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The completion should be a dict containing the completion text.

        Parameters:
        context (list): List of dicts containing the context history.
        completion (dict): The completion to predict.
        
        Returns:
        float: The prediction score for the completion.
        """
        pass

    @abstractmethod
    def predict_pairwise(self, context: list, option1: dict, option2: dict) -> float:
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The options should be dicts containing the completion text.

        Parameters:
        context (list): List of dicts containing the conversation history.
        option1 (dict): The first completion to compare.
        option2 (dict): The second completion to compare.

        Returns:
        bool: True if the first option is preferred over the second, False otherwise.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a precomputed program from the specified path.
        The program should be a JSON file containing the program.

        Parameters:
        path (str): The path to the precomputed program.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        The model will be saved as a JSON file and the persona will be saved as a text file.

        Parameters:
        path (str): The path to save the model.
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: list) -> list:
        """
        Evaluate the model on the test data.  The test data should be a list of user preferences, with each row containing a chosen and rejected completion.
        The data should be in the format {"context": List[dict], "chosen": dict, "rejected": dict}.

        Parameters:
        test_data (list): List of user preferences of the form {"context": List[dict], "chosen": dict, "rejected": dict}.
        """
        pass