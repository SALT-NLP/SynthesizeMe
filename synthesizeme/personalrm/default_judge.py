import random
import pandas as pd
import uuid
import importlib.resources
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import bootstrap
import numpy as np


from synthesizeme.personalrm.personalrm import PersonalRM
from synthesizeme.utils.utils import setup, exact_match, convert_df_to_dspy
from synthesizeme.utils.dspy_methods import GeneratePersonaProgram, LLMAsAJudgeProgramPersona, LLMAsAJudgeProgram
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from platformdirs import user_data_dir



class DefaultJudge(PersonalRM):

    def __init__(self, model_id="litellm_proxy/meta-llama/Llama-3.3-70B-Instruct", model_url="http://localhost:7410/v1", seed=42, lm=None, num_workers=1):
        """
        Initialize the SynthesizeMe class.

        Args:
            kwargs: Keyword arguments for the SynthesizeMe class.

        Returns:
            None
        """
        super().__init__()

        self.model_id = model_id
        self.model_url = model_url
        self.seed = seed
        self.lm = lm
        self.num_workers = num_workers
        
        if self.lm is None:
            self.lm = setup(model=self.model_id, local_api_base=self.model_url)

        self.program = LLMAsAJudgeProgram()


    def fit(self, data, val_data=None):
        pass

    def predict(self, context: list, completion: dict):
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The completion should be a dict containing the completion text.
        """
        raise NotImplementedError("This method is not defined for DefaultJudge.  DefaultJudge by requires pairwise preferences. Please use the predict_pairwise method instead.")
    
    def predict_pairwise(self, context: list, option1: dict, option2: dict) -> int:
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The options should be dicts containing the completion text.
        """
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        preference = self.program(conversation=context, completion_one=option1, completion_two=option2).preference
        
        if preference == "First":
            return 1
        elif preference == "Second":
            return -1
        else:
            return 0

    def load(self, model_path):
        """
        Load the model from the specified path.
        """
        pass

    def save(self, model_path):
        """
        Save the model to the specified path.
        The model will be saved as a JSON file.
        """
        pass

    def evaluate(self, test_data: list):
        """
        Evaluate the model on the test data.
        """
        if self.num_workers is None:
            self.num_workers = 1

        test_preferences = list(convert_df_to_dspy(pd.DataFrame(test_data)))

        prompts = [example["conversation"] for example in test_preferences]
        chosen = [example["completion_one"] if example["chosen"] == "First" else example["completion_two"] for example in test_preferences]
        rejected = [example["completion_two"] if example["chosen"] == "First" else example["completion_one"] for example in test_preferences]

        with ThreadPoolExecutor(self.num_workers) as executor:
            results = list(executor.map(self.predict_pairwise, prompts, chosen, rejected))

        with ThreadPoolExecutor(self.num_workers) as executor:
            results_reversed = list(executor.map(self.predict_pairwise, prompts, rejected, chosen))

        overall_results = results + [-1 * result for result in results_reversed]
        overall_results = [max(0, result) for result in overall_results]

        confidence_interval = bootstrap((overall_results, ), np.mean, confidence_level=0.95, method='basic')

        return {
            "mean": np.mean(overall_results),
            "lower_bound": confidence_interval.confidence_interval.low,
            "upper_bound": confidence_interval.confidence_interval.high,
            "results": overall_results
        }
