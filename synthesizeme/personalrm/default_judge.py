import random
import pandas as pd
import uuid
import importlib.resources
import os

from synthesizeme.personalrm.personalrm import PersonalRM
from synthesizeme.utils.utils import setup, exact_match, convert_df_to_dspy
from synthesizeme.utils.dspy_methods import GeneratePersonaProgram, LLMAsAJudgeProgramPersona, LLMAsAJudgeProgram
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from platformdirs import user_data_dir

class DefaultJudge(PersonalRM):

    def __init__(self, model_id="litellm_proxy/meta-llama/Llama-3.3-70B-Instruct", model_url="http://localhost:7410/v1", seed=42, lm=None):
        """
        Initialize the SynthesizeMe class.

        Args:
            kwargs: Keyword arguments for the SynthesizeMe class.

        Returns:
            None
        """
        super().__init__()

        if user_id is None:
            user_id = uuid.uuid4().hex

        self.user_id = user_id
        self.model_id = model_id
        self.model_url = model_url
        self.seed = seed
        self.lm = lm

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
    
    def predict_pairwise(self, context: list, option1: dict, option2: dict):
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The options should be dicts containing the completion text.
        """
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        return self.program.predict(context=context, completion_one=option1, completion_two=option2)

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