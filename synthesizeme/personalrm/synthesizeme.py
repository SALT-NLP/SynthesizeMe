import random
import pandas as pd
import uuid
import importlib.resources
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import bootstrap
import numpy as np
import tqdm
import threading
from synthesizeme.personalrm.personalrm import PersonalRM
from synthesizeme.utils.utils import setup, exact_match, convert_df_to_dspy
from synthesizeme.utils.dspy_methods import GeneratePersonaProgram, LLMAsAJudgeProgramPersona, LLMAsAJudgeProgram
from synthesizeme.dspy_patch.random_search import BootstrapFewShotWithRandomSearchFast
from synthesizeme.utils.prompts import format_llm_judge_prompt, format_generation_prompt
from platformdirs import user_data_dir

class Unimplemented():
    def forward(**kwargs):
        raise NotImplementedError("Please use the fit method to train the model first.")

class SynthesizeMe(PersonalRM):

    def __init__(self, user_id=None, train_val_ratio=0.7, max_bootstrapped_demos=-1, max_labeled_demos=4, num_search_candidates=10, output_dir=None, model_id="litellm_proxy/meta-llama/Llama-3.3-70B-Instruct", model_url="http://localhost:7410/v1", seed=42, stop_at_score=80.0, num_workers=8, persona_synthesis_program_path=None, lm=None, num_workers_bootstrap=24, progress_update_hook=None):
        """
        Initialize the SynthesizeMe class.

        Args:
            train_preferences (List[Dict]): List of training preferences of the form {"context": List[dict], "chosen": dict, "rejected": dict}
            val_preferences (list): List of validation preferences.  If None, use train_preferences and train_val_ratio to split the data.
            user_id (str): User ID.
            train_val_ratio (float): Ratio of training to validation data.
            max_bootstrapped_demos (int): Maximum number of demonstrations to include in the personalized prompt. If -1, no limit.
            max_labeled_demos (int): Maximum number of labeled demonstrations to include in the personalized prompt. If -1, no limit.
            num_search_candidates (int): Number of search candidates.  These are the number of fewshot combinations that we will try.
            output_dir (str): Output directory.
            model_id (str): Model ID for litellm language model.
            model_url (str): Model URL for litellm language model, typically only needed for locally hosted models.
            seed (int): Random seed.
            stop_at_score (float): Stop the optimization when the score reaches this value.
            num_workers (int): Number of threads to use for the optimization.
            num_workers_bootstrap (int): Number of threads to use for the bootstrapping.
            persona_synthesis_program_path (str): Path to the precomputed persona synthesis program.  If None, will generate a new one.
            lm (dspy.LM): Language model to use.  If None, will use the setup function to create a new one.
            num_workers_bootstrap (int): Number of threads to use for the bootstrapping.
            progress_update_hook (function): A function to call for progress updates.

        Returns:
            None
        """
        super().__init__()

        if output_dir is None:
            data_dir = user_data_dir("synthesizeme")
            data_dir = data_dir + "/users/"

        if persona_synthesis_program_path is None:
            persona_synthesis_program_path = importlib.resources.files("synthesizeme").joinpath("prompts/llama70b.json")

        self.user_id = user_id
        self.train_val_ratio = train_val_ratio
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_search_candidates = num_search_candidates
        self.output_dir = output_dir
        self.model_id = model_id
        self.model_url = model_url
        self.seed = seed
        self.stop_at_score = stop_at_score
        self.num_workers = num_workers
        self.persona_synthesis_program_path = persona_synthesis_program_path
        self.lm = lm
        self.num_workers_bootstrap = num_workers_bootstrap
        self.progress_update_hook = progress_update_hook
        self.last_progress = 0

        # NEW: a lock to prevent race conditions in the progress-bar updates
        self._progress_lock = threading.Lock()

        if self.progress_update_hook is None:
            self.pbar = True
            self.progress_update_hook = self._default_progress_update_hook
        else:
            self.pbar = False

        self.program = Unimplemented()

        if self.lm is None:
            self.lm = setup(model=self.model_id, local_api_base=self.model_url)

    # Default progress update hook (tqdm progress bar)
    def _default_progress_update_hook(self, progress, message):
            if self.pbar is None:
                return

            with self._progress_lock:                    # <<— LOCK GUARD
                delta = progress - self.last_progress
                if delta > 0:
                    self.pbar.update(delta)
                    self.pbar.set_description(message)
                    self.last_progress = progress

                if self.last_progress >= 100 or progress >= 100:
                    self.pbar.close()
                    self.pbar = None

    def fit(self, data, val_data=None):
        rng = random.Random(self.seed)

        if self.pbar:
            self.pbar = tqdm.tqdm(desc="Progress", leave=True, position=0, bar_format="{l_bar}{bar:10}{r_bar}", total=100)

        if self.user_id is None:
            if hasattr(data[0], "user_id"):
                self.user_id = data[0].user_id
            else:
                self.user_id = uuid.uuid4().hex

        if len(data) == 0:
            raise ValueError("No training preferences provided.")
        
        train_preferences = data
        val_preferences = val_data

        max_bootstrapped_demos = self.max_bootstrapped_demos
        if self.max_bootstrapped_demos == -1:
            max_bootstrapped_demos = len(train_preferences)

        max_labeled_demos = self.max_labeled_demos
        if self.max_labeled_demos == -1:
            max_labeled_demos = len(train_preferences)

        # Split the data into training and validation sets
        if val_preferences is None:
            all_preferences = train_preferences
            # rng.shuffle(all_preferences) # don't shuffle because if the user provides the data in temporal order, we want to keep it that way
            split_index = int(len(all_preferences) * self.train_val_ratio)
            train_preferences = all_preferences[:split_index]
            val_preferences = all_preferences[split_index:]

        if len(val_preferences) == 0:
            raise ValueError("Not enough preferences for validation.")

        # First generate the persona
        generate_persona = GeneratePersonaProgram(output_dir=self.output_dir, max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos, num_threads_inner=self.num_workers, num_candidates=self.num_search_candidates, stop_at_score=self.stop_at_score, num_workers_bootstrap=self.num_workers_bootstrap, progress_update_hook=self.progress_update_hook)

        try:
            if self.persona_synthesis_program_path is not None:
                generate_persona.load(self.persona_synthesis_program_path)
        except Exception as e:
            print(f"Error loading persona synthesis program: {e}")
            print("This may be due to incompatibility between the version of dspy you are using and the version of dspy used to generate the program.")

            import dspy
            import json
            print(f"You are using version {dspy.__version__}.")
            program_version = json.load(open(self.persona_synthesis_program_path))['metadata']['dependency_versions']['dspy']
            print(f"The program was generated using version {program_version}.")
            print("You may need to install a compatible version of dspy.")
            raise e

        # TODO: We need to put these train_preferences and val_preferences into the dspy format
        train_preferences = list(convert_df_to_dspy(pd.DataFrame(train_preferences)))
        val_preferences = list(convert_df_to_dspy(pd.DataFrame(val_preferences)))

        persona = generate_persona(user_train=train_preferences, user_val=val_preferences, user_id=self.user_id)

        amount_completed = 0
        def update_progress(score, seed, program):
            nonlocal amount_completed
            amount_completed += 1
            if self.progress_update_hook is not None:
                self.progress_update_hook((amount_completed / (self.num_search_candidates + 3)) * 50.0 + 50.0, f"Tested {amount_completed} potential demo sets with persona.")

        # Now we need to optimize the demonstrations

        bootstrap_optimizer = BootstrapFewShotWithRandomSearchFast(
            max_bootstrapped_demos=min(max_bootstrapped_demos, len(train_preferences)),
            max_labeled_demos=min(max_labeled_demos, len(train_preferences)),
            num_candidate_programs=self.num_search_candidates,
            num_threads=self.num_workers,
            stop_at_score=self.stop_at_score,
            metric=exact_match,
            num_workers=self.num_workers_bootstrap,
            on_eval_complete=update_progress,
        )

        program = LLMAsAJudgeProgramPersona(persona.persona)

        optimized = bootstrap_optimizer.compile(program, trainset=train_preferences, valset=val_preferences)

        self.program = optimized
        self.persona = persona.persona

        return optimized

    def predict(self, context: list, completion: dict):
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The completion should be a dict containing the completion text.
        """
        raise NotImplementedError("This method is not defined for SynthesizeMe.  SynthesizeMe by default requires pairwise preferences. Please use the predict_pairwise method instead or use the SynthesizeMeRM class to get a scalar reward model.")
    
    def predict_pairwise(self, context: list, option1: dict, option2: dict):
        """
        Predict using the fitted model. The context should be a list of dicts, with each dict containing a role and content. The options should be dicts containing the completion text.
        """
        if self.program == Unimplemented():
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        result = self.program(conversation=context, completion_one=option1, completion_two=option2).preference

        if result == "First":
            return 1
        elif result == "Second":
            return -1
        else:
            return 0

    def load(self, path=user_data_dir("synthesizeme") + "/users/"):
        """
        Load a precomputed program from the specified path.
        The program should be a JSON file containing the program.
        """
        if self.user_id is None:
            raise ValueError("No user_id provided. Please provide a user_id to load the program.")
        
        persona = None

        with open(path + self.user_id + "_persona.txt", "r") as f:
            persona = f.read()

        program = LLMAsAJudgeProgramPersona(persona)

        program.load(path + self.user_id + ".json")

        return program
    
    def save(self, path=user_data_dir("synthesizeme") + "/users/"):
        """
        Save the program to the specified path.
        The program will be saved as a JSON file.
        """
        if self.program == Unimplemented():
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + self.user_id + "_persona.txt", "w") as f:
            f.write(self.program.persona)

        self.program.save(path + self.user_id + ".json")

        return path + self.user_id + ".json"
    
    def get_persona(self):
        """
        Get the synthesized persona.
        """
        if self.persona is None:
            raise ValueError("No persona synthesized. Please fit the model first.")

        return self.persona
    
    def get_demos(self):
        """
        Get the optimized demonstrations.
        """
        if self.program == Unimplemented():
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        demos = []
        for example in self.program.judge_persona.predict.demos:
            if example.augmented:
                demos.append({"conversation": example.conversation, "first_completion": example.first_completion, "second_completion": example.second_completion, "reasoning": example.reasoning, "preference": example.preference})
            else:
                demos.append({"conversation": example.conversation, "first_completion": example.first_completion, "second_completion": example.second_completion, "preference": example.preference})

        return demos

    def get_generation_prompt(self):
        """
        Get the personalized generation prompt.
        """
        if self.persona is None:
            raise ValueError("No persona synthesized. Please fit the model first.")
        if self.program == Unimplemented():
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        return format_generation_prompt(self.persona, self.get_demos())

    def get_llm_judge_prompt(self):
        """
        Get the personalized reward model prompt.
        """
        if self.persona is None:
            raise ValueError("No persona synthesized. Please fit the model first.")
        if self.program == Unimplemented():
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        if self.program is None:
            raise ValueError("Model not trained. Please train the model first using the fit method.")
        
        return format_llm_judge_prompt(self.persona, self.get_demos())
    
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
