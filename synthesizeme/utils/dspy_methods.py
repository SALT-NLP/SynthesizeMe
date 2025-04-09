import dspy
import os

from synthesizeme.teleprompt.random_search import BootstrapFewShotWithRandomSearchFast
from synthesizeme.utils.utils import exact_match, repeat_dspy_call
from typing import Literal

DEFAULT_PERSONA_PROMPT = """Given a conversation and two completions from different models, alongside some prior judgements and a user persona, determine which completion the human judge is more likely to prefer.  Use any provided context as well as the provided persona to speculate about the personal preferences of the judge.  You are a personalized reward model for this user, so think carefully about what this user will like."""

class LLMAsAJudge(dspy.Signature):
    """Given a conversation and two completions from different models, determine which completion the human judge is more likely to prefer.  Use any provided context to learn about the personal preferences of the judge before making a decision.  If no context is provided it can be useful to speculate about the preferences of the judge.  It's okay to be wrong, let's explore the space of possibilities and hypothesize about what might be true.  Please hypothesize between 1-3 speculations about the judge's preferences or persona when reasoning.  Draw from the context of the conversation and the completions as well as the user written statements to make your decision."""
    conversation: str = dspy.InputField(desc="The conversation context leading up to the completions.")
    first_completion: str = dspy.InputField(desc="The first of the two possible completions to judge between.")
    second_completion: str = dspy.InputField(desc="The second of the two possible completions to judge between.")
    preference: Literal['First', 'Second'] = dspy.OutputField(desc="The completion that the judge is more likely to prefer.  Possible values are 'First' and 'Second'.")

class LLMAsAJudgeProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(LLMAsAJudge)

    def forward(self,conversation, completion_one, completion_two):
        prediction = None
        prediction = repeat_dspy_call(self.judge, n=4, conversation=conversation, first_completion=completion_one, second_completion=completion_two)
        if prediction is None:
            # dspy.settings.lm.inspect_history(n=2)
            return dspy.Prediction(reasoning="Error", preference="Tie", output="Error")
        
        if "First" in prediction.preference and not "Second" in prediction.preference:
            return dspy.Prediction(reasoning=prediction.reasoning, preference="First")
        elif "Second" in prediction.preference and not "First" in prediction.preference:
            return dspy.Prediction(reasoning=prediction.reasoning, preference="Second")
        else:
            return dspy.Prediction(reasoning=prediction.reasoning, preference="Tie")

class LLMAsAJudgePersonaInformed(dspy.Signature):
    """Given a conversation and two completions from different models, alongside some prior judgements and a user persona, determine which completion the human judge is more likely to prefer.  Use any provided context as well as the provided persona to speculate about the personal preferences of the judge.  You are serving as a personalized reward model for this user, so think carefully about what this user will like."""
    conversation:str = dspy.InputField(desc="The conversation context leading up to the completions.")
    first_completion:str = dspy.InputField(desc="The first of the two possible completions to judge between.")
    second_completion:str = dspy.InputField(desc="The second of the two possible completions to judge between.")
    preference:Literal['First', 'Second'] = dspy.OutputField(desc="The completion that the judge is more likely to prefer.  Possible values are 'First' and 'Second'.")

class SynthesizePersona(dspy.Signature):
    """Given a set of user judgements on prior conversations, as well as reasoning for those judgements, concisely build a user persona that can be used to describe the preferences of this person and anything we might know about them."""
    past_judgements:str = dspy.InputField(desc="A set of user judgements on prior conversations alongside reasoning for those judgements.")
    synthesized_persona:str = dspy.OutputField(desc="A synthesized user persona that can be used to inform future judgements.")

class GeneratePersonaProgram(dspy.Module):
    def __init__(self, output_dir=None, max_bootstrapped_demos=2147483647, max_labeled_demos=4, num_threads_inner=1, num_candidates=10, stop_at_score=80.0, num_workers_bootstrap=24):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizePersona)
        self.output_dir = output_dir
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_threads_inner = num_threads_inner
        self.num_candidates = num_candidates
        self.stop_at_score = stop_at_score
        self.num_workers_bootstrap = num_workers_bootstrap
        
    def forward(self, user_train, user_val, user_id):

        # Bootstrap reasoning to build up a user persona from training data
        bootstrap_optimizer = BootstrapFewShotWithRandomSearchFast(
            max_bootstrapped_demos=min(self.max_bootstrapped_demos, len(user_train)),
            max_labeled_demos=min(self.max_labeled_demos, len(user_train)),
            num_candidate_programs=self.num_candidates, # TODO: We may want to tweak this parameter
            num_threads=self.num_threads_inner,
            stop_at_score=self.stop_at_score,
            metric=exact_match,
            num_workers=self.num_workers_bootstrap,
        )

        program = LLMAsAJudgeProgram()

        optimized = bootstrap_optimizer.compile(program, trainset=user_train, valset=user_val, restrict=range(-2, self.num_candidates))

        # History of demos and reasoning
        history = []

        demos_list = optimized.judge.predict.demos
        if len(demos_list) == 0:
            print(f"User: {user_id} found no useful demos.  Using all examples.")
            demos_list = user_train

        for demo in optimized.judge.predict.demos:
            if "reasoning" in demo:
                conversation = f"""================================================\n**Conversation**: {demo['conversation']}\n===
**Completion One (FIRST COMPLETION)**: «{demo['first_completion']}»\n===
**Completion Two (SECOND COMPLETION)**: «{demo['second_completion']}»\n===
**Reasoning**: «{demo['reasoning']}»\n===
**TRUE USER PREFERENCE**: {demo['preference']}\n================================================\n\n"""
            else:
                conversation = f"""================================================\n**Conversation**: {demo['conversation']}\n===
**Completion One (FIRST COMPLETION)**: «{demo['completion_one']}»\n===
**Completion Two (SECOND COMPLETION)**: «{demo['completion_two']}»\n===
**TRUE USER PREFERENCE**: {demo['chosen']}\n================================================\n\n"""
            history.append(conversation)

        history_str = "\n\n".join(history)

        # Write the history to a file
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            with open(self.output_dir + f"{user_id}.history", "w") as f:
                f.write(history_str)

        # Synthesize a persona from the bootstrapped reasoning
        synthesized_persona = self.synthesize(past_judgements=history_str)

        # Return the synthesized persona
        return dspy.Prediction(persona=synthesized_persona.synthesized_persona, reasoning=synthesized_persona.reasoning)
    
class LLMAsAJudgeProgramPersona(dspy.Module):
    def __init__(self, persona):
        super().__init__()

        custom_prompt = DEFAULT_PERSONA_PROMPT + "\n" +\
            f"The user you are judging completions for has the FOLLOWING PERSONA: ===\n{persona}\n===\n\n" +\
            "Now, given the conversation and two completions, decide which completion the user is more likely to prefer.  Remember to consider the user's persona and preferences as you make your decision."

        PersonalizedSignature = type("PersonalizedSignature", (LLMAsAJudgePersonaInformed,), {"__doc__": custom_prompt})

        self.judge_persona = dspy.ChainOfThought(PersonalizedSignature)
        self.persona = persona

    def forward(self,conversation, completion_one, completion_two):
        prediction = repeat_dspy_call(self.judge_persona, n=4, conversation=conversation, first_completion=completion_one, second_completion=completion_two)

        if prediction is None:
            # dspy.settings.lm.inspect_history(n=2)
            return dspy.Prediction(reasoning="Error", preference="Tie", output="Error")

        if "First" in prediction.preference and not "Second" in prediction.preference:
            return dspy.Prediction(reasoning=prediction.reasoning, preference="First")
        elif "Second" in prediction.preference and not "First" in prediction.preference:
            return dspy.Prediction(reasoning=prediction.reasoning, preference="Second")
        else:
            return dspy.Prediction(reasoning=prediction.reasoning, preference="Tie")