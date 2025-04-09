import os
import random
import dspy
import pandas as pd

from dotenv import load_dotenv
from synthesizeme.utils.format_conv import format_conversation


def setup(model="azure/gpt-4o-mini-240718", local_api_base="http://localhost:7410/v1"):
    # Load the environment variables
    load_dotenv()
    together_key = os.getenv('TOGETHERAI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')

    azure_key, azure_version, azure_base = os.getenv('AZURE_API_KEY'), os.getenv('AZURE_API_VERSION'), os.getenv('AZURE_API_BASE')

    # Set the seed
    random.seed(42)

    lm = None

    if model.startswith("azure"):
        lm = dspy.LM(model, api_base=azure_base, api_key=azure_key, api_version=azure_version, max_tokens=8192)
    elif model.startswith("openai"):
        lm = dspy.LM(model, api_key=openai_key, tpm=200000, rpm=500, max_tokens=8192)
    elif model.startswith("together"):
        lm = dspy.LM(model, api_key=together_key, max_tokens=8192)
    elif model.startswith("gemini"):
        lm = dspy.LM(model, api_key=gemini_key, max_tokens=8192)
    else:
        model = f"litellm_proxy/{model}"
        lm = dspy.LM(model,
             api_base=local_api_base,
             api_key="local", model_type='chat', max_tokens=8192)

    dspy.configure(lm=lm)

    return lm

def flip_augmentation(dspy_data):
    for example in dspy_data:
        yield example
        yield dspy.Example({
            "conversation": example.conversation,
            "completion_one": example.completion_two,
            "completion_two": example.completion_one,
            "chosen": "First" if example.chosen == "Second" else "Second",
            "index": example.index,
            "conversation_id": example.conversation_id,
            "type": example.type,
            "dataset": example.dataset,
            "user_id": example.user_id
        }).with_inputs("conversation", "completion_one", "completion_two")


def convert_df_to_dspy(df, user_id=None):
    for i, row in df.iterrows():
        flip = row["flip"] if "flip" in row else False
        yield dspy.Example({
            "conversation": format_conversation(row["context"]),
            "completion_one": format_conversation(row["chosen"]) if flip else format_conversation(row["rejected"]),
            "completion_two": format_conversation(row["rejected"]) if flip else format_conversation(row["chosen"]),
            "chosen": "First" if flip else "Second",
            "index": i,
            "conversation_id": row["conversation_id"] if "conversation_id" in row else f"{user_id}_{i}",
            "type": row["type"] if "type" in row else "default",
            "dataset": row["dataset"] if "dataset" in row else "default",
            "user_id": row["user_id"] if "user_id" in row else user_id
        }).with_inputs("conversation", "completion_one", "completion_two")

def convert_user_df_to_dspy(df):
    for user in df["user_id"].unique():
        df_user = df[df["user_id"] == user]
        df_train = df_user[df_user["split"] == "train"]
        df_val = df_user[df_user["split"] == "val"]
        df_test = df_user[df_user["split"] == "test"]

        yield dspy.Example({
            "user_train": list(convert_df_to_dspy(df_train, user)),
            "user_val": list(flip_augmentation(convert_df_to_dspy(df_val, user))),
            "user_test": list(flip_augmentation(convert_df_to_dspy(df_test, user))),
            "user_id": user
        }).with_inputs("user_train", "user_val", "user_id")

def exact_match(gold, pred, trace=None):
    return 1 if gold.chosen == pred.preference else 0

def get_exact_match_func_save():
    outputs = {"index": [], "conversation_id": [], "user_id": [], "type": [], "dataset": [], "reasoning": [], "output": [], "parsed_output": [], "reward_chosen": [], "reward_rejected": [], "correct": []}

    def exact_match_store_output(gold, pred, trace=None):
        outputs["index"].append(gold.index)
        outputs["conversation_id"].append(gold.conversation_id)
        outputs["user_id"].append(gold.user_id)
        outputs["type"].append(gold.type)
        outputs["dataset"].append(gold.dataset)
        outputs["reasoning"].append(pred.reasoning)
        outputs["output"].append(pred.preference)
        outputs["parsed_output"].append(pred.preference)
        outputs["reward_chosen"].append(0)
        outputs["reward_rejected"].append(0)
        outputs["correct"].append(gold.chosen == pred.preference)

        return 1 if gold.chosen == pred.preference else 0
    
    def get_output():
        return pd.DataFrame(outputs)

    return exact_match_store_output, get_output

def repeat_dspy_call(call, n=4, **kwargs):
    preserved_temperature = dspy.settings.lm.kwargs["temperature"]
    results = None
    for i in range(n):
        try:
            dspy.settings.lm.kwargs["temperature"] = preserved_temperature * (0.001 + i / n)
            results = call(**kwargs)
            break # if successful, break
        except Exception as e: 
            print(f"Attempt {i + 1} failed: {e}")
    dspy.settings.lm.kwargs["temperature"] = preserved_temperature
    return results

def exact_match(gold, pred, trace=None):
    return 1 if gold.chosen == pred.preference else 0