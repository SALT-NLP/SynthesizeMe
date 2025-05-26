from datasets import load_dataset
from abc import ABC, abstractmethod
import pandas as pd
class Dataset(ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self):
        return self.ds

    def get_dataframe(self, split: str="train"):

        if split == "all":
            return [self.ds[split].to_pandas() for split in self.ds.keys()]

        return self.ds[split].to_pandas()
    
    @abstractmethod
    def get_user_data_df(self, user_id: str):
        pass

    @abstractmethod
    def get_user_data(self, user_id: str):
        pass

    @abstractmethod
    def get_all_user_data(self):
        pass

    @abstractmethod
    def get_user_ids(self):
        pass

class SynthesizeMeDataset(Dataset):
    def __init__(self, hf_dataset_name: str="MichaelR207/chatbot_arena_personalized_0125"):
        self.ds = load_dataset(hf_dataset_name)

        # Drop the unnecessary columns
        self.ds = self.ds.remove_columns(['chosen_score', 'rejected_score', 'dataset', 'query_reasoning', 'personalizable_query', 'response_reasoning', 'personalizable_responses', 'reasoning_gemini/gemini-1.5-pro', 'prediction_flip_gemini/gemini-1.5-pro', 'reasoning_flip_gemini/gemini-1.5-pro', 'reasoning_meta-llama/Llama-3.3-70B-Instruct', 'prediction_flip_meta-llama/Llama-3.3-70B-Instruct', 'reasoning_flip_meta-llama/Llama-3.3-70B-Instruct', 'prediction_gemini/gemini-1.5-pro', 'prediction_meta-llama/Llama-3.3-70B-Instruct', 'prediction_azure/gpt-4o-mini-240718', 'reasoning_azure/gpt-4o-mini-240718', 'prediction_flip_azure/gpt-4o-mini-240718', 'reasoning_flip_azure/gpt-4o-mini-240718', 'prediction_Qwen/Qwen2.5-72B-Instruct', 'reasoning_Qwen/Qwen2.5-72B-Instruct', 'prediction_flip_Qwen/Qwen2.5-72B-Instruct', 'reasoning_flip_Qwen/Qwen2.5-72B-Instruct', 'prediction_meta-llama/Llama-3.1-70B-Instruct', 'reasoning_meta-llama/Llama-3.1-70B-Instruct', 'prediction_flip_meta-llama/Llama-3.1-70B-Instruct', 'reasoning_flip_meta-llama/Llama-3.1-70B-Instruct', 'agreement'])

        self.joint_df = pd.concat([self.ds[split].to_pandas() for split in self.ds.keys()])

    def get_dataframe(self, split: str="train"):
        if split == "all":
            return self.joint_df
        return self.ds[split].to_pandas()

    def get_user_data_df(self, user_id: str):
        return self.joint_df[self.joint_df['user_id'] == user_id]

    def get_user_data(self, user_id: str):
        df = self.get_user_data_df(user_id)
        train_data = df[df['split'] == 'train'].to_dict(orient='records')
        val_data = df[df['split'] == 'val'].to_dict(orient='records')
        test_data = df[df['split'] == 'test'].to_dict(orient='records')
        return train_data, val_data, test_data

    def get_all_user_data(self, split: str="all"):
        df = self.get_dataframe(split)

        # group by user_id and split
        grouped = df.groupby('user_id')
        return {user_id: (group[group['split'] == 'train'].to_dict(orient='records'),
                          group[group['split'] == 'val'].to_dict(orient='records'),
                          group[group['split'] == 'test'].to_dict(orient='records')) for user_id, group in grouped}
    
    def get_user_ids(self, split: str="all"):
        df = self.get_dataframe(split)
        return df['user_id'].unique()

class ChatbotArenaDataset(SynthesizeMeDataset):

    def __init__(self):
        super().__init__(hf_dataset_name="MichaelR207/chatbot_arena_personalized_0125")


class PrismDataset(SynthesizeMeDataset):

    def __init__(self):
        super().__init__(hf_dataset_name="MichaelR207/prism_personalized_0125")

if __name__ == "__main__":
    ds = ChatbotArenaDataset()
    ids = ds.get_user_ids()
    train, val, test = ds.get_all_user_data()[ids[13]]
    print(test)
