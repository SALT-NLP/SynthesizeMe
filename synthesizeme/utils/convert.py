import pandas as pd
import json

def convert_wildfeedback_to_json(wildfeedback_csv: str) -> dict[list[dict]]:
    """
    Convert a CSV file from WildFeedback into a list of dictionaries.
    Each dictionary represents a row in the CSV file.

    Args:
        wildfeedback_csv (str): Path to the WildFeedback CSV file.

    Returns:
        list[dict]: List of dictionaries representing the CSV data.
    """
    df = pd.read_csv(wildfeedback_csv)
    # drop all columns besides 'contents' and 'user'
    df = df[['contents', 'user']]

    # group by 'user'
    grouped = df.groupby('user')

    # As we loop over the contents we should continuously build up a context of the previous contents and add all pairwise chosen and rejected completions
    # The format of each entry will be {'conversation': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}], 'chosen': ..., 'rejected': ...}
    # We will return a mapping of user to a list of these entries
    result = {}

    for user, group in grouped:
        print(f"Processing user: {user} with {len(group)} entries")
        # Initialize the user's list if not already done
        if user not in result:
            result[user] = []

        # reverse the order of the group to process the most recent entries first
        group = group.iloc[::-1].reset_index(drop=True)

        # Iterate over the contents and create a conversation context in reverse order
        for index, row in group.iterrows():
            context = []
            last_chosen = None
            rejected_list = []
            turn = -1
            text = json.loads(row['contents'])

            # Iterate over the text to build the conversation context
            for i, item in enumerate(text):
                if item['turn'] > turn and item['role'] == 'user':
                    if last_chosen is not None:
                        # For all rejected items, we need to pair them with chosen and add to the users dataset
                        if rejected_list:
                            for rejected in rejected_list:
                                result[user].append({
                                    'context': context[:],
                                    'chosen': last_chosen,
                                    'rejected': rejected
                                })

                        # Add the last chosen item to the context
                        context.append(last_chosen)

                    # Reset the rejected list for the next turn
                    rejected_list = []

                    # Add the current item to the context
                    context.append({'role': item['role'], 'content': item['content']})

                    turn = item['turn']

                if item['role'] == 'assistant' or 'completion' in item['role']:
                    if item['chosen'] == 'True':
                        last_chosen = {'role': 'assistant', 'content': item['content']}
                    elif item['chosen'] == 'False':
                        rejected_list.append({'role': 'assistant', 'content': item['content']})

            # After the loop, if there are any remaining rejected items, add them to the result
            if last_chosen is not None and rejected_list:
                for rejected in rejected_list:
                    result[user].append({
                        'context': context[:],
                        'chosen': last_chosen,
                        'rejected': rejected
                    })

    return result
            