import pandas as pd
import json
from collections import defaultdict

def parse_conversation_row(row: dict) -> list[dict]:
    """
    Process a single row (a dictionary with a "contents" field) to extract conversation
    entries based on the provided logic. It supports rows where "contents" is either
    a JSON-encoded string (as in a CSV file) or a list of dictionaries.

    Args:
        row (dict): A row containing at least the "contents" key.

    Returns:
        list[dict]: A list of conversation fragments with keys: 'context', 'chosen', and 'rejected'.
    """
    entries = []
    context = []
    last_chosen = None
    rejected_list = []
    turn = -1

    contents = row['contents']
    # If contents is stored as JSON string, decode it.
    if isinstance(contents, str):
        try:
            text = json.loads(contents)
        except Exception as e:
            raise ValueError("Error parsing JSON in contents field") from e
    else:
        text = contents

    # Process each conversational item in the contents list.
    for item in text:
        if item['turn'] > turn and item['role'] == 'user':
            # When the turn increases and the item is from the user:
            if last_chosen is not None:
                # Add pairings for all the rejected completions so far.
                if rejected_list:
                    for rejected in rejected_list:
                        entries.append({
                            'context': context[:],
                            'chosen': last_chosen,
                            'rejected': rejected
                        })
                # Add the last chosen message to the context.
                context.append(last_chosen)
            # Reset the rejected list for the next user prompt.
            rejected_list = []
            # Append the current user message to the context.
            context.append({'role': item['role'], 'content': item['content']})
            turn = item['turn']

        # Process assistant or completion messages.
        if item['role'] == 'assistant' or 'completion' in item['role']:
            if item.get('chosen') == 'True':
                last_chosen = {'role': 'assistant', 'content': item['content']}
            elif item.get('chosen') == 'False':
                rejected_list.append({'role': 'assistant', 'content': item['content']})

    # After looping, if we have remaining rejected completions, add them.
    if last_chosen is not None and rejected_list:
        for rejected in rejected_list:
            entries.append({
                'context': context[:],
                'chosen': last_chosen,
                'rejected': rejected
            })
    return entries

def convert_wildfeedback_rows_to_json(rows: list[dict]) -> dict[list[dict]]:
    """
    Convert a list of row dictionaries (each with a "contents" field) into a structured
    mapping from user to conversation fragments.

    Args:
        rows (list[dict]): List of dictionaries each containing at least "contents" and "user" keys.

    Returns:
        dict[list[dict]]: Mapping of user IDs to a list of conversation entry dictionaries.
    """
    result = defaultdict(list)
    # Group rows by 'user'
    grouped_rows = {}
    for row in rows:
        user = row['user']
        grouped_rows.setdefault(user, []).append(row)
    
    # Process each user's rows in reverse order (to match the original functionality).
    for user, user_rows in grouped_rows.items():
        print(f"Processing user: {user} with {len(user_rows)} entries")
        for row in reversed(user_rows):
            entries = parse_conversation_row(row)
            result[user].extend(entries)
    return dict(result)

def convert_wildfeedback_csv_to_json(wildfeedback_csv: str) -> dict[list[dict]]:
    """
    Convert a CSV file from WildFeedback into a mapping of user IDs to lists
    of conversation fragments. This function preserves the original functionality
    while using the generic parser for processing each row's "contents".

    Args:
        wildfeedback_csv (str): Path to the WildFeedback CSV file.

    Returns:
        dict[list[dict]]: A mapping from user to their conversation entries.
    """
    df = pd.read_csv(wildfeedback_csv)
    # Retain only the columns of interest.
    df = df[['contents', 'user']]
    rows = df.to_dict(orient='records')
    return convert_wildfeedback_rows_to_json(rows)