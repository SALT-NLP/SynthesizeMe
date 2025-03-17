import numpy as np

def format_conversation(conversation, conv_format="""[{role}]: {content}"""):
    """
    Formats the conversation using the conversation format
    """
    if type(conversation) == np.ndarray:
        conversation = conversation.tolist()

    if not type(conversation) == list:
        conversation = [conversation]

    output = []
    for turn in conversation:
        output.append(conv_format.format(role=turn['role'], content=turn['content']))
    return '\n\n'.join(output)

def format_conversation_hf(conversation):
    """
    Formats the conversation for HuggingFace
    """
    if type(conversation) == np.ndarray:
        conversation = conversation.tolist()

    if not type(conversation) == list:
        conversation = [conversation]

    output = []
    for turn in conversation:
        revised_role = 'user' if turn['role'] == 'user' else 'assistant' if turn['role'] == 'model' or turn['role'] == 'assistant' else turn['role']
        output.append({'role': revised_role, 'content': turn['content']})
    return output