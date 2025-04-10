import random

### LLM JUDGE PROMPTS ###
LLM_JUDGE_SYSTEM_PROMPT = """Your input fields are:
1. `conversation` (str): The conversation context leading up to the completions.
2. `first_completion` (str): The first of the two possible completions to judge between.
3. `second_completion` (str): The second of the two possible completions to judge between.

Your output fields are:
1. `reasoning` (str)
2. `preference` (Literal['First', 'Second']): The completion that the judge is more likely to prefer.  Possible values are 'First' and 'Second'.

All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## conversation ## ]]
{{conversation}}

[[ ## first_completion ## ]]
{{first_completion}}

[[ ## second_completion ## ]]
{{second_completion}}

Outputs will be a JSON object with the following fields.

{{
  "reasoning": "{{reasoning}}",
  "preference": "{{preference}}        # note: the value you produce must be one of: First; Second"
}}

In adhering to this structure, your objective is: 
        Given a conversation and two completions from different models, alongside some prior judgements and a user persona, determine which completion the human judge is more likely to prefer.  Use any provided context as well as the provided persona to speculate about the personal preferences of the judge.  You are a personalized reward model for this user, so think carefully about what this user will like.
        The user you are judging completions for has the FOLLOWING PERSONA: ===
        {persona}
        ===
        
        Now, given the conversation and two completions, decide which completion the user is more likely to prefer.  Remember to consider the user's persona and preferences as you make your decision.
"""

LLM_JUDGE_USER_TURN_PROMPT = """[[ ## conversation ## ]]
{conversation}

[[ ## first_completion ## ]]
{first_completion}

[[ ## second_completion ## ]]
{second_completion}

Respond with a JSON object in the following order of fields: `reasoning`, then `preference` (must be formatted as a valid Python Literal['First', 'Second']).
"""

LLM_JUDGE_ASSISTANT_TURN_PROMPT = """{{
  "reasoning": "{reasoning}",
  "preference": "{preference}"
}}"""

def format_llm_judge_prompt(persona, demos):
    system_prompt = LLM_JUDGE_SYSTEM_PROMPT.format(persona=persona)

    full_prompt = [{"role": "system", "content": system_prompt}]
    for demo in demos:
        # user prompt
        full_prompt.append({"role": "user", "content": LLM_JUDGE_USER_TURN_PROMPT.format(
            conversation=demo["conversation"],
            first_completion=demo["first_completion"] if 'reasoning' in demo else demo['completion_one'],
            second_completion=demo["second_completion"] if 'reasoning' in demo else demo['completion_two']
        )})
        # assistant prompt
        if 'reasoning' in demo:
            full_prompt.append({"role": "assistant", "content": LLM_JUDGE_ASSISTANT_TURN_PROMPT.format(
                reasoning=demo["reasoning"],
                preference=demo["preference"]
            )})
        else:
            full_prompt.append({"role": "assistant", "content": LLM_JUDGE_ASSISTANT_TURN_PROMPT.format(
                reasoning="Not provided for this particular example.",
                preference=demo["chosen"]
            )})

    full_prompt.append({"role": "user", "content": LLM_JUDGE_USER_TURN_PROMPT})

    return full_prompt
########################

### GENERATIVE PROMPTS ###
OPENING_PROMPT = """I'm going to tell you a bit about my personality and preferences.  I want you to keep it in mind and let it inform your responses and how you interact with me for the rest of the conversation.  Do you think you could do that?

Here's my persona:

{persona}"""

FIRST_DEMO_PROMPT = """I want to check how well you understand me with a few prediction exercises.  Can you read both of these LLM completions and tell me which one you think I'd prefer?  Then in the future you can respond more like the ones I like!  Please reason about what I as "the user" might like, but don't address me directly in your reasoning.  The reasoning is for your own future understanding.

Here's one:

{conversation}"""

FIRST_DEMO_PROMPT_NO_REASONING = """I want to check how well you understand me with a few prediction exercises.  Can you read both of these LLM completions and tell me which one you think I'd prefer?  Then in the future you can respond more like the ones I like! Please just respond with the preference of the first or second completion.

Here's one:

{conversation}"""

FOLLOWUPS = ["Great that was correct! Let's try another one.", "Yes, correct! Here's another one.", "That's right, I do prefer that one! Please take a look at this one.", "Yep! That was correct! Let's gather more data with another one.", "You got it! Let's do another."]
ADD_REASONING = ["Please include your reasoning for my preference this time.", "It would be great for you to provide your reasoning on this next one.", "Can you add in your reasoning for my preference for this one?", "Add your reasoning on this one!", "Please explain why you think I prefer one to the other.", "Can you please add your reasoning for this one?"]
NO_REASONING = ["For this one let's not include any reasoning, just your prediction of my preference.", "Just answer direct without reasoning.", "Please just give your prediction without explanation.", "Just share what your prediction is without reasoning.", "Let me know your prediction, no need for an explanation on this one.", "Just let me know your prediction this time, no reasoning needed."]
POST_REASONING = ["With that all in mind, I'd say {preference}.", "For those reasons, I think you prefer the {preference} completion.", "Based on my reasoning, I think you have a preference for the {preference} completion.", "I think you prefer the {preference} completion for those reasons."]
MODEL_WRAP_UP = ["Did I get it right?", "Let me know if I got this one right!", "How'd I do?", "What do you think? Do I understand your preferences?", "Was I right?", "So which do you actually prefer?"]
CLOSINGS = ["You nailed it!", "You got them all right, you really have a good sense of my preferences!", "You did great! I think you really understand me now.", "Wow, you got them all right! I'm impressed."]

FINAL_USER_PROMPT = "Now I want to transition into a normal conversation. Could you use everything you learned about me to tailor your responses to me from now on? Thanks!"
FINAL_MODEL_PROMPT = "Awesome! I’ll make sure to keep everything I’ve learned in mind and tailor my responses to match your preferences. I’m here to make the conversation as insightful and useful as possible for you!"

def format_conversation_choice(conversation, first_completion, second_completion):
    output = ["==="]
    output.append(f"**Conversation**: {conversation}")
    output.append("===")
    output.append(f"**Completion One (FIRST COMPLETION)**: {first_completion}")
    output.append("===")
    output.append(f"**Completion Two (SECOND COMPLETION)**: {second_completion}")
    output.append("===")

    return "\n".join(output)

def format_persona_prefix(persona):
    conversation = [{"role": "user", "content": OPENING_PROMPT.format(persona=f"{persona}")}]
    conversation.append({"role": "assistant", "content": "Got it! I’ll make sure to keep this in mind in how I engage with you. Let me know how I can help!"})
    return conversation

def format_generation_prompt(persona, demos, rand=random.Random()):
    conversation = format_persona_prefix(persona)
    for i, demo in enumerate(demos):
        preference = demo['preference'] if 'reasoning' in demo else demo['chosen']
        first = demo['first_completion'] if 'reasoning' in demo else demo['completion_one']
        second = demo['second_completion'] if 'reasoning' in demo else demo['completion_two']
        conversation_formatted = format_conversation_choice(demo['conversation'], first, second)
        if i == 0:
            if 'reasoning' in demo:
                conversation.append({"role": "user", "content": FIRST_DEMO_PROMPT.format(conversation=conversation_formatted)})
                conversation.append({"role": "assistant", "content": f"{demo['reasoning']} {rand.choice(POST_REASONING).format(preference=preference)} {rand.choice(MODEL_WRAP_UP)}"})
            else:
                conversation.append({"role": "user", "content": FIRST_DEMO_PROMPT_NO_REASONING.format(conversation=conversation_formatted)})
                conversation.append({"role": "assistant", "content": f"{preference}."})
        else:
            if 'reasoning' in demo:
                conversation.append({"role": "user", "content": f"{rand.choice(FOLLOWUPS)} {rand.choice(ADD_REASONING)}\n\n{conversation_formatted}"})
                conversation.append({"role": "assistant", "content": f"{demo['reasoning']} {rand.choice(POST_REASONING).format(preference=preference)} {rand.choice(MODEL_WRAP_UP)}"})
            else:
                conversation.append({"role": "user", "content": f"{rand.choice(FOLLOWUPS)} {rand.choice(NO_REASONING)}\n\n{conversation_formatted}"})
                conversation.append({"role": "assistant", "content": f"{preference}."})

    if len(demos) > 0:
        conversation.append({"role": "user", "content": f"{rand.choice(CLOSINGS)} {FINAL_USER_PROMPT}"})
        conversation.append({"role": "assistant", "content": FINAL_MODEL_PROMPT})

    return conversation

########################