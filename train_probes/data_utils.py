from functools import partial
import re
import difflib


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def extract_text_between_tags(s, tag):
    """Extract text between the last occurrence of <tag> and </tag> in s.

    Also strips the example answer prefix that appears in the task instruction.
    """
    s = s.split("<answer> C) 42 </answer>")[-1]
    first_tag = f"<{tag}>"
    i = s.rindex(first_tag)
    try:
        j = s.rindex(f"</{tag}>")
    except ValueError:
        j = len(s)
    return s[i + len(first_tag):j]


def search_list_right(list_to_search, element_to_find):
    """Return the last index of element_to_find in list_to_search, or -1."""
    for index in reversed(range(len(list_to_search))):
        if list_to_search[index] == element_to_find:
            return index
    return -1


def show_diff(text1, text2):
    """Print a line-by-line diff between two strings."""
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    differ = difflib.Differ()
    diff = differ.compare(lines1, lines2)
    print('\n'.join(diff))


def remove_leading_repeated_word(text, word_to_remove):
    """Strip leading repetitions of word_to_remove from text."""
    pattern = r'^(?:' + re.escape(word_to_remove) + r')+(.*)$'
    return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)


def get_predicted_multiple_choice(full_answer, tag='answer'):
    """Extract and normalise the predicted letter from a model's full answer."""
    raw_predicted = extract_text_between_tags(full_answer, tag)
    predicted = raw_predicted.strip().lower()
    return predicted


def normalize_answer(answer):
    """Return just the first letter of an answer string (lower-cased)."""
    tokens = answer.split()
    letter = tokens[0]
    return letter[:1].lower()


def is_correct_multiple_choice(full_answer, correct, tag='answer'):
    """Return True if the predicted answer letter matches the correct letter."""
    assert isinstance(full_answer, str)
    assert isinstance(correct, str)
    try:
        raw_predicted = get_predicted_multiple_choice(full_answer, tag=tag)
        predicted = normalize_answer(raw_predicted)
        correct = correct.strip().lower()
        return predicted == correct
    except ValueError as e:
        print(e)
        print("Full answer", full_answer)
        print("Correct", correct)
        return False
    except IndexError as e:
        print(e)
        return False


def dict_of_lists_to_list_of_dicts(input_dict):
    """Convert a dictionary of equal-length lists into a list of dictionaries."""
    keys = input_dict.keys()
    first_key = list(keys)[0]
    num_elements = len(input_dict[first_key])
    output_list = []
    for i in range(num_elements):
        new_dict = {}
        for key in keys:
            assert len(input_dict[key]) == num_elements
            new_dict[key] = input_dict[key][i]
        output_list.append(new_dict)
    return output_list


def to_chat(example, system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_instruction=None, prompt_key="Question", options_key="options",
            conversation_format="user/assistant"):
    """Convert a multiple-choice example into a chat-formatted dict.

    Appends lettered options (A, B, C, …) to the question text and builds a
    system/user message list.  Converts integer answer indices to letter strings
    in-place.
    """
    prompt = example[prompt_key]
    for o in range(len(example[options_key])):
        prompt += f"\n{'ABCDEFGHIJKLMNOP'[o]}) {example[options_key][o]}"
    assert task_instruction
    if type(example["answer"]) == int:
        example["answer"] = "abcdefghijklmnop"[example["answer"]]
    example["chat"] = [{"role": "system", "content": task_instruction},
                       {"role": "user", "content": prompt}]
    del example["options"]
    return example


def prepare_dataset(dataset, tokenizer, num_proc=16,
                    prompt_key="prompt",
                    task_instruction=None,
                    conversation_format="user/assistant",
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    add_generation_prompt=True, tokenize=False):
    """Apply chat formatting and the model's chat template to a HuggingFace Dataset.

    Returns the dataset with an additional ``formatted_chat`` column containing
    the tokenizer-formatted prompt string ready for model input.
    """
    ds = dataset
    transforms = []

    f = partial(to_chat, system_prompt=system_prompt,
                prompt_key="Question", task_instruction=task_instruction,
                conversation_format=conversation_format)
    transforms.append(f)

    def apply_chat(example):
        chat = example["chat"]
        example["formatted_chat"] = tokenizer.apply_chat_template(
            chat,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt)
        del example["chat"]
        return example

    transforms.append(apply_chat)

    # Dry run on one example to surface errors early
    for ex in ds:
        for transform in transforms:
            ex = transform(ex)
        break

    for transform in transforms:
        ds = ds.map(transform, num_proc=num_proc, load_from_cache_file=False)

    return ds
