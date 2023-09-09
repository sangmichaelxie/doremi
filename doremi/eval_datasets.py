from datasets import load_dataset, concatenate_datasets
import string


def substring_until(s, split_strs):
    idx = len(s)
    for split_str in split_strs:
        try:
            new_idx = s.index(split_str)
            if new_idx < idx:
                idx = new_idx
        except Exception:
            pass
    return s[:idx]


def pred_postprocess_default(pred):
    pred = pred.strip().lower()
    return substring_until(pred, ['\n']).strip().lower().translate(str.maketrans('', '', string.punctuation))


def eval_func_default(answer, pred, prompt, model=None, tokenizer=None, inputs=None, trainer=None):
    if not isinstance(answer, list):
        answer = [answer.strip().lower().translate(str.maketrans('', '', string.punctuation))]
    else:
        answer = [a.strip().lower().translate(str.maketrans('', '', string.punctuation)) for a in answer]
    return pred in answer


def get_eval_dataset(dataset_name, num_shots, seed=42):

    # defaults
    top_k = 1
    top_p = 0
    temperature = 1
    num_shots = num_shots
    max_new_tokens = 20
    shuffle_train = True

    eval_func = eval_func_default
    pred_postprocess_func = pred_postprocess_default

    # load fewshot dataset
    if dataset_name == 'trivia_qa':
        dataset = load_dataset(dataset_name, name='rc.nocontext')
        dataset_train = dataset['train']
        dataset_val = dataset['validation']
        input_key = 'question'
        output_key = 'answer'

        def prompt_transform(ex, context_exs):
            prompt = '\n\n'.join([f"Question: {c_ex[input_key]}\nAnswer: {c_ex[output_key]['aliases'][0]}" for c_ex in context_exs])
            prompt += f"\n\nQuestion: {ex[input_key]}\nAnswer:"

            answer_list = ex[output_key]['aliases']
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'natural_questions':
        dataset = load_dataset("lucadiliello/naturalquestionsshortqa")
        dataset_train = dataset['train']
        dataset_val = dataset['validation']

        def prompt_transform(ex, context_exs):
            prompt = '\n\n'.join([f"Q: {c_ex['question']}?\n\nA: {c_ex['answers'][0]}"
                                  for c_ex in context_exs])
            prompt += f"\n\nQ: {ex['question']}?\n\nA:"

            answer_list = ex['answers']
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'web_questions':
        dataset = load_dataset(dataset_name)
        dataset_train = dataset['train']
        dataset_val = dataset['test']

        def prompt_transform(ex, context_exs):
            prompt = '\n\n'.join([f"Question: {c_ex['question']}\nAnswer: {c_ex['answers'][0]}"
                                  for c_ex in context_exs])
            prompt += f"\n\nQuestion: {ex['question']}\nAnswer:"

            answer_list = ex['answers']
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'lambada':
        dataset = load_dataset(dataset_name)
        dataset_train = dataset['validation']
        dataset_val = dataset['test']

        def prompt_transform(ex, context_exs):
            words = ex['text'].split(' ')
            ex_input = ' '.join(words[:-1])
            ex_answer = words[-1]

            context_ex_toks = [c_ex['text'].split(' ') for c_ex in context_exs]
            prompt = '\n\n'.join([f"Input: {' '.join(c_ex_toks[:-1])}\nOutput: {c_ex_toks[-1]}"
                                  for c_ex_toks in context_ex_toks])
            prompt += f"\n\nInput: {ex_input}\nOutput:"
            prompt = "Complete the following sentences.\n\n" + prompt

            answer_list = [ex_answer]
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'squad_v2':
        dataset = load_dataset(dataset_name)
        # dataset_train = dataset['train']
        shuffle_train = False

        dataset_val = dataset['validation']

        # get indices for each title
        dataset_val_chunks = []
        dataset_train_chunks = []
        all_titles = set([ex['title'] for ex in dataset_val])
        for i, title in enumerate(all_titles):
            title_dataset_val = dataset_val.filter(lambda x: x['title'] == title).shuffle(seed + i)
            title_dataset_train = title_dataset_val.select(list(reversed(range(len(title_dataset_val)))))
            assert(len(title_dataset_train) == len(title_dataset_val))
            dataset_train_chunks.append(title_dataset_train)
            dataset_val_chunks.append(title_dataset_val)

        dataset_train = concatenate_datasets(dataset_train_chunks)
        dataset_val = concatenate_datasets(dataset_val_chunks)

        def prompt_transform(ex, context_exs):
            for c_ex in [ex] + context_exs:
                if len(c_ex['answers']['text']) == 0:
                    c_ex['answers']['text'] = ['unanswerable']

                assert(c_ex['title'] == ex['title'])

            prompt = f"Title: {ex['title']}\n\nBackground: {ex['context']}\n\n"
            prompt += '\n\n'.join([f"Question: {c_ex['question']}\n\nAnswer (use Background or answer \"unanswerable\"): {c_ex['answers']['text'][0]}"])
            prompt += f"\n\nQuestion: {ex['question']}\n\nAnswer (use Background or answer \"unanswerable\"):"

            answer_list = ex['answers']['text']
            return {'prompt': prompt, 'answer': answer_list}

        def eval_func(answer, pred, prompt, model, tokenizer, inputs, trainer):
            if not isinstance(answer, list):
                answer = [answer.strip().lower().translate(str.maketrans('', '', string.punctuation))]
            else:
                answer = [a.strip().lower().translate(str.maketrans('', '', string.punctuation)) for a in answer]
            return pred in answer

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return {
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,
        'num_shots': num_shots,
        'max_new_tokens': max_new_tokens,
        'prompt_transform': prompt_transform,
        'dataset_train': dataset_train,
        'shuffle_train': shuffle_train,
        'dataset_val': dataset_val,
        'eval_func': eval_func,
        'pred_postprocess_func': pred_postprocess_func, }
