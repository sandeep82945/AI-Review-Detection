import numpy as np
import datasets
from tqdm import tqdm
import math
import json
from transformers import AutoTokenizer,  LlamaTokenizer, BartTokenizer, T5Tokenizer
from unilm import UniLMTokenizer
from lightgbm import Booster


def load_probability():
    """
    The `load_probability()` is utilized to load the dictionary required for LLMDet from the Hugging Face repository.
    The initial loading process incurs a certain time overhead.
    """

    # The initial utilization of LLMDet involves loading the dictionary file from HuggingFace.
    # And subsequently caching it in the local cache for future reference.
    dm = datasets.DownloadManager()
    files = dm.download_and_extract('https://huggingface.co/datasets/TryMore/n_grams_probability/resolve/main/n-grams_probability.tar.gz')
    model = ["gpt2", "opt", "unilm", "llama", "bart", "t5", "bloom", "neo", "vicuna" , "gpt2_large", "opt_3b"]
    global_vars = globals()
    for item in model:
        n_grams = np.load(f'{files}/npz/{item}.npz', allow_pickle=True)
        global_vars[item] = n_grams["t5"]


def perplexity(text_set_token_ids, n_grams_probability, vocab_size):
    """
    The `perplexity()` is used to calculate proxy perplexity with dictionary load in `load_probability()`.
    For each Language Model that has constructed an n-grams dictionary, a corresponding proxy perplexity will be computed."
    """
    
    test_perplexity = []

    for k in tqdm(range(len(text_set_token_ids))):
        text_token_ids = text_set_token_ids[k]
        ppl = 0
        number_3_grams = 0
        number_4_grams = 0
        number_2_grams = 0
        for i in range(2, len(text_token_ids) - 1):

            # Calculate the perplexity with 4-grams samples probability
            if tuple([text_token_ids[i - j] for j in range(2, -1, -1)]) in n_grams_probability[4].keys():
                if text_token_ids[i + 1] in n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])]:
                    if n_grams_probability[5][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])][n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])].tolist().index(text_token_ids[i + 1])] > 0:
                        ppl = ppl + math.log2(n_grams_probability[5][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])][n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])].tolist().index(text_token_ids[i + 1])])
                else:
                    top_k = len(n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])])
                    sum_probs = sum(n_grams_probability[5][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])])
                    if (1 - sum_probs) > 0:
                        ppl = ppl + math.log2((1 - sum_probs) / (vocab_size - top_k))
                number_4_grams = number_4_grams + 1

            # Calculate the perplexity with 3-grams samples probability
            elif tuple([text_token_ids[i - 1], text_token_ids[i]]) in n_grams_probability[2].keys():
                if text_token_ids[i + 1] in n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])]:
                    if n_grams_probability[3][tuple([text_token_ids[i - 1], text_token_ids[i]])][n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])].tolist().index(text_token_ids[i + 1])] > 0:
                        ppl = ppl + math.log2(
                            n_grams_probability[3][tuple([text_token_ids[i - 1], text_token_ids[i]])][n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])].tolist().index(text_token_ids[i + 1])])
                else:
                    top_k = len(n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])])
                    sum_probs = sum(n_grams_probability[3][tuple([text_token_ids[i - 1], text_token_ids[i]])])
                    if (1 - sum_probs) > 0:
                        ppl = ppl + math.log2((1 - sum_probs) / (vocab_size - top_k))
                number_3_grams = number_3_grams + 1

            # Calculate the perplexity with 2-grams samples probability
            elif tuple([text_token_ids[i]]) in n_grams_probability[0].keys():
                if text_token_ids[i + 1] in n_grams_probability[0][tuple([text_token_ids[i]])]:
                    if n_grams_probability[1][tuple([text_token_ids[i]])][n_grams_probability[0][tuple([text_token_ids[i]])].tolist().index(text_token_ids[i + 1])] > 0:
                        ppl = ppl + math.log2(n_grams_probability[1][tuple([text_token_ids[i]])][n_grams_probability[0][tuple([text_token_ids[i]])].tolist().index(text_token_ids[i + 1])])
                else:
                    top_k = len(n_grams_probability[0][tuple([text_token_ids[i]])])
                    sum_probs = sum(n_grams_probability[1][tuple([text_token_ids[i]])])
                    if (1 - sum_probs) > 0:
                        ppl = ppl + math.log2((1 - sum_probs) / (vocab_size - top_k))
                number_2_grams = number_2_grams + 1

        perplexity = ppl / (number_2_grams + number_3_grams + number_4_grams + 1)
        test_perplexity.append(-perplexity)

    return test_perplexity


def detect(text):
    if isinstance(text, str):
        test_text = [text]
    elif isinstance(text, list):
        if isinstance(text[0], str):
            test_text = text
        else:
            raise ValueError(
                "The type of `text` which you input is not a string or list. "
                "Please enter the correct data type for `text`."
            )
    else:
        raise ValueError(
            "The type of `text` which you input is not a string or list. "
            "Please enter the correct data type for `text`."
        )
    dm = datasets.DownloadManager()
    # files = dm.download_and_extract('https://huggingface.co/datasets/wukx/n-grams_sample_probability/resolve/main/n_grams.zip')

    model_information = [{"model_name": "gpt2", "vocab_size": 50265, "model_probability": "gpt2"},
                         {"model_name": "facebook/opt-1.3b", "vocab_size": 50257, "model_probability": "opt"},
                         {"model_name": "microsoft/unilm-base-cased", "vocab_size": 28996, "model_probability": "unilm"},
                         {"model_name": "baffo32/decapoda-research-llama-7B-hf", "vocab_size": 32000, "model_probability": "llama"},
                         {"model_name": "facebook/bart-base", "vocab_size": 50265, "model_probability": "bart"},
                         {"model_name": "google/flan-t5-base", "vocab_size": 32128, "model_probability": "t5"},
                         {"model_name": "bigscience/bloom-560m", "vocab_size": 250880, "model_probability": "bloom"},
                         {"model_name": "EleutherAI/gpt-neo-2.7B", "vocab_size": 50257, "model_probability": "neo"},
                         {"model_name": "lmsys/vicuna-7b-delta-v1.1", "vocab_size": 32000, "model_probability": "vicuna"},
                         {"model_name": "gpt2-large", "vocab_size": 50265, "model_probability": "gpt2_large"},
                         {"model_name": "facebook/opt-2.7b", "vocab_size": 50257, "model_probability": "opt_3b"}]
    #
    # labels_to_number = {"Human_write": 0, "GPT-2": 1, "OPT": 2, "UniLM": 3, "llama": 4, "bart": 5, "t5": 6, "bloom": 7,
    #                     "neo": 8}

    # Calculate the proxy perplexity
    perplexity_result = []
    for model in model_information:
        if any([(model_type in model["model_name"]) for model_type in ["unilm"]]):
            tokenizer = UniLMTokenizer.from_pretrained(model["model_name"])
        elif any([(model_type in model["model_name"]) for model_type in ["llama", "vicuna"]]):
            tokenizer = LlamaTokenizer.from_pretrained(model["model_name"])
        elif any([(model_type in model["model_name"]) for model_type in ["t5"]]):
            tokenizer = T5Tokenizer.from_pretrained(model["model_name"])
        elif any([(model_type in model["model_name"]) for model_type in ["bart"]]):
            tokenizer = BartTokenizer.from_pretrained(model["model_name"])
        else:
            tokenizer = AutoTokenizer.from_pretrained(model["model_name"])

        text_token_ids = []
        for text in test_text:
            if any([(model_type in model["model_name"]) for model_type in ["gpt"]]):
                text_tokenize = tokenizer.tokenize(text, truncation=True)
            else:
                text_tokenize = tokenizer.tokenize(text)
            text_token_ids.append(tokenizer.convert_tokens_to_ids(text_tokenize))


        if model["model_probability"] in globals():
            perplexity_result.append(perplexity(text_token_ids, globals()[model["model_probability"]], model["vocab_size"]))
        else:
            raise ValueError(
                "The {} does not exist, please load n-grams probability!".format(model["model_probability"])
            )

    # The input features of classiffier
    features = np.stack([perplexity_result[i] for i in range(len(perplexity_result))], axis=1)

    # Load classiffier model
    model_files = dm.download_and_extract('https://huggingface.co/datasets/TryMore/n_grams_probability/resolve/main/LightGBM_model.zip')
    model = Booster(model_file=f'{model_files}/nine_LightGBM_model.txt')
    y_pred = model.predict(features)
    label = ["Human_write", "GPT-2", "OPT", "UniLM", "LLaMA", "BART", "T5", "Bloom", "GPT-neo"]
    test_result = [{label[i]: y_pred[j][i] for i in range(len(label))} for j in range(len(y_pred))]
    for i in range(len(test_result)):
        test_result[i] = dict(sorted(test_result[i].items(), key=lambda x: x[1], reverse=True))

    return test_result
