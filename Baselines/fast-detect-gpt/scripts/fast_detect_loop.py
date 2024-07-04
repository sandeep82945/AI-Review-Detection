import random
import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')

    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

# run interactive local inference
def run(args, output_file):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text from folder
    folder_path = "/Data/sandeep/Vardhan/Paraphrased Dataset/AI_ICLR"  # Replace this with the path to your folder
    texts = process_text_files_in_folder(folder_path)
    with open(output_file, 'w') as f:
        for file_name, text in texts:
            if len(text) == 0:
                continue
            # evaluate text
            tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.reference_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = reference_model(**tokenized).logits[:, :-1]
                crit = criterion_fn(logits_ref, logits_score, labels)
            # estimate the probability of machine-generated text
            prob = prob_estimator.crit_to_prob(crit)
            result = f'File: {file_name}, Fast-DetectGPT criterion: {crit:.4f}, Probability: {prob * 100:.0f}%\n'
            f.write(result)

def process_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def process_text_files_in_folder(folder_path):
    text_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    texts = []
    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)
        text = process_text_file(file_path)
        texts.append((file_name, text))  # Store file name along with text
    return texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    output_file = "output.txt"  # Path to output file
    run(args, output_file)
