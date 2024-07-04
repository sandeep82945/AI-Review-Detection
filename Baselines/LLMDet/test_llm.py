import os
import llmdet
from tqdm import tqdm

def classify(text):
    max_score = 0
    result = llmdet.detect(text)
    human_score = result[0]['Human_write']
    for key, val in result[0].items():
        max_score = max(max_score, val)
    if human_score == max_score:
        return "Human written"
    else:
        return "AI written"

llmdet.load_probability()

folder_path = "/Data/sandeep/Vardhan/Datasets/Attacked Data/Noun1/AI ICLR"
output_file = "/Data/sandeep/Vardhan/Results/Attacked/Noun/llmdet_noun_ai_iclr.txt"

with open(output_file, "w") as f_out:
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f_in:
                text = f_in.read()
                detection_result = classify(text)
                f_out.write(f"{file_name} -> {detection_result}\n")
