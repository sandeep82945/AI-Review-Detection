import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deployment import preprocess, detect

# init
device ="cuda" if torch.cuda.is_available() else "cpu" # use 'cuda:0' if GPU is available
# model_dir = "nealcly/detection-longformer" # model in our paper
model_dir = "yaful/DeepfakeTextDetection"  # model in the online demo
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

def deepfake(input_folder, output_folder):

    # Open output file in append mode
    with open(output_folder, 'a') as output:
        # Loop through each file in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):  # Assuming all input files are text files
                file_path = os.path.join(input_folder, filename)
                # Read text from file
                with open(file_path, 'r') as file:
                    text = file.read()
                
                # Preprocess
                # text = preprocess(text)
                
                # Detection
                result = detect(text, tokenizer, model, device)
                
                # Write result along with filename to output file
                output.write(f"{filename} -> {result}\n")


folder_list = ["AI ICLR","AI NeurIPS","Human ICLR","Human NeurIPS"]
output_list = ["ai_iclr","ai_neur","human_iclr","human_neur"]

folder_path = "/Data/sandeep/Vardhan/Datasets/Attacked Data/Noun1/"

for i in range(len(folder_list)):
    deepfake(folder_path+folder_list[i],f"/Data/sandeep/Vardhan/Results/Attacked/Noun/deepfake_noun_{output_list[i]}.txt")

# input_folder = "/Data/sandeep/Vardhan/Paraphrased Defence Dataset/HUMAN_ICLR"  # Specify the folder where your input files are located
# output_file = "Results/Paraphrased Defence/deepfake_paraphrased_human_iclr.txt"  # Output file to store the results