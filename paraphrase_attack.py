import json
import google.generativeai as genai

# Configuration of the model
genai.configure(api_key='GOOGLE_API_KEY')
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def paraphraser(text):
    """Function for paraphrasing"""
    prompt = "Paraphrase the following review"
    chat = model.start_chat(history=[])

    response = chat.send_message(prompt+": "+ text )

    return response.text

def paraphrase_dataset(reviews, isAI):
    """Function for paraphrasing the dataset"""
    paraphrased_revw = {}
    if (isAI):
      for i in range(len(reviews)):
        paraphrased_revw[list(reviews.keys())[i]] = paraphraser(list(reviews.values())[i])
    else:
       for i in range(len(reviews)):
        para_revw = []
        for j in range(len(list(reviews.values())[i])):
            para_revw.append(paraphraser(reviews[list(reviews.keys())[i]][j]))
        paraphrased_revw[list(reviews.keys())[i]] = para_revw

    return paraphrased_revw

if __name__=='__main__':

    # loading the dataset
    ai_iclr = json.load(open("Dataset/ai_iclr.json","r"))
    ai_neur = json.load(open("Dataset/ai_neur.json","r"))
    human_iclr = json.load(open("Dataset/human_iclr.json","r"))
    human_neur = json.load(open("Dataset/human_neur.json","r"))

    # Paraphrasing the dataset
    ai_iclr = paraphrase_dataset(ai_iclr, 1)
    ai_neur = paraphrase_dataset(ai_neur, 1)
    human_iclr = paraphrase_dataset(human_iclr, 0)
    human_neur = paraphrase_dataset(human_neur, 0)

    json.dump(ai_iclr, open("Dataset/Attacked Dataset/Paraphrasing Attack/para_ai_iclr.json","w"))
    json.dump(ai_neur, open("Dataset/Attacked Dataset/Paraphrasing Attack/para_ai_neur.json","w"))
    json.dump(human_iclr, open("Dataset/Attacked Dataset/Paraphrasing Attack/para_human_iclr.json","w"))
    json.dump(human_neur, open("Dataset/Attacked Dataset/Paraphrasing Attack/para_human_neur.json","w"))