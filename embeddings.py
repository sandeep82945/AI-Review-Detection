from openai import OpenAI
import numpy
import json

client = OpenAI(api_key = "Your-OpenAI-API-Key")

def get_embedding(text, model="text-embedding-3-small"):
   """Function for creating embeddings"""
   text = text.replace("\n", " ")
   text = text.replace("\t", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def revw_embedding(revw,isAI):
    embed = {}
    if (isAI):
       for i in range(len(revw)):
          n = get_embedding(list(revw.values())[i])
          embed[list(revw.keys())[i]] = n
    else:
       for i in range(len(revw)):
          k = list(revw.keys())[i]
          embed[k] = []
          for j in range(len(list(revw.values())[i])):
             n = get_embedding(list(revw.values())[i][j])
             embed[k].append(n)
          

if __name__=='__main__':
    datasets = ["ai_iclr","ai_neur","human_iclr","human_neur"]

    # Actual Reviews
    for i in range(len(datasets)):
        revw = json.load(f"Dataset/{datasets[i]}.json")
        if(i<2):
            embed = revw_embedding(datasets[i],1)
            json.dump(embed, f"Embeddings/{datasets[i]}.json")
        else:
            embed = revw_embedding(datasets[i],0)
            json.dump(embed, f"Embeddings/{datasets[i]}.json")