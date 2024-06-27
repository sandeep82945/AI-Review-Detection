import json 
import numpy 
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

def word_replacer(list1, list2, txt):
    """"To replace words in paraphrase which are similar to words found in reverse"""
    word_replace = {}
    for word in list2:
        if word not in list1:
            syn =[]
            for syno in wordnet.synsets(word):
                for lemma in syno.lemmas():
                    syn.append(lemma.name())

            for syno in syn:
                if syno in list1:
                    word_replace[word] = syno
                    break
    # Now replace the word
    for i in range(len(list(word_replace.keys()))):
        txt = txt.replace(list(word_replace.keys())[i], list(word_replace.values())[i])

    return txt

# Get all the adjective, noun, adverb that are synonym of each other in reverse and paraphrased, and then replace them with each other
def para_modifier(reversed,paraphrased):
    """To Modify a single review"""
    notations= [['JJ','JJS','JJR'],['NN','NNS','NNP','NNPS'],['RB','RBR','RBS']]
    reversed_all_adj = [tag[0] for tag in pos_tag(word_tokenize(reversed)) if tag[1] in notations[0]]
    reversed_all_noun = [tag[0] for tag in pos_tag(word_tokenize(reversed)) if tag[1] in notations[1]]
    reversed_all_adverb = [tag[0] for tag in pos_tag(word_tokenize(reversed)) if tag[1] in notations[2]]

    paraphrased_all_adj = [tag[0] for tag in pos_tag(word_tokenize(paraphrased)) if tag[1] in notations[0]]
    paraphrased_all_noun = [tag[0] for tag in pos_tag(word_tokenize(paraphrased)) if tag[1] in notations[1]]
    paraphrased_all_adverb = [tag[0] for tag in pos_tag(word_tokenize(paraphrased)) if tag[1] in notations[2]]

    # Now we have to form a word replace pair
    paraphrased = word_replacer(reversed_all_adj, paraphrased_all_adj,paraphrased)
    paraphrased = word_replacer(reversed_all_noun, paraphrased_all_noun,paraphrased)
    paraphrased = word_replacer(reversed_all_adverb, paraphrased_all_adverb,paraphrased)

    return paraphrased

def revw_para_modifier(reversed, paraphrased, isAI=1):
  """To modify the whole dataset"""
  modified = {}

  if (isAI):
    for i in range(len(paraphrased)):
      modified[list(paraphrased.keys())[i]] = para_modifier(list(reversed.values())[list(reversed.keys()).index(list(paraphrased.keys())[i])], list(paraphrased.values())[i])

  else:
    for i in range(len(paraphrased)):
      yup = []
      for j in range(len(list(paraphrased.values())[i])):
        yup.append(para_modifier(list(reversed.values())[list(reversed.keys()).index(list(paraphrased.keys())[i])], list(paraphrased.values())[i][j]))
      modified[list(paraphrased.keys())[i]] = yup

  return modified

if __name__=='__main__':
  # Loading files
  gpt4_iclr = json.load(open("Dataset/Regenerated Dataset/gpt-4/gpt4_regenerated_iclr.json","r"))
  gpt4_neur = json.load(open("Dataset/Regenerated Dataset/gpt-4/gpt4_regenerated_neur.json","r"))

  # paraphrased files
  ai_iclr_para = json.load(open("Dataset/Attacked Dataset/Paraphrasing Attack/para_ai_iclr.json","r"))
  ai_neur_para = json.load(open("Dataset/Attacked Dataset/Paraphrasing Attack/para_ai_neur.json","r"))
  human_iclr_para = json.load(open("Dataset/Attacked Dataset/Paraphrasing Attack/para_human_iclr.json","r"))
  human_neur_para = json.load(open("Dataset/Attacked Dataset/Paraphrasing Attack/para_human_neur.json","r"))

  # Actual reviews
  ai_iclr = json.load(open("Dataset/ai_iclr.json","r"))
  ai_neur = json.load(open("Dataset/ai_neur.json","r"))
  human_iclr = json.load(open("Dataset/human_iclr.json","r"))
  human_neur = json.load(open("Dataset/human_neur.json","r"))

  # Modifying the paraphrased reviews
  ai_iclr_para_modified = revw_para_modifier(gpt4_iclr,ai_iclr_para)
  ai_neur_para_modified = revw_para_modifier(gpt4_neur,ai_neur_para)
  human_iclr_para_modified = revw_para_modifier(gpt4_iclr,human_iclr_para,0)
  human_neur_para_modified = revw_para_modifier(gpt4_neur,human_neur_para,0)

  # Modifying the actual reviews
  ai_iclr_modified = revw_para_modifier(gpt4_iclr,ai_iclr)
  ai_neur_modified = revw_para_modifier(gpt4_neur,ai_neur)
  human_iclr_modified = revw_para_modifier(gpt4_iclr,human_iclr,0)
  human_neur_modified = revw_para_modifier(gpt4_neur,human_neur,0)

  # Saving the paraphrased reviews
  json.dump(ai_iclr_para_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/ai_iclr_para_modified.json","w"))
  json.dump(ai_neur_para_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/ai_neur_para_modified.json","w"))
  json.dump(human_iclr_para_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/human_iclr_para_modified.json","w"))
  json.dump(human_neur_para_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/human_neur_para_modified.json","w"))

  # Saving the actural reviews
  json.dump(ai_iclr_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/ai_iclr_modified.json","w"))
  json.dump(ai_neur_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/ai_neur_modified.json","w"))
  json.dump(human_iclr_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/human_iclr_modified.json","w"))
  json.dump(human_neur_modified,open("Dataset/Attacked Dataset/Modified Paraphrasing Attack/Modified Paraphrased/human_neur_modified.json","w"))