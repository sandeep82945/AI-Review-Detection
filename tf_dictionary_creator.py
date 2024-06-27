import json 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

def createDictionary(reviews, dictType):
    """Function for creating Token frequency dictionary"""
    notations= [['JJ','JJS','JJR'],['NN','NNS','NNP','NNPS'],['RB','RBR','RBS']]

    dict = {}
    total_revw = 0

    for i in range(len(reviews)):
        already_taken = []
        total_revw+=1
        tags = pos_tag(word_tokenize(reviews[i]))

        for tag in tags:
            if (tag[1] in notations[dictType]):
                if (tag[0].lower() not in already_taken):  # considering a token only once in a review
                    already_taken.append(tag[0].lower())
                    if (tag[0].lower() not in dict.keys()):  
                        dict[tag[0].lower()] = 1
                    else:   
                        dict[tag[0].lower()] += 1
    
    for key in dict.keys():
        dict[key] = dict[key]/total_revw

    return dict

if __name__=='__main__':
    ai_iclr = json.load(open('Dataset/ai_iclr.json',"r"))
    ai_neur = json.load(open('Dataset/ai_neur.json',"r"))
    human_iclr = json.load(open('Dataset/human_iclr.json',"r"))
    human_neur = json.load(open('Dataset/human_neur.json',"r"))

    # For formation of dictionary we are considering both the datasets together
    ai_revw = list(ai_iclr.values()) + list(ai_neur.values())
    human_revw = list(human_iclr.values()) + list(human_neur.values())
    human_revw = [a for revw in human_revw for a in revw]

    # Forming the dictionaries
    ai_adj_dict = createDictionary(ai_revw, 0)
    human_adj_dict = createDictionary(human_revw, 0)

    ai_noun_dict = createDictionary(ai_revw, 1)
    human_noun_dict = createDictionary(human_revw, 1)

    ai_adverb_dict = createDictionary(ai_revw, 2)
    human_adverb_dict = createDictionary(human_revw, 2)


    json.dump(ai_adj_dict, open("Dictionaries/ai_adj_dict.json","w"))
    json.dump(human_adj_dict, open("Dictionaries/human_adj_dict.json","w"))
    json.dump(ai_noun_dict, open("Dictionaries/ai_noun_dict.json","w"))
    json.dump(human_noun_dict, open("Dictionaries/human_noun_dict.json","w"))
    json.dump(ai_adverb_dict, open("Dictionaries/ai_adverb_dict.json","w"))
    json.dump(human_adverb_dict, open("Dictionaries/human_adverb_dict.json","w"))