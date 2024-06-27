import json 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

def get_synonyms(word):
    """Function for finding synonym of a word"""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def top_keys(dictionary, n=100):
    # Sort the dictionary by values in descending order
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

    # Extract keys corresponding to the top n values
    top_keys_list = [item[0] for item in sorted_items[:n]]

    return top_keys_list

def synonym_attack_dict(dictionary):
    """Function for figuring out words to be replaced"""

    # Sort the dictionary by values in descending order
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

    # Extract keys corresponding to the top n values
    n = 100
    top_keys_list = [item[0] for item in sorted_items[:n]]
    
    # for the top keys find out their synonyms
    syn_attack = {}
    for i in range(len(top_keys_list)):
        syn_word = get_synonyms(top_keys_list[i])

        for a in syn_word:
            if a in dictionary.keys():
                syn_attack[top_keys_list[i]] = a
                break

    return syn_attack

def synonym_attack(review,dictionary,attackType):
    """function for performing synonym attack in a review"""

    syn_attack = synonym_attack_dict(dictionary)

    # To figure out which type of attack we are doing, so we will find only those kind of words
    notations= [['JJ','JJS','JJR'],['NN','NNS','NNP','NNPS'],['RB','RBR','RBS']]

    
    all_words = [tag[0] for tag in pos_tag(word_tokenize(review)) if (tag[1] in notations[attackType])]

    for a in all_words:
        if a in list(syn_attack.keys()):
            review = review.replace(a, syn_attack[a])
    
    return review

def synonym_attacker_dataset(reviews, isAI, dictionary, attackType):
    """function for making a datatset with synonyms"""

    if (isAI):
        for i in range(len(reviews)):
            revw = list(reviews.keys())[i]   # selecting a review
            reviews[revw] = synonym_attack(reviews[revw], dictionary, attackType)

    else:
        for i in range(len(reviews)):
            revw = reviews[list(reviews.keys())[i]]
            new_revw = []
            for j in range(len(revw)):
                new_revw.append(synonym_attack(revw[j], dictionary, attackType))

            reviews[list(reviews.keys())[i]] = new_revw

    return reviews

if __name__ == "__main__":
    # Loading the Datasets
    ai_iclr = json.load(open("Dataset/ai_iclr.json"))
    ai_neur = json.load(open("Dataset/ai_neur.json"))
    human_iclr = json.load(open("Dataset/human_iclr.json"))
    human_neur = json.load(open("Dataset/human_neur.json"))

    # Loading the dictionaries
    ai_adj_dict = json.load(open("Dictionaries/ai_adj_dict.json","r"))
    ai_noun_dict = json.load(open("Dictionaries/ai_noun_dict.json","r"))
    ai_adverb_dict = json.load(open("Dictionaries/ai_adverb_dict.json","r"))

    # Adjective attack
    ai_iclr_adj_attack = synonym_attacker_dataset(ai_iclr, 1, ai_adj_dict,0)
    human_iclr_adj_attack = synonym_attacker_dataset(human_iclr, 0, ai_adj_dict,0)
    ai_neur_adj_attack = synonym_attacker_dataset(ai_neur, 1, ai_adj_dict,0)
    human_neur_adj_attack = synonym_attacker_dataset(human_neur, 0, ai_adj_dict,0)

    # Noun Attack
    ai_iclr_noun_attack = synonym_attacker_dataset(ai_iclr, 1, ai_noun_dict,0)
    human_iclr_noun_attack = synonym_attacker_dataset(human_iclr, 0, ai_noun_dict,0)
    ai_neur_noun_attack = synonym_attacker_dataset(ai_neur, 1, ai_noun_dict,0)
    human_neur_noun_attack = synonym_attacker_dataset(human_neur, 0, ai_noun_dict,0)

    # Adverb Attack
    ai_iclr_adverb_attack = synonym_attacker_dataset(ai_iclr, 1, ai_adverb_dict,0)
    human_iclr_adverb_attack = synonym_attacker_dataset(human_iclr, 0, ai_adverb_dict,0)
    ai_neur_adverb_attack = synonym_attacker_dataset(ai_neur, 1, ai_adverb_dict,0)
    human_neur_adverb_attack = synonym_attacker_dataset(human_neur, 0, ai_adverb_dict,0)


    # Saving the datasets
    json.dump(ai_iclr_adj_attack, open("Dataset/Attacked Dataset/Adjective Attack/ai_iclr.json","w"))
    json.dump(ai_neur_adj_attack, open("Dataset/Attacked Dataset/Adjective Attack/ai_neur.json","w"))
    json.dump(human_iclr_adj_attack, open("Dataset/Attacked Dataset/Adjective Attack/human_iclr.json","w"))
    json.dump(human_neur_adj_attack, open("Dataset/Attacked Dataset/Adjective Attack/human_neur.json","w"))

    json.dump(ai_iclr_adverb_attack, open("Dataset/Attacked Dataset/Adverb Attack/ai_iclr.json","w"))
    json.dump(ai_neur_adverb_attack, open("Dataset/Attacked Dataset/Adverb Attack/ai_neur.json","w"))
    json.dump(human_iclr_adverb_attack, open("Dataset/Attacked Dataset/Adverb Attack/human_iclr.json","w"))
    json.dump(human_neur_adverb_attack, open("Dataset/Attacked Dataset/Adverb Attack/human_neur.json","w"))

    json.dump(ai_iclr_noun_attack, open("Dataset/Attacked Dataset/Noun Attack/ai_iclr.json","w"))
    json.dump(ai_neur_noun_attack, open("Dataset/Attacked Dataset/Noun Attack/ai_neur.json","w"))
    json.dump(human_iclr_noun_attack, open("Dataset/Attacked Dataset/Noun Attack/human_iclr.json","w"))
    json.dump(human_neur_noun_attack, open("Dataset/Attacked Dataset/Noun Attack/human_neur.json","w"))