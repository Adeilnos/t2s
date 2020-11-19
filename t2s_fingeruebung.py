import xml.etree.ElementTree as ET
import os
import spacy
import matplotlib.pyplot as plt
from functools import reduce


paths = []
for root, dirs, files in os.walk("Traning\\"):
    for name in files:
        if name.endswith('.xml'):
            paths.append(os.path.join(root, name))
 
#paths = ['Traning\\CP\\46_N_22_E.xml', 'Traning\\CP\\47_N_25_E.xml']
#paths = ['Traning\\CP\\46_N_22_E.xml']

    
#For every document, store the information in a list
complete_PoS = [] #PoS:[Token] [{'NOUN':['David'],'VERB':['attack','do']},{'NOUN':['Leo']}]
complete_tags = [] #Tag:{attrib:value}
complete_sentences = [] #[Words]
complete_sentences_length = [] #[length of sentence]

def concatenate_dicts(d0,d1):
    result = {key: value + d1.get(key,[]) for key, value in d0.items()}
    return result
#For every file, extract the information
for path in paths:
    #Load xml tree
    tree = ET.parse(path)
    #Get the root of xml file (<SpaceEvalTaskv1.2>)
    root = tree.getroot()
    #This contains the actual data (<TEXT>)
    text = root[0]

    #Contains the tags
    tags = root[1]
    #Store the data
    cdata = text.text

    #Load nlp model
    nlp = spacy.load("en_core_web_sm")
    #Analyze the data with our nlp model
    doc = nlp(cdata)

    #For every PoS tag, a list of values will be saved
    token_pos = {}

    #key -> PoS-tag, value -> list of tokens
    for token in doc:
        token_pos.setdefault(token.pos_,[]).append(token.text)
    #key -> tag, value -> dict of attributes
    tag_values = {}
    for child in tags:
        tag_values.setdefault(child.tag,[]).append(child.attrib)


    sentences = [sent.string.strip() for sent in doc.sents]
    sentence_length = [len(sentence.split()) for sentence in sentences]
    
    #For every xml file, add them to the list
    complete_PoS.append(token_pos)
    complete_tags.append(tag_values)
    complete_sentences.append(sentences)
    complete_sentences_length.append(sentence_length)
    
#zusammengeführte dicts für alle xml dateien
complete_dict_pos = reduce(lambda x,y: concatenate_dicts(x,y),complete_PoS)
complete_dict_tags = reduce(lambda x,y: concatenate_dicts(x,y),complete_tags)


#Aufgabe 2.3.1
for key in complete_dict_pos:
    print(f"Der PoS {key} kommt {len(complete_dict_pos[key])} Mal vor")
print("-----------------------------------------------------------------")

#Aufgabe 2.3.2
for tag in complete_dict_tags:
    print(f"{tag} kommt {len(complete_dict_tags[tag])} Mal vor")
print("-----------------------------------------------------------------")

#Aufgabe 2.3.3
qslink_occurence = {}
for qslink in complete_dict_tags['QSLINK']:
    qslink_occurence.setdefault(qslink['relType'],[]).append(1)

for qslink in qslink_occurence:
    print(f"Der QSLINK {qslink} kommt {len(qslink_occurence[qslink])} Mal vor")
print("-----------------------------------------------------------------")

#Aufgabe 2.3.4
complete_list_sentence_length = reduce(lambda x,y: x+y,complete_sentences_length)
    
    