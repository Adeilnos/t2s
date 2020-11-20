import xml.etree.ElementTree as ET
import os
import spacy
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np


paths = []
for root, dirs, files in os.walk("Traning\\"):
    for name in files:
        if name.endswith('.xml'):
            paths.append(os.path.join(root, name))

#paths = ['Traning\\CP\\46_N_22_E.xml', 'Traning\\CP\\47_N_25_E.xml']
#paths = ['Traning\\CP\\46_N_22_E.xml']

#liste der gesamten tags aller xml files
complete_PoS = []
complete_tags = []
complete_sentences = []
complete_sentences_length = []

#fuegt 2 dicts zusammen
def concatenate_dicts(d0,d1):
    result = {key: value + d1.get(key,[]) for key, value in d0.items()}
    return result

#fuer jedes xml file
for path in paths:
    #lade xml tree
    tree = ET.parse(path)
    #hole die wurzel
    root = tree.getroot()
    #relevante Daten
    text = root[0]

    #tags
    tags = root[1]
    #Daten
    cdata = text.text

    #laedt nlp modell
    nlp = spacy.load("en_core_web_sm")
    #Analyse
    doc = nlp(cdata)

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
    
    #fuege jedes xml in die complete liste ein
    complete_PoS.append(token_pos)
    complete_tags.append(tag_values)
    complete_sentences.append(sentences)
    complete_sentences_length.append(sentence_length)
    
#zusammengeführte dicts für alle xml dateien
complete_dict_pos = reduce(lambda a,b: concatenate_dicts(a,b),complete_PoS)
complete_dict_tags = reduce(lambda a,b: concatenate_dicts(a,b),complete_tags)


print("Aufgabe 2.3.1---------------------------------------------------------------------------------------")
for key in complete_dict_pos:
    print(f"{len(complete_dict_pos[key])} x PoS {key}")

print("Aufgabe 2.3.2---------------------------------------------------------------------------------------")
for tag in complete_dict_tags:
    print(f"{len(complete_dict_tags[tag])} x {tag}")

print("Aufgabe 2.3.3---------------------------------------------------------------------------------------")
qslink_occurence = {}
for qslink in complete_dict_tags['QSLINK']:
    qslink_occurence.setdefault(qslink['relType'],[]).append(1)

for qslink in qslink_occurence:
    print(f"{len(qslink_occurence[qslink])} x QSLINK {qslink}")

print("Aufgabe 2.3.4---------------------------------------------------------------------------------------")
complete_list_sentence_laenge = reduce(lambda a,b: a+b,complete_sentences_length)

plt.figure(figsize=(17,10)) 
plt.hist(complete_list_sentence_laenge,bins=np.arange(0,max(complete_list_sentence_laenge))-0.5)
plt.xticks(range(0,max(complete_list_sentence_laenge)+1))
plt.yticks(range(0,max(np.bincount(complete_list_sentence_laenge)+1)))
plt.title('Satzlaengenhaeufigkeit')
plt.ylabel('Haeufigkeit')
plt.xlabel('Satzlaenge')
plt.tight_layout()
plt.savefig("satzlaenge.png")
plt.show()

print("Aufgabe 2.3.5---------------------------------------------------------------------------------------")
#trigger paare werden als tupel in die liste gespeichert
qslink_trigger_paare = []
for qslink in complete_dict_tags['QSLINK']:
    if qslink['trigger'] != "":
        qslink_trigger_paare.append((qslink['id'],qslink['trigger']))

#oslink trigger paare
olink_trigger_paare = []
for olink in complete_dict_tags['OLINK']:
    if olink['trigger'] != "":
        olink_trigger_paare.append((olink['id'],olink['trigger']))

for pair in qslink_trigger_paare:
    print(f"QSLINK id {pair[0]} getriggert durch id {pair[1]}")

for pair in olink_trigger_paare:
    print(f"OLINK id {pair[0]} getriggert durch id {pair[1]}")

print("Aufgabe 2.3.6---------------------------------------------------------------------------------------")
motion_dict = {}
for motion in complete_dict_tags['MOTION']:
    motion_dict.setdefault(motion['motion_class'],[]).append(1)
#sortieren nach dem haeufigst vorkommenden verb
sorted_motions = sorted(motion_dict.items(), key = lambda a: len(a[1]))

#filtern der 5 hoechsten vorkomnisse
for pair in list(reversed(sorted_motions))[:5]:
    print(f"{len(pair[1])} x {pair[0]}")
    
#print("Aufgabe 2.4---------------------------------------------------------------------------------------")

