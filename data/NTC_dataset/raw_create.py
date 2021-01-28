from collections import defaultdict
import json
import sys
import pickle
import math
import re


def NTC2json(fname, with_bccwj=True):
  docID = ''
  sentID = 0
  wordID = 0
  d = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str)))))

  skip_flg = False
  start_pos = 7
  end_pos = 16
  for line in open(fname, encoding="utf-8"):
    if with_bccwj and (line.startswith("#! SEGMENT_S") or line.startswith("#! GROUP_S")):
        continue
    if line.startswith("#"):
      if with_bccwj and not skip_flg:
        start_pos = 12
        end_pos = 15
        skip_flg = True
        continue
      if with_bccwj:
        docID = line.split("\t")[2].strip().replace("_", "")
        sentID = 0
        wordID = 0
      else:
        if docID != int(line[start_pos:end_pos]):
          docID = int(line[start_pos:end_pos])
          sentID = 0
          wordID = 0
    elif line.startswith("*"):
      setsuID = int(line.strip().split()[1])
      depID = int(re.match(r"([-]?[0-9])*", line.strip().split()[2]).group(0))
    elif line.startswith("EOS"):
      sentID += 1
    else:
      if with_bccwj:
        info = line.replace("\n", "").split("\t")
        tmp = info[-1]
        items = info[1].split(",")
        info = [info[0]] + items + [tmp]
        info = [info[0], info[7], info[10], info[6], info[5], info[6], info[-2], info[-1]]
      else:
        info = line.strip().split("\t")
      d[docID][sentID][setsuID][wordID]["info"] = info
      d[docID][sentID][setsuID][wordID]["saki"] = depID
      if info[7] != "_":
        for pa in info[7].split():
          k, v = pa.split("=")
          v = v.strip("\"")
          d[docID][sentID][setsuID][wordID][k] = v
      wordID += 1

  return d

def create_PAdict(d):
  pa = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
  predID = 0
  for docID in sorted(d.keys()):
    for sentID in sorted(d[docID].keys()):
      for setsuID in sorted(d[docID][sentID].keys()):
        for wordID, v in sorted(d[docID][sentID][setsuID].items()):
          if v.get("type", "") == "pred":
            pa[docID][sentID][predID]["pred_ID"] = "{}_{}_{}_{}".format(docID, sentID, setsuID, wordID)
            find_argument("ga", docID, sentID, setsuID, predID, v, pa, d)
            find_argument("o", docID, sentID, setsuID, predID, v, pa, d)
            find_argument("ni", docID, sentID, setsuID, predID, v, pa, d)
            predID += 1
            
  return pa

def delete_intra(pa):
  for docID in sorted(pa.keys()):
    for sentID in sorted(pa[docID].keys()):
      for predID in sorted(pa[docID][sentID].keys()):
        if "dep" in pa[docID][sentID][predID]["ga_type"] and "intra" in pa[docID][sentID][predID]["ga_type"]:
          temp_type = list()
          temp_ID = list()
          for _type, _ID in zip(pa[docID][sentID][predID]["ga_type"], pa[docID][sentID][predID]["ga_ID"]):
            if _type == "dep":
              temp_type.append(_type)
              temp_ID.append(_ID)
              break
          pa[docID][sentID][predID]["ga_type"] = temp_type    
          pa[docID][sentID][predID]["ga_ID"] = temp_ID    

        if "dep" in pa[docID][sentID][predID]["o_type"] and "intra" in pa[docID][sentID][predID]["o_type"]:
          temp_type = list()
          temp_ID = list()
          for _type, _ID in zip(pa[docID][sentID][predID]["o_type"], pa[docID][sentID][predID]["o_ID"]):
            if _type == "dep":
              temp_type.append(_type)
              temp_ID.append(_ID)
              break
          pa[docID][sentID][predID]["o_type"] = temp_type    
          pa[docID][sentID][predID]["o_ID"] = temp_ID   

        if "dep" in pa[docID][sentID][predID]["ni_type"] and "intra" in pa[docID][sentID][predID]["ni_type"]:
          temp_type = list()
          temp_ID = list()
          for _type, _ID in zip(pa[docID][sentID][predID]["ni_type"], pa[docID][sentID][predID]["ni_ID"]):
            if _type == "dep":
              temp_type.append(_type)
              temp_ID.append(_ID)
              break
          pa[docID][sentID][predID]["ni_type"] = temp_type    
          pa[docID][sentID][predID]["ni_ID"] = temp_ID    
  
  return pa

def create_Cdict(d):
  cand = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
  candID = 0
  for docID in sorted(d.keys()):
    for sentID in sorted(d[docID].keys()):
      for setsuID in sorted(d[docID][sentID].keys()):
        for wordID, v in sorted(d[docID][sentID][setsuID].items()):
          info = v["info"]
          if info[3].startswith("名詞"):
            cand[docID][sentID][candID]["cand_ID"] = "{}_{}_{}_{}".format(docID, sentID, setsuID, wordID)
            if "id" in v:
              cand[docID][sentID][candID]["cand_type"] = "positive noun"
            else:
              cand[docID][sentID][candID]["cand_type"] = "negative noun"
            candID += 1  
          else:
            if "id" in v:
              cand[docID][sentID][candID]["cand_ID"] = "{}_{}_{}_{}".format(docID, sentID, setsuID, wordID)
              cand[docID][sentID][candID]["cand_type"] = "positive unnoun"
              candID += 1
            # add all token
            else:
              cand[docID][sentID][candID]["cand_ID"] = "{}_{}_{}_{}".format(docID, sentID, setsuID, wordID)
              cand[docID][sentID][candID]["cand_type"] = "negative unnoun"
              candID += 1


  return cand 

def find_argument(kaku, docID, sentID, setsuID, predID, v, pa, d):
  if kaku in v:
    if v[kaku] == "exo1" or v[kaku] == "exo2" or v[kaku] == "exog":
      pa[docID][sentID][predID][kaku+"_ID"] = v[kaku]
      pa[docID][sentID][predID][kaku+"_type"] = "exo"
    else:
      for kaku_sentID in sorted(d[docID].keys()):
        for kaku_setsuID in sorted(d[docID][kaku_sentID].keys()):
          for kaku_wordID, kaku_v in sorted(d[docID][kaku_sentID][kaku_setsuID].items()):
            if kaku_v.get("id", "") == v[kaku]:
              if kaku_sentID == sentID:
                if kaku_v["saki"] == setsuID or v["saki"] == kaku_setsuID:
                  pa[docID][sentID][predID][kaku+"_ID"].append("{}_{}_{}_{}".format(docID, sentID, kaku_setsuID, kaku_wordID))
                  pa[docID][sentID][predID][kaku+"_type"].append("dep")
                else:
                  pa[docID][sentID][predID][kaku+"_ID"].append("{}_{}_{}_{}".format(docID, sentID, kaku_setsuID, kaku_wordID))
                  pa[docID][sentID][predID][kaku+"_type"].append("intra")
              #elif kaku_sentID != sentID:
              #  pa[docID][sentID][predID][kaku+"_ID"].append("{}_{}_{}_{}".format(docID, kaku_sentID, kaku_setsuID, kaku_wordID))
              #  pa[docID][sentID][predID][kaku+"_type"].append("inter")

def create_dataset(d, pa, cand, train_json, more=False, train_f=False):
  w2i = create_w2i(train_json)
  x_dataset = list()
  y_dataset = list()
  z_dataset = list()
  a_dataset = list()
  b_dataset = list()

  for docID, v in sorted(pa.items()):
    for sentID, vv in sorted(v.items()):
      for predID, vvv in sorted(vv.items()):
        if "dep" in vvv["ga_type"] or "intra" in vvv["ga_type"] or "dep" in vvv["o_type"] or "intra" in vvv["o_type"] or "dep" in vvv["ni_type"] or "intra" in vvv["ni_type"]:
          _, _, setsuID, wordID = vvv["pred_ID"].split("_")
          setsuID = int(setsuID)
          wordID = int(wordID)
          
          if d[docID][sentID][setsuID][wordID]["info"][0] in w2i:
            #pred = w2i[d[docID][sentID][setsuID][wordID]["info"][0]]
            pred = d[docID][sentID][setsuID][wordID]["info"][0]
          else:
            #pred = 0
            pred = d[docID][sentID][setsuID][wordID]["info"][0]
          pred_saki = d[docID][sentID][setsuID][wordID]["saki"]
          
          setsu_p = list()
          for bp_wordID in sorted(d[docID][sentID][setsuID].keys()):
            if d[docID][sentID][setsuID][bp_wordID]["info"][0] in w2i:
              #setsu_p.append(w2i[d[docID][sentID][setsuID][bp_wordID]["info"][0]])
              setsu_p.append(d[docID][sentID][setsuID][bp_wordID]["info"][0])
            else:
              #setsu_p.append(0)
              setsu_p.append(d[docID][sentID][setsuID][bp_wordID]["info"][0])
          
          arg_sentID = sentID
          sent_list = list()
          sent_ans = list()
          sent_index = list()  # new
          sent_arg_type = list()
          sent_position = list()
          sent_predicate = list()
          for arg_candID in sorted(cand[docID][arg_sentID].keys()):
            _, _, arg_setsuID, arg_wordID = cand[docID][arg_sentID][arg_candID]["cand_ID"].split("_")
            arg_setsuID = int(arg_setsuID)
            arg_wordID = int(arg_wordID)
            if d[docID][arg_sentID][arg_setsuID][arg_wordID]["info"][0] in w2i:
              #arg = w2i[d[docID][arg_sentID][arg_setsuID][arg_wordID]["info"][0]]
              arg = d[docID][arg_sentID][arg_setsuID][arg_wordID]["info"][0]
            else:
              #arg = 0
              arg = d[docID][arg_sentID][arg_setsuID][arg_wordID]["info"][0]
            arg_saki = d[docID][arg_sentID][arg_setsuID][arg_wordID]["saki"]
            
            sent_predicate.append(d[docID][arg_sentID][arg_setsuID][arg_wordID]["type"])  # predicate type add
            #print(d[docID][arg_sentID][arg_setsuID][arg_wordID]["info"][0], d[docID][arg_sentID][arg_setsuID][arg_wordID]["type"])  # predicate type add

            setsu_a = list()
            for ba_wordID in sorted(d[docID][arg_sentID][arg_setsuID].keys()):
              if d[docID][arg_sentID][arg_setsuID][ba_wordID]["info"][0] in w2i:
                #setsu_a.append(w2i[d[docID][arg_sentID][arg_setsuID][ba_wordID]["info"][0]])
                setsu_a.append(d[docID][arg_sentID][arg_setsuID][ba_wordID]["info"][0])
              else:
                #setsu_a.append(0)
                setsu_a.append(d[docID][arg_sentID][arg_setsuID][ba_wordID]["info"][0])
            if more:
              mae_setsuID = arg_setsuID-1
              ato_setsuID = arg_setsuID+1
              mae_setsu_a = list()
              ato_setsu_a = list()
              if mae_setsuID in d[docID][arg_sentID]:
                for ba_wordID in sorted(d[docID][arg_sentID][mae_setsuID].keys()):
                  if d[docID][arg_sentID][mae_setsuID][ba_wordID]["info"][0] in w2i:
                    mae_setsu_a.append(w2i[d[docID][arg_sentID][mae_setsuID][ba_wordID]["info"][0]])
                  else:
                    mae_setsu_a.append(0)
              else:
                mae_setsu_a.append(-1)
              if ato_setsuID in d[docID][arg_sentID]:
                for ba_wordID in sorted(d[docID][arg_sentID][ato_setsuID].keys()):
                  if d[docID][arg_sentID][ato_setsuID][ba_wordID]["info"][0] in w2i:
                    ato_setsu_a.append(w2i[d[docID][arg_sentID][ato_setsuID][ba_wordID]["info"][0]])
                  else:
                    ato_setsu_a.append(0)
              else:
                ato_setsu_a.append(-1)
              sent_list.append([arg, mae_setsu_a, setsu_a, ato_setsu_a])
            
            else:  
              sent_list.append([arg, setsu_a])
           
            ans = list()
            index = list()  # new
            arg_type = list()
            position = list()
            flag = False
            #if train_f:
            #  ID1, labelID1 = "ni_ID", 2
            #  ID2, labelID2 = "o_ID", 1
            #  ID3, labelID3 = "ga_ID", 0
            #else:
            #  ID1, labelID1 = "ga_ID", 0
            #  ID2, labelID2 = "o_ID", 1
            #  ID3, labelID3 = "ni_ID", 2
            
            ID1, labelID1 = "ni_ID", 2
            ID2, labelID2 = "o_ID", 1
            ID3, labelID3 = "ga_ID", 0

            for i, ni_id in enumerate(vvv[ID1]):
              if ni_id == cand[docID][arg_sentID][arg_candID]["cand_ID"]:
                ans.append(labelID1)
                index.append(math.fabs(wordID-arg_wordID))  # new
                position.append((create_position_feature(wordID, arg_wordID, setsuID, arg_setsuID)))
                if setsuID == arg_setsuID:
                  arg_type.append("pred")
                elif arg_saki == setsuID:
                  arg_type.append("dep")
                elif pred_saki == arg_setsuID:
                  arg_type.append("rentai")
                else:
                  arg_type.append("zero")
                flag = True
            if not flag:
              for i, o_id in enumerate(vvv[ID2]):
                if o_id == cand[docID][arg_sentID][arg_candID]["cand_ID"]:
                  ans.append(labelID2)
                  index.append(math.fabs(wordID-arg_wordID))  # new
                  position.append((create_position_feature(wordID, arg_wordID, setsuID, arg_setsuID)))
                  if setsuID == arg_setsuID:
                    arg_type.append("pred")
                  elif arg_saki == setsuID:
                    arg_type.append("dep")
                  elif pred_saki == arg_setsuID:
                    arg_type.append("rentai")
                  else:
                    arg_type.append("zero")
                  flag = True
            if not flag:
              for i, ga_id in enumerate(vvv[ID3]):
                if ga_id == cand[docID][arg_sentID][arg_candID]["cand_ID"]:
                  ans.append(labelID3)
                  index.append(math.fabs(wordID-arg_wordID))  # new
                  position.append((create_position_feature(wordID, arg_wordID, setsuID, arg_setsuID)))
                  if setsuID == arg_setsuID:
                    arg_type.append("pred")
                  elif arg_saki == setsuID:
                    arg_type.append("dep")
                  elif pred_saki == arg_setsuID:
                    arg_type.append("rentai")
                  else:
                    arg_type.append("zero")
                  flag = True
            if not flag:
              ans.append(3)
              index.append(math.fabs(wordID-arg_wordID))  # new
              position.append((create_position_feature(wordID, arg_wordID, setsuID, arg_setsuID)))
              if setsuID == arg_setsuID:
                arg_type.append("pred")
              elif arg_saki == setsuID:
                arg_type.append("dep")
              elif pred_saki == arg_setsuID:
                arg_type.append("rentai")
              else:
                arg_type.append("zero")
            
            if len(ans) == 1:
              sent_ans.append(ans[0])
              sent_index.append(index[0])  # new
              sent_arg_type.append(arg_type[0])
              sent_position.append(position[0])
            else:
              print("oi!oi!!oi!!!")
              sent_list.pop()
              for label, i, t, p in zip(ans, index, arg_type, position):
                sent_ans.append(label)
                sent_index.append(i)  # new
                sent_arg_type.append(t)
                sent_position.append(p)
                if more:
                  sent_list.append([arg, mae_setsu_a, setsu_a, ato_setsu_a])
                else:  
                  sent_list.append([arg, setsu_a])

          if len(sent_ans) != len(sent_index):
            print('oi!')

          if not train_f:
            # check
            ga_list = list()
            o_list = list()
            ni_list = list()
            new_sent_ans = sent_ans

            for i, label in enumerate(sent_ans):
              if label == 0:
                ga_list.append(i)
              elif label == 1:
                o_list.append(i)
              elif label == 2:
                ni_list.append(i)
            
            if len(ga_list) > 1:
              dist = dict()
              min_dist = 100000
              for i in ga_list:
                dist[i] = sent_index[i]  # key=index, value=dist
                if min_dist > sent_index[i]:
                  min_dist = sent_index[i]
                  min_i = i
              for k,v in sorted(dist.items()):
                if k != min_i:
                  new_sent_ans[k] = 3
                
            if len(o_list) > 1:
              dist = dict()
              min_dist = 100000
              for i in o_list:
                dist[i] = sent_index[i]  # key=index, value=dist
                if min_dist > sent_index[i]:
                  min_dist = sent_index[i]
                  min_i = i
              for k,v in sorted(dist.items()):
                if k != min_i:
                  new_sent_ans[k] = 3

            if len(ni_list) > 1:
              dist = dict()
              min_dist = 100000
              for i in ni_list:
                dist[i] = sent_index[i]  # key=index, value=dist
                if min_dist > sent_index[i]:
                  min_dist = sent_index[i]
                  min_i = i
              for k,v in sorted(dist.items()):
                if k != min_i:
                  new_sent_ans[k] = 3
            
            sent_ans = new_sent_ans

          x_dataset.append([pred, setsu_p, sent_list])
          y_dataset.append(sent_ans)
          z_dataset.append(sent_arg_type)
          a_dataset.append(sent_position)
          b_dataset.append(sent_predicate)

  return x_dataset, y_dataset, z_dataset, a_dataset, b_dataset

def create_collocated_sentence_ntc():
  next_sentences = {mode: {} for mode in ['train', 'dev', 'test']}
  previous_sentences = {mode: {} for mode in ['train', 'dev', 'test']}
  for mode in ['train', 'dev', 'test']:
    with Path(f'./{mode}.txt').open('r') as f:
      texts = f.readlines()

    sentence_id = ''
    previous_sentence, current_sentence = '', ''
    previous_sentence_words, current_sentence_words = [], []
    for text in texts:
      if text.startswith('# S-ID:'):
        is_next_sentence = True if sentence_id == text[7: 16] else False
        sentence_id = text[7: 16]
        continue
      elif text.startswith('* '):
        continue
      elif text.startswith('EOS'):
        if is_next_sentence:
          if previous_sentence in next_sentences[mode].keys() and ''.join(next_sentences[mode][previous_sentence]) != current_sentence:
            print(f'[{mode}]: [{sentence_id}]' + ', '.join([previous_sentence, ''.join(next_sentences[mode][previous_sentence]), current_sentence]))
          next_sentences[mode][previous_sentence] = current_sentence_words
          previous_sentences[mode][current_sentence] = previous_sentence_words

        previous_sentence, previous_sentence_words = current_sentence, current_sentence_words.copy()
        current_sentence, current_sentence_words = '', []
      else:
        word = text.split('\t')[0]
        current_sentence = current_sentence + word
        current_sentence_words.append(word)

  return next_sentences['train'], next_sentences['dev'], next_sentences['test'], previous_sentences['train'], previous_sentences['dev'], previous_sentences['test']

def create_collocated_sentence_bccwj():
  modes = ["train", "dev", "test"]
  listpath = {mode: Path(f'../BCCWJ-DepParaPAS/file_{mode}.txt') for mode in modes}
  filelists = {}
  for mode in modes:
    with listpath[mode].open('r', encoding='utf-8') as f:
      filelists[mode] = [line.rstrip() for line in f.readlines()]

  base_path = Path(
    "../BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/raw")
  datalist = {mode: [] for mode in modes}

  next_sentences = {mode: {} for mode in modes}
  previous_sentences = {mode: {} for mode in modes}
  for mode in modes:
    for filename in filelists[mode]:
      sentence_id = ''
      previous_sentence, current_sentence = '', ''
      previous_sentence_words, current_sentence_words = [], []
      with Path(str((base_path / filename).resolve()).replace(".conll", ".cabocha")).open('r') as f:
        texts = f.readlines()

      is_next_sentence = False
      for text in texts:
        if text.startswith('#! '):
          continue
        elif text.startswith('* '):
          continue
        elif text.startswith('EOS'):
          if is_next_sentence:
            if previous_sentence in next_sentences[mode].keys() and ''.join(
                    next_sentences[mode][previous_sentence]) != current_sentence:
              print(f'[{mode}]: [{filename}]' + ', '.join(
                [previous_sentence, ''.join(next_sentences[mode][previous_sentence]), current_sentence]))
            next_sentences[mode][previous_sentence] = current_sentence_words
            previous_sentences[mode][current_sentence] = previous_sentence_words

          previous_sentence, previous_sentence_words = current_sentence, current_sentence_words.copy()
          current_sentence, current_sentence_words = '', []
          is_next_sentence = True
        else:
          word = text.split('\t')[0]
          current_sentence = current_sentence + word
          current_sentence_words.append(word)
  return next_sentences['train'], next_sentences['dev'], next_sentences['test'], previous_sentences['train'], previous_sentences['dev'], previous_sentences['test']

def create_w2i(d):  # from train data
  count = defaultdict(int)
  for docID, v in sorted(d.items()):
    for sentID, vv in sorted(v.items()):
      for setsuID, vvv in sorted(vv.items()):
        for wordID, vvvv in sorted(vvv.items()):
          surface = vvvv["info"][0]
          count[surface] += 1

  w2i = defaultdict(lambda: len(w2i)+1)
  for docID, v in sorted(d.items()):
    for sentID, vv in sorted(v.items()):
      for setsuID, vvv in sorted(vv.items()):
        for wordID, vvvv in sorted(vvv.items()):
          surface = vvvv["info"][0]
          if count[surface] >= 2:
            w2i[surface]
          else:
            w2i[surface] = 0

  return w2i

def create_wp2i():
  wp2i = dict()
  for i in range(10):
    wp2i[str(i)] = i
  wp2i["10+"] = 10
  c = 11
  for i in range(-1,-10,-1):
    wp2i[str(i)] = c
    c += 1
  wp2i["-10+"] = c
  
  return wp2i

def create_sp2i():
  sp2i = dict()
  for i in range(10):
    sp2i[str(i)] = i
  sp2i["10+"] = 10
  c = 11
  for i in range(-1,-10,-1):
    sp2i[str(i)] = c
    c += 1
  sp2i["-10+"] = c
  
  return sp2i

def create_position_feature(wordID, arg_wordID, setsuID, arg_setsuID):
  word_position = wordID - arg_wordID
  setsu_position = setsuID - arg_setsuID
  if word_position >= 10:
    wp = wp2i["10+"]
  elif word_position < 10 and word_position > -10:
    wp = wp2i[str(word_position)]
  elif word_position <= -10:
    wp = wp2i["-10+"]
  
  if setsu_position >= 10:
    sp = sp2i["10+"]
  elif setsu_position < 10 and setsu_position > -10:
    sp = sp2i[str(setsu_position)]
  elif setsu_position <= -10:
    sp = sp2i["-10+"]
  
  return wp, sp


if __name__ == "__main__":
  from pathlib import Path
  with_bccwj = True
  if with_bccwj:
    ns_train, ns_dev, ns_test, ps_train, ps_dev, ps_test = create_collocated_sentence_bccwj()
    tags = ["train", "dev", "test"]
    listpath = {"train": Path("../BCCWJ-DepParaPAS/file_train.txt"),
                "dev": Path("../BCCWJ-DepParaPAS/file_dev.txt"),
                "test": Path("../BCCWJ-DepParaPAS/file_test.txt")}
    filelists = {}
    for tag in tags:
      with listpath[tag].open('r', encoding='utf-8') as f:
        filelists[tag] = [line.rstrip() for line in f.readlines()]

    base_path = Path(
      "../BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/raw")
    datalist = {"train": [], "dev": [], "test": []}
    for tag in tags:
      for filename in filelists[tag]:
        datalist[tag].append(NTC2json(str((base_path / filename).resolve()).replace(".conll", ".cabocha"), with_bccwj=with_bccwj))

    dataset = {"train": datalist["train"][0], "dev": datalist["dev"][0], "test": datalist["test"][0]}

    for tag in tags:
      for item in datalist[tag]: ## 結果がlistで返ってくるのでdefault dictに入れ直す
        for key in item.keys():
          if key not in dataset[tag].keys():
            dataset[tag][key] = item[key]
    train, dev, test = dataset["train"], dataset["dev"], dataset["test"]

  else:
    ns_train, ns_dev, ns_test, ps_train, ps_dev, ps_test = create_collocated_sentence_ntc()
    base_path = Path("../NTC_dataset")
    train = NTC2json(str((base_path / "train.txt").resolve()), with_bccwj=with_bccwj)
    dev = NTC2json(str((base_path / "dev.txt").resolve()), with_bccwj=with_bccwj)
    test = NTC2json(str((base_path / "test.txt").resolve()), with_bccwj=with_bccwj)

  w2i = create_w2i(train)
  print(len(w2i))
  wp2i = create_wp2i()
  sp2i = create_sp2i()
  wp2i["pad"] = len(wp2i)
  sp2i["pad"] = len(sp2i)
  #with open("/work/omori/pasa/data/NTC_dataset/train/wp2i.pkl", mode="wb") as f:
  #  pickle.dump(wp2i, f)
  #with open("/work/omori/pasa/data/NTC_dataset/train/sp2i.pkl", mode="wb") as f:
  #  pickle.dump(sp2i, f)
  #exit()  

  pa_train = create_PAdict(train)
  pa_train = delete_intra(pa_train)  # for ouchi
  cand_train = create_Cdict(train)
  pa_dev = create_PAdict(dev)
  pa_dev = delete_intra(pa_dev)  # for ouchi
  cand_dev = create_Cdict(dev)
  pa_test = create_PAdict(test)
  pa_test = delete_intra(pa_test)  # for ouchi
  cand_test = create_Cdict(test)
  
  #x_train, y_train, z_train, a_train, b_train = create_dataset(train, pa_train, cand_train, train, train_f=True)
  x_train, y_train, z_train, a_train, b_train = create_dataset(train, pa_train, cand_train, train)
  x_dev, y_dev, z_dev, a_dev, b_dev = create_dataset(dev, pa_dev, cand_dev, train)
  x_test, y_test, z_test, a_test, b_test = create_dataset(test, pa_test, cand_test, train)
  #with open("raw_train_sahen.pkl", mode="wb") as f:
  #  pickle.dump(list(zip(x_train, y_train, z_train, a_train)), f)
  #with open("raw_dev_sahen.pkl", mode="wb") as f:
  #  pickle.dump(list(zip(x_dev, y_dev, z_dev, a_dev)), f)
  #with open("raw_test_sahen.pkl", mode="wb") as f:
  #  pickle.dump(list(zip(x_test, y_test, z_test, a_test)), f)
  #with open("/work/omori/multi_pasa/data/NTC_dataset/train/train_sahen_multip.pkl", mode="wb") as f:
  #  pickle.dump(list(zip(x_train, y_train, z_train, a_train, b_train)), f)
  #with open("/work/omori/multi_pasa/data/NTC_dataset/dev/dev_sahen_multip.pkl", mode="wb") as f:
  #  pickle.dump(list(zip(x_dev, y_dev, z_dev, a_dev, b_dev)), f)
  #with open("/work/omori/multi_pasa/data/NTC_dataset/test/test_sahen_multip.pkl", mode="wb") as f:
  #  pickle.dump(list(zip(x_test, y_test, z_test, a_test, b_test)), f)

  if with_bccwj:
    # with base_path.joinpath("raw_train_bccwj.pkl").open(mode="wb") as f:
    #   pickle.dump(list(zip(x_train, y_train, z_train, a_train, b_train)), f)
    # with base_path.joinpath("raw_dev_bccwj.pkl").open(mode="wb") as f:
    #   pickle.dump(list(zip(x_dev, y_dev, z_dev, a_dev, b_dev)), f)
    # with base_path.joinpath("raw_test_bccwj.pkl").open(mode="wb") as f:
    #   pickle.dump(list(zip(x_test, y_test, z_test, a_test, b_test)), f)

    # with base_path.joinpath("next_sentence_train_bccwj.pkl").open(mode="wb") as f:
    #   pickle.dump(ns_train, f)
    # with base_path.joinpath("next_sentence_dev_bccwj.pkl").open(mode="wb") as f:
    #   pickle.dump(ns_dev, f)
    # with base_path.joinpath("next_sentence_test_bccwj.pkl").open(mode="wb") as f:
    #   pickle.dump(ns_test, f)
    #
    # with base_path.joinpath("next_sentence_train_bccwj.txt").open(mode="w") as f:
    #   for key, value in ns_train.items():
    #     value = ''.join(value)
    #     f.write(f'{key}, {value}\n')
    # with base_path.joinpath("next_sentence_dev_bccwj.txt").open(mode="w") as f:
    #   for key, value in ns_dev.items():
    #     value = ''.join(value)
    #     f.write(f'{key}, {value}\n')
    # with base_path.joinpath("next_sentence_test_bccwj.txt").open(mode="w") as f:
    #   for key, value in ns_test.items():
    #     value = ''.join(value)
    #     f.write(f'{key}, {value}\n')


    with base_path.joinpath("previous_sentence_train_bccwj.pkl").open(mode="wb") as f:
      pickle.dump(ps_train, f)
    with base_path.joinpath("previous_sentence_dev_bccwj.pkl").open(mode="wb") as f:
      pickle.dump(ps_dev, f)
    with base_path.joinpath("previous_sentence_test_bccwj.pkl").open(mode="wb") as f:
      pickle.dump(ps_test, f)

    with base_path.joinpath("previous_sentence_train_bccwj.txt").open(mode="w") as f:
      for key, value in ps_train.items():
        value = ''.join(value)
        f.write(f'{key}, {value}\n')
    with base_path.joinpath("previous_sentence_dev_bccwj.txt").open(mode="w") as f:
      for key, value in ps_dev.items():
        value = ''.join(value)
        f.write(f'{key}, {value}\n')
    with base_path.joinpath("previous_sentence_test_bccwj.txt").open(mode="w") as f:
      for key, value in ps_test.items():
        value = ''.join(value)
        f.write(f'{key}, {value}\n')

  else:
    # with base_path.joinpath("raw_train.pkl").open(mode="wb") as f:
    #   pickle.dump(list(zip(x_train, y_train, z_train, a_train, b_train)), f)
    # with base_path.joinpath("raw_dev.pkl").open(mode="wb") as f:
    #   pickle.dump(list(zip(x_dev, y_dev, z_dev, a_dev, b_dev)), f)
    # with base_path.joinpath("raw_test.pkl").open(mode="wb") as f:
    #   pickle.dump(list(zip(x_test, y_test, z_test, a_test, b_test)), f)

    # with base_path.joinpath("next_sentence_train.pkl").open(mode="wb") as f:
    #   pickle.dump(ns_train, f)
    # with base_path.joinpath("next_sentence_dev.pkl").open(mode="wb") as f:
    #   pickle.dump(ns_dev, f)
    # with base_path.joinpath("next_sentence_test.pkl").open(mode="wb") as f:
    #   pickle.dump(ns_test, f)
    #
    # with base_path.joinpath("next_sentence_train.txt").open(mode="w") as f:
    #   for key, value in ns_train.items():
    #     value = ''.join(value)
    #     f.write(f'{key}, {value}\n')
    # with base_path.joinpath("next_sentence_dev.txt").open(mode="w") as f:
    #   for key, value in ns_dev.items():
    #     value = ''.join(value)
    #     f.write(f'{key}, {value}\n')
    # with base_path.joinpath("next_sentence_test.txt").open(mode="w") as f:
    #   for key, value in ns_test.items():
    #     value = ''.join(value)
    #     f.write(f'{key}, {value}\n')


    with base_path.joinpath("previous_sentence_train.pkl").open(mode="wb") as f:
      pickle.dump(ps_train, f)
    with base_path.joinpath("previous_sentence_dev.pkl").open(mode="wb") as f:
      pickle.dump(ps_dev, f)
    with base_path.joinpath("previous_sentence_test.pkl").open(mode="wb") as f:
      pickle.dump(ps_test, f)

    with base_path.joinpath("previous_sentence_train.txt").open(mode="w") as f:
      for key, value in ps_train.items():
        value = ''.join(value)
        f.write(f'{key}, {value}\n')
    with base_path.joinpath("previous_sentence_dev.txt").open(mode="w") as f:
      for key, value in ps_dev.items():
        value = ''.join(value)
        f.write(f'{key}, {value}\n')
    with base_path.joinpath("previous_sentence_test.txt").open(mode="w") as f:
      for key, value in ps_test.items():
        value = ''.join(value)
        f.write(f'{key}, {value}\n')

  


