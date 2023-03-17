# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
sentiment_word_list = ['positive', 'negative', 'neutral']
# laptop_acos_aspect_cate_list = ['OS#DESIGN_FEATURES', 'SOFTWARE#USABILITY', 'SHIPPING#OPERATION_PERFORMANCE', 'HARD_DISC#DESIGN_FEATURES', 'KEYBOARD#GENERAL', 'DISPLAY#PRICE', 'DISPLAY#OPERATION_PERFORMANCE', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'LAPTOP#MISCELLANEOUS', 'BATTERY#GENERAL', 'SUPPORT#QUALITY', 'PORTS#OPERATION_PERFORMANCE', 'MEMORY#GENERAL', 'PORTS#USABILITY', 'HARDWARE#DESIGN_FEATURES', 'MOTHERBOARD#QUALITY', 'SOFTWARE#PORTABILITY', 'LAPTOP#PRICE', 'GRAPHICS#USABILITY', 'LAPTOP#DESIGN_FEATURES', 'WARRANTY#GENERAL', 'HARD_DISC#GENERAL', 'CPU#OPERATION_PERFORMANCE', 'LAPTOP#GENERAL', 'PORTS#GENERAL', 'BATTERY#OPERATION_PERFORMANCE', 'HARD_DISC#OPERATION_PERFORMANCE', 'MULTIMEDIA_DEVICES#QUALITY', 'FANS&COOLING#GENERAL', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY', 'PORTS#QUALITY', 'KEYBOARD#USABILITY', 'MEMORY#DESIGN_FEATURES', 'CPU#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE', 'SUPPORT#DESIGN_FEATURES', 'SOFTWARE#QUALITY', 'MEMORY#USABILITY', 'GRAPHICS#DESIGN_FEATURES', 'POWER_SUPPLY#CONNECTIVITY', 'HARDWARE#USABILITY', 'GRAPHICS#GENERAL', 'MULTIMEDIA_DEVICES#PRICE', 'MULTIMEDIA_DEVICES#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'FANS&COOLING#OPERATION_PERFORMANCE', 'POWER_SUPPLY#QUALITY', 'HARDWARE#QUALITY', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OPTICAL_DRIVES#DESIGN_FEATURES', 'DISPLAY#DESIGN_FEATURES', 'Out_Of_Scope#GENERAL', 'COMPANY#OPERATION_PERFORMANCE', 'SOFTWARE#PRICE', 'GRAPHICS#OPERATION_PERFORMANCE', 'BATTERY#DESIGN_FEATURES', 'KEYBOARD#DESIGN_FEATURES', 'KEYBOARD#QUALITY', 'SOFTWARE#DESIGN_FEATURES', 'POWER_SUPPLY#GENERAL', 'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'CPU#GENERAL', 'KEYBOARD#PORTABILITY', 'DISPLAY#QUALITY', 'OPTICAL_DRIVES#GENERAL', 'POWER_SUPPLY#OPERATION_PERFORMANCE', 'MOUSE#GENERAL', 'HARDWARE#GENERAL', 'SOFTWARE#GENERAL', 'PORTS#CONNECTIVITY', 'LAPTOP#QUALITY', 'MEMORY#QUALITY', 'OS#PRICE', 'SOFTWARE#OPERATION_PERFORMANCE', 'Out_Of_Scope#OPERATION_PERFORMANCE', 'SUPPORT#GENERAL', 'DISPLAY#GENERAL', 'CPU#QUALITY', 'SHIPPING#GENERAL', 'COMPANY#PRICE', 'DISPLAY#USABILITY', 'FANS&COOLING#QUALITY', 'MOUSE#DESIGN_FEATURES', 'OS#OPERATION_PERFORMANCE', 'OS#GENERAL', 'SUPPORT#PRICE', 'COMPANY#QUALITY', 'WARRANTY#QUALITY', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE', 'OS#QUALITY', 'BATTERY#QUALITY', 'FANS&COOLING#DESIGN_FEATURES', 'LAPTOP#CONNECTIVITY', 'SUPPORT#OPERATION_PERFORMANCE', 'LAPTOP#PORTABILITY', 'SHIPPING#PRICE', 'HARD_DISC#QUALITY', 'LAPTOP#USABILITY', 'CPU#PRICE', 'POWER_SUPPLY#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#CONNECTIVITY', 'OS#USABILITY', 'SHIPPING#QUALITY', 'COMPANY#GENERAL', 'PORTS#DESIGN_FEATURES']
laptop_acos_aspect_cate_list = ['os design_features', 'software usability', 'shipping operation_performance', 'hard_disc design_features', 'keyboard general', 'display price', 'display operation_performance', 'multimedia_devices operation_performance', 'optical_drives operation_performance', 'laptop miscellaneous', 'battery general', 'support quality', 'ports operation_performance', 'memory general', 'ports usability', 'hardware design_features', 'motherboard quality', 'software portability', 'laptop price', 'graphics usability', 'laptop design_features', 'warranty general', 'hard_disc general', 'cpu operation_performance', 'laptop general', 'ports general', 'battery operation_performance', 'hard_disc operation_performance', 'multimedia_devices quality', 'fans&cooling general', 'hard_disc miscellaneous', 'ports portability', 'ports quality', 'keyboard usability', 'memory design_features', 'cpu design_features', 'hardware operation_performance', 'support design_features', 'software quality', 'memory usability', 'graphics design_features', 'power_supply connectivity', 'hardware usability', 'graphics general', 'multimedia_devices price', 'multimedia_devices general', 'motherboard operation_performance', 'fans&cooling operation_performance', 'power_supply quality', 'hardware quality', 'multimedia_devices design_features', 'optical_drives design_features', 'display design_features', 'out_of_scope general', 'company operation_performance', 'software price', 'graphics operation_performance', 'battery design_features', 'keyboard design_features', 'keyboard quality', 'software design_features', 'power_supply general', 'company design_features', 'memory operation_performance', 'cpu general', 'keyboard portability', 'display quality', 'optical_drives general', 'power_supply operation_performance', 'mouse general', 'hardware general', 'software general', 'ports connectivity', 'laptop quality', 'memory quality', 'os price', 'software operation_performance', 'out_of_scope operation_performance', 'support general', 'display general', 'cpu quality', 'shipping general', 'company price', 'display usability', 'fans&cooling quality', 'mouse design_features', 'os operation_performance', 'os general', 'support price', 'company quality', 'warranty quality', 'laptop operation_performance', 'keyboard operation_performance', 'os quality', 'battery quality', 'fans&cooling design_features', 'laptop connectivity', 'support operation_performance', 'laptop portability', 'shipping price', 'hard_disc quality', 'laptop usability', 'cpu price', 'power_supply design_features', 'multimedia_devices connectivity', 'os usability', 'shipping quality', 'company general', 'ports design_features']
# res_acos_aspect_cate_list = ['RESTAURANT#PRICES', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL', 'DRINKS#QUALITY', 'RESTAURANT#GENERAL', 'LOCATION#GENERAL', 'FOOD#PRICES', 'FOOD#QUALITY', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#PRICES', 'DRINKS#STYLE_OPTIONS', 'FOOD#STYLE_OPTIONS']
res_acos_aspect_cate_list = ['restaurant prices', 'service general', 'ambience general', 'drinks quality', 'restaurant general', 'location general', 'food prices', 'food quality', 'restaurant miscellaneous', 'drinks prices', 'drinks style_options', 'food style_options']


# 原始数据读取 data_path +  -> text:list[str]  label:list[str]
def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels

def read_line_examples_from_acos(data_path, E_I='ALL'):
    text = ''
    lines = []
    items = []
    with open(data_path, 'r') as f:
        text = f.read()
    lines = text.split('\n')
    for l in lines:
        items.append(l.split('\t'))
    acos_texts = []
    acos_labels =[]
    for i in items:
        acos_texts.append(i[:1])
        acos_labels.append(i[1:])
    print("sentiment type is: ", E_I)
    # 过滤EAEO等情况
    def _filter_(E_I):
        _filtered_text = []
        _filtered_labels = []
        for i in zip(acos_texts, acos_labels):
            _texts = i[0]
            _labels = i[1]
            if E_I == "EAEO":
                _filter_labels = list(filter(lambda x: x[:5]!='-1,-1' and x[-5:]!='-1,-1', _labels))
            if E_I == "IAEO":
                _filter_labels = list(filter(lambda x: x[:5]=='-1,-1' and x[-5:]!='-1,-1', _labels))
            if E_I == "EAIO":
                _filter_labels = list(filter(lambda x: x[-5:]=='-1,-1' and x[:5]!='-1,-1', _labels))
            if E_I == "IAIO":
                _filter_labels = list(filter(lambda x: x[:5]=='-1,-1' and x[-5:]=='-1,-1', _labels))
            if _filter_labels != []:
                _filtered_text.append(_texts)
                _filtered_labels.append(_filter_labels)              
        return _filtered_text, _filtered_labels
    
    if E_I == "ALL":
        return acos_texts, acos_labels
    else:
        new_texts, new_labels = _filter_(E_I)
        print(f"the len of {E_I} is", len(new_texts), " and ", len(new_labels))
        return new_texts, new_labels

def get_transformed_io_acos(acos_texts, acos_labels):
    # lyx add
    # transfomrmed acos_texts
    # acos_items = read_line_examples_from_acos(data_path)
    # acos_texts = []
    # acos_labels =[]
    # for i in acos_items:
    #     acos_texts.append(i[:1])
    #     acos_labels.append(i[1:])
    
    acos_texts_splited = []
    acos_labels_splited = []
    for i, acos_text in enumerate(acos_texts):
        acos_texts_splited.append(acos_text[0].split(" "))
    
    def _senttag2word_and_get_index(x):
        # get senttag2word
        if len(x) == 1:
            if int(x) == 0:
                x = "negative"
            elif int(x) == 1:
                x = "neutral"
            else:
                x = "positive"
        # get index
        if ',' in x and '-1' not in x.split(","):
            pos = x.index(",")
            x = x[:pos] + "," + str((int(x.split(",")[-1]) - 1))
        else:
            pass
        return x

    acos_labels_splited = []
    for i in acos_labels:
        acos_one_label_splited = []
        for j in i:
            split_one_label = j.split(" ")
            split_one_label = list(map(_senttag2word_and_get_index, split_one_label))
            acos_one_label_splited.append(split_one_label)
        acos_labels_splited.append(acos_one_label_splited)


    acos_texts_and_labels = list(zip(acos_texts_splited, acos_labels_splited))
    # transform all index into word
    index2word_labels = []
    for index, t_l in enumerate(acos_texts_and_labels):
        def _replace_word(x):
                if ',' in x and '-1' not in x.split(","):
                    pos = x.index(",")
                    start = int(x[:pos]) # 1
                    # print("start is", start)
                    end = int(x[pos:].replace(',', ''))  # 4 youwyx
                    # print("end is", end)
                    x_list = t_l[0][start:end+1]
                    x_sent = ' '.join(x_list)
                    # print("x_sent is:", x_sent) 
                    x = x_sent
                if ',' in x and '-1' in x.split(","):
                    x = 'NULL'
                return x
        index2word_one_label = []
        for l in t_l[1]:
            index2word_one_label.append(list(map(_replace_word, l)))
        index2word_labels.append(index2word_one_label)
    # transformed index2word_labels to string 
    word_labels = []
    for i in index2word_labels:
        temp_labels = []
        for j in i:
            temp_one_label = ', '.join(j)
            temp_one_label = '(' + temp_one_label + ')'
            temp_labels.append(temp_one_label)
        word_labels.append('; '.join(temp_labels))
    return acos_texts_splited, word_labels

def listtostr(arr):
    return " ".join(arr)

# 生成为模板target 和target句子
def get_para_acos_targets(texts, labels, use_sent_flag, use_prompt_flag):
    goal_name_list = ['find [MASK] aspect, [MASK] opinion.']
    # # 是否用句子生成
    # use_sent_flag = False
    # # 是否用prompt
    # use_prompt_flag = False
    # 计算
    count = 0
    EA_EO_count = 0
    EA_IO_count = 0
    IA_EO_count = 0
    IA_IO_count = 0
    #输入带有<aspect>等的input
    targets = []
    template_text = []
    # index为texts索引
    for index, label in enumerate(labels):
        label = label.replace('(','')
        label = label.replace(')','')
        one_sent_label = label.split('; ')
        count += 1
        
        template = []
        querys = []
        # index_label为label里面的每个label的索引,方便<aspect_0>
        for index_label, o_l in enumerate(one_sent_label):
            # debug o_l
            # print(o_l)
            
            if o_l.split(", ")[0] != 'NULL' and o_l.split(", ")[3] != 'NULL':
                # EA_EO_count += 1
                a = o_l.split(", ")[0]
                c = o_l.split(", ")[1].replace("#", " ").lower()
                s = o_l.split(", ")[2]
                s = sentword2opinion[s]
                o = o_l.split(", ")[3]
                goal = goal_name_list[0]
                if use_sent_flag:
                    # one_template = f"It is explicit aspect, explicit opinion, {c} is {s} because {a} is {o}."
                    one_template = f"{c} is {s} because {a} is {o}."
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
                if use_prompt_flag:
                    text_one_template = "".join(one_template.replace(c, "<extra_id_0>").replace(s, "<extra_id_1>").replace(o, "[aspect]").replace(a, "[opinion]"))
                    # text_one_template = "[<extra_id_0> is <extra_id_1> because [aspect] is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because it is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because of the [aspect]. \
                    # <extra_id_0> is <extra_id_1> taking everything into account.]"
                    query = f"goal: {goal_name_list[0]} \n input: {listtostr(texts[index])} \n sentiment options: [great,ok,bad] \n target options: {text_one_template}"
                else:
                    query = listtostr(texts[index])
                    
            if o_l.split(", ")[0] == 'NULL' and o_l.split(", ")[3] != 'NULL':
                # EA_IO_count += 1
                a = o_l.split(", ")[0]
                c = o_l.split(", ")[1].replace("#", " ").lower()
                s = o_l.split(", ")[2]
                s = sentword2opinion[s]
                o = o_l.split(", ")[3]
                goal = goal_name_list[0]
                if use_sent_flag:
                    # one_template = f"It is explicit aspect, implicit opinion, {c} is {s} because it is {o}."
                    one_template = f"{c} is {s} because it is {o}."
                    
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
                if use_prompt_flag:
                    text_one_template = "".join(one_template.replace(c, "<extra_id_0>").replace(s, "<extra_id_1>").replace(o, "<extra_id_2>").replace(a, "<extra_id_3>"))
                    # text_one_template = "[<extra_id_0> is <extra_id_1> because [aspect] is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because it is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because of the [aspect]. \
                    # <extra_id_0> is <extra_id_1> taking everything into account.]"
                    query = f"goal: {goal_name_list[0]} \n input: {listtostr(texts[index])} \n sentiment options: [great,ok,bad] \n target options: {text_one_template}"
                else:
                    query = listtostr(texts[index])
                    
            if o_l.split(", ")[0] != 'NULL' and o_l.split(", ")[3] == 'NULL':
                # IA_EO_count += 1
                a = o_l.split(", ")[0]
                c = o_l.split(", ")[1].replace("#", " ").lower()
                s = o_l.split(", ")[2]
                s = sentword2opinion[s]
                o = o_l.split(", ")[3]
                goal = goal_name_list[0]
                if use_sent_flag:
                    # one_template = f"It is implicit aspect, explicit opinion, {c} is {s} because of the {a}."
                    one_template = f"{c} is {s} because of the {a}."
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
                if use_prompt_flag:
                    text_one_template = "".join(one_template.replace(c, "<extra_id_0>").replace(s, "<extra_id_1>").replace(o, "<extra_id_2>").replace(a, "<extra_id_3>"))
                    # text_one_template = "[<extra_id_0> is <extra_id_1> because [aspect] is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because it is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because of the [aspect]. \
                    # <extra_id_0> is <extra_id_1> taking everything into account.]"
                    query = f"goal: {goal_name_list[0]} \n input: {listtostr(texts[index])} \n sentiment options: [great,ok,bad] \n target options: {text_one_template}"
                else:
                    query = listtostr(texts[index])
            if o_l.split(", ")[0] == 'NULL' and o_l.split(", ")[3] == 'NULL':
                # IA_IO_count += 1
                a = o_l.split(", ")[0]
                c = o_l.split(", ")[1].replace("#", " ").lower()
                s = o_l.split(", ")[2]
                s = sentword2opinion[s]
                o = o_l.split(", ")[3]
                goal = goal_name_list[0]
                if use_sent_flag:
                    # one_template = f"It is implicit aspect, implicit opinion, {c} is {s} taking everything into account."
                    one_template = f"{c} is {s} taking everything into account."
                else:
                    one_template = f"{a}, {c}, {s}, {o}"
                if use_prompt_flag:
                    text_one_template = "".join(one_template.replace(c, "<extra_id_0>").replace(s, "<extra_id_1>").replace(o, "<extra_id_2>").replace(a, "<extra_id_3>"))
                    # text_one_template = "[<extra_id_0> is <extra_id_1> because [aspect] is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because it is [opinion]. \
                    # <extra_id_0> is <extra_id_1> because of the [aspect]. \
                    # <extra_id_0> is <extra_id_1> taking everything into account.]"
                    query = f"goal: {goal_name_list[0]} \n input: {listtostr(texts[index])} \n sentiment options: [great,ok,bad] \n target options: {text_one_template}"
                else:
                    query = listtostr(texts[index])
            # 获得模板和句子 query就是一条句子的里的所有四元组模板
            querys.append(query)
            template.append(one_template)
        template_text.append((' [SSEP] '.join(querys)).split())
        targets.append(' [SSEP] '.join(template))

    return template_text, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=1024, E_I='ALL', use_sent_flag=True, use_prompt_flag=True):
        # './data/rest16/train.txt'
        self.data_path = f'data/acos/{data_dir}/{data_type}.tsv'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.inputs = []
        self.targets = []
        self.E_I = E_I
        self.use_sent_flag = use_sent_flag
        self.use_prompt_flag = use_prompt_flag
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):
        temp_input, temp_labels = read_line_examples_from_acos(self.data_path, self.E_I)
        inputs, targets = get_transformed_io_acos(temp_input, temp_labels)
        # demo
        # print(" demo data : ", inputs[9], targets[9])
        inputs, targets = get_para_acos_targets(inputs ,targets, use_sent_flag=self.use_sent_flag, use_prompt_flag=self.use_prompt_flag)
        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


if __name__ == "__main__":
    from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('model_cache/cache_t5')
    acos_dataset = ABSADataset(tokenizer=tokenizer,
                               data_dir='rest16',
                               data_type='train',
                               E_I='IAIO',
                               use_sent_flag=True,
                               use_prompt_flag=False)
    print(len(acos_dataset))
    data_sample = acos_dataset[128]

    print(data_sample['source_ids'].size())
    print(data_sample['target_ids'].size())
    print('Input is \n', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output is \n', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
   

    # model = T5ForConditionalGeneration.from_pretrained('model_cache/cache_t5')
    # input_ids = tokenizer('translate English to German: That is good.', return_tensors='pt').input_ids
    # labels = tokenizer('Das ist gut.', return_tensors='pt').input_ids
    # decoder_hidden_states = model(input_ids=input_ids, labels=labels,  output_hidden_states=True).decoder_hidden_states 
    # print(decoder_hidden_states[-1].size())
