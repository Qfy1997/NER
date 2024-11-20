import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import time
import csv

def read_file(file_name):
    data=[]
    with open(file_name,"r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def trans_word_to_id_and_vec(vectors_file):
    word2id={}
    word2vec={}
    id2word={}
    with open(vectors_file,"r") as f:
        data = f.readlines()
    for i in range(len(data)):
        word2id[data[i].strip().split(' ')[0]]=i
        word2vec[data[i].strip().split(' ')[0]]=data[i].strip().split(' ')[1:]
        id2word[i]=data[i].strip().split(' ')[0]
    return word2id,word2vec,id2word

class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,n_hidden,bidirectional=True)
        self.out = nn.Linear(n_hidden*2,num_classes)
    
    def attention_net(self,lstm_output, final_state):
        hidden = final_state.view(-1,n_hidden*2,1)
        attn_weights = torch.bmm(lstm_output,hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights,1)
        context = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()
    
    def forward(self, X, left_span, right_span):
        input= self.embedding(X)
        input = input[:,left_span:right_span+1,:]
        input = input.permute(1,0,2)
        hidden_state = torch.zeros(1*2,len(X),n_hidden)
        cell_state = torch.zeros(1*2,len(X),n_hidden)
        output,(final_hidden_state,final_cell_state) = self.lstm(input,(hidden_state,cell_state))
        output = output.permute(1,0,2)
        attn_output, attention = self.attention_net(output,final_hidden_state)
        return self.out(attn_output),attention

if __name__=='__main__':
    # train_file_name="./dataset/MSRA/train.json"
    # test_file_name="./dataset/MSRA/test1.json"
    # word_vec_file="./dataset/MSRA/MSRA_vectors.txt"
    # train_file_name="./dataset/litbank/train.json"
    # test_file_name="./dataset/litbank/test.json"
    # word_vec_file="./dataset/litbank/litbank_vectors.txt"
    # train_file_name="./dataset/Genia/train.json"
    # test_file_name="./dataset/Genia/test.json"
    # word_vec_file="./dataset/Genia/Genia_vectors.txt"

    # train_file_name="./dataset/Germeval14/train.json"
    # test_file_name="./dataset/Germeval14/test.json"
    # word_vec_file="./dataset/Germeval14/Germeval_vectors.txt"

    # train_file_name="./dataset/cleananer/train.json"
    # test_file_name="./dataset/cleananer/test.json"
    # word_vec_file="./dataset/cleananer/cleananer_vectors.txt"

    # train_file_name="./dataset/ptner/train_selective.json"
    # test_file_name="./dataset/ptner/test_selective.json"
    # word_vec_file="./dataset/ptner/selective_vectors.txt"

    # train_file_name="./dataset/ptner/train_total.json"
    # test_file_name="./dataset/ptner/test_total.json"
    # word_vec_file="./dataset/ptner/total_vectors.txt"

    # train_file_name="./dataset/nerel/train_json_data.json"
    # test_file_name="./dataset/nerel/test_json_data.json"
    # word_vec_file="./dataset/nerel/nerel_vectors.txt"

    train_file_name="./dataset/CADEC/train_data.json"
    test_file_name="./dataset/CADEC/test_data.json"
    word_vec_file="./dataset/CADEC/cadec_vectors.txt"


    train_data = read_file(train_file_name)
    test_data = read_file(test_file_name)
    word2id,word2vec,id2word = trans_word_to_id_and_vec(word_vec_file)
    embedding_dim=300
    n_hidden=200
    # num_classes = 3 # ns:0,nt:1,nr:2
    # num_classes = 6 # 'PER':0,'FAC':1,'LOC':2,'GPE':3,'ORG':4,'VEH':5
    # num_classes = 5 # 'protein':0, 'DNA':1, 'cell_line':2, 'RNA':3, 'cell_type':4
    # num_classes = 12 # {'LOCderiv', 'ORG', 'OTH', 'LOCpart', 'PERderiv', 'PERpart', 'PER', 'OTHderiv', 'OTHpart', 'ORGderiv', 'LOC', 'ORGpart'}
    # num_classes = 4 # {'LOC', 'PERS', 'ORG', 'MISC'} cleananer
    # num_classes = 5 # pt_selective_dict={'PER':0, 'ORG':1, 'TMP':2, 'LOC':3, 'VAL':4}  # pt selective
    # num_classes = 10 # {'ORG', 'OTR', 'TMP', 'ABS', 'ACO', 'COI', 'LOC', 'OBR', 'VAL', 'PER'} pt_total
    # num_classes=29 # {'NATIONALITY', 'CITY', 'EVENT', 'PERCENT', 'PRODUCT', 'PENALTY', 'NUMBER', 'PROFESSION', 'STATE_OR_PROVINCE', 'AGE', 'LOCATION', 'LANGUAGE', 'IDEOLOGY', 'ORGANIZATION', 'TIME', 'COUNTRY', 'DISTRICT', 'DATE', 'DISEASE', 'PERSON', 'CRIME', 'AWARD', 'MONEY', 'RELIGION', 'FACILITY', 'ORDINAL', 'LAW', 'FAMILY', 'WORK_OF_ART'} nerel
    num_classes = 5 # {'ADR':0, 'Disease':1, 'Drug':2, 'Finding':3, 'Symptom':4}

    vocab_size = len(word2id)

    print(len(test_data))
    sentences = []
    index=[]
    labels=[]
    all_label=[]
    # MSRA_label_dict={'ns':0,'nt':1,'nr':2}
    # MSRA_lebal_dict={0:'ns',1:'nt',2:'nr'}
    # litbank_label_dict={'PER':0,'FAC':1,'LOC':2,'GPE':3,'ORG':4,'VEH':5}
    # litbank_lebal_dict={0:'PER',1:'FAC',2:'LOC',3:'GPE',4:'ORG',5:'VEH'}
    # genia_label_dict={'protein':0, 'DNA':1, 'cell_line':2, 'RNA':3, 'cell_type':4}
    # genia_lebal_dict={0:'protein',1:'DNA',2:'cell_line',3:'RNA',4:'cell_type'}
    # Germeva14_label_dict={'LOCderiv':0, 'ORG':1, 'OTH':2, 'LOCpart':3, 'PERderiv':4, 'PERpart':5, 'PER':6, 'OTHderiv':7, 'OTHpart':8, 'ORGderiv':9, 'LOC':10, 'ORGpart':11}
    # Germeval14_lebal_dict={0:'LOCderiv',1:'ORG',2:'OTH',3:'LOCpart',4:'PERderiv',5:'PERpart',6:'PER',7:'OTHderiv',8:'OTHpart',9:'ORGderiv',10:'LOC',11:'ORGpart'}
    # cleananer_dict={'LOC':0, 'PERS':1, 'ORG':2, 'MISC':3}
    # cleananer_lebal_dict={0:'LOC',1:'PERS',2:'ORG',3:'MISC'}
    # pt_selective_dict={'PER':0, 'ORG':1, 'TMP':2, 'LOC':3, 'VAL':4} 
    # pt_selective_lebal_dict={0:'PER',1:'ORG',2:'TMP',3:'LOC',4:'VAL'}
    # pt_total_dict={'ORG':0, 'OTR':1, 'TMP':2, 'ABS':3, 'ACO':4, 'COI':5, 'LOC':6, 'OBR':7, 'VAL':8, 'PER':9}
    # nerel_dict={'NATIONALITY':0, 'CITY':1, 'EVENT':2, 'PERCENT':3, 'PRODUCT':4, 'PENALTY':5, 'NUMBER':6, 'PROFESSION':7, 'STATE_OR_PROVINCE':8, 'AGE':9, 'LOCATION':10, 'LANGUAGE':11, 'IDEOLOGY':12, 'ORGANIZATION':13, 'TIME':14, 'COUNTRY':15, 'DISTRICT':16, 'DATE':17, 'DISEASE':18, 'PERSON':19, 'CRIME':20, 'AWARD':21, 'MONEY':22, 'RELIGION':23, 'FACILITY':24, 'ORDINAL':25, 'LAW':26, 'FAMILY':27, 'WORK_OF_ART':28}
    # nerel_lebal_dict={0:'NATIONALITY',1:'CITY',2:'EVENT',3:'PERCENT',4:'PRODUCT',5:'PERNALTY',6:'NUMBER',7:'PROFESSION',8:'STATE_OR_PROVINCE',9:'AGE',10:'LOCATION',11:'LANGUAGE',12:'IDEOLOGY',13:'ORGANIZATION',14:'TIME',15:'COUNTRY',16:'DISRICT',17:'DATE',18:'DISEASE',19:'PERSON',20:'CRIME',21:'AWARD',22:'MONEY',23:'RELIGION',24:'FACILITY',25:'ORDINAL',26:'LAW',27:'FAMILY',28:'WORK_OF_ART'}
    cadec_dict={'ADR':0, 'Disease':1, 'Drug':2, 'Finding':3, 'Symptom':4}
    # cadec_lebal_dict={0:'ADR',1:'Disease',2:'Drug',3:'Finding',4:'Symptom'}

    for i in range(len(test_data)):
        sub_label=[]
        sub_all_label=[]
        sub_index=[]
        # sentences.append(" ".join(test_data[i]['text'])) # for MSRA
        sentences.append("".join(test_data[i]['text'])) # for litbank
        for j in range(len(test_data[i]['label'])):
            sub_index.append([test_data[i]['label'][j][0],test_data[i]['label'][j][1]])
            sub_label.append(cadec_dict[test_data[i]['label'][j][2]])
            sub_all_label.append(test_data[i]['label'][j])
        index.append(sub_index)
        labels.append(sub_label)
        all_label.append(sub_all_label)
    
    # model = BiLSTM_Attention()
    # model = torch.load('bilstm_attention_MSRA_600.pth')
    # model = torch.load('bilstm_attention_litbank_600.pth')
    # model = torch.load('bilstm_attention_genia_600.pth')
    # model = torch.load('bilstm_attention_germeval_600.pth')
    # model = torch.load('bilstm_attention_cleananer_600.pth')
    # model = torch.load('bilstm_attention_pt_selective_600.pth')
    # model = torch.load('bilstm_attention_pt_total_600.pth')
    # model = torch.load('bilstm_attention_nerel_1500.pth')
    model = torch.load('bilstm_attention_cadec_600.pth')


    # 定义一个测试文本 test_text
    # test_text = '中 共 中 央 致 中 国 致 公 党 十 一 大 的 贺 词'
    # test_text = 'The old lady pulled her spectacles down and looked over them about the room ; then she put them up and looked out under them .'
    # test_text = 'Thyroid hormone receptors form distinct nuclear protein- dependent and independent complexes with a thyroid hormone response element .'
    # test_text = 'T1951 bis 1953 wurde der nördliche Teil als Jugendburg des Kolpingwerkes gebaut .'
    # test_text='الصالحية المفرق - غيث الطراونة - أمر جلالة الملك عبدالله الثاني أمس بتنفيذ حزمة من المشاريع التعليمية والصحية والتنموية وأخرى مرتبطة بالأندية الشبابية و 27 وحدة سكنية في قضاء الصالحية ونايفة في البادية الشرقية خلال ستة اشهر بتمويل من الديوان الملكي الهاشمي .'
    # test_text='Foi o mais influente de os pensadores de os EUA , criador de o pragmatismo .'
    # test_text='Сводный экономический департамент входит в блок денежно-кредитной политики .'
    test_text='little improvement with the pain .'

    # 并转换为张量
    sub_lis=[]
    for n in test_text.split():
        if n in word2id.keys():
            sub_lis.append(word2id[n])
        else:
            sub_lis.append(word2id['<unk>'])
    
    tests = [np.asarray(sub_lis)]
    # print(sub_lis)
    # tests.append([torch.LongTensor(np.asarray(sub_lis))])
    test_batch = torch.LongTensor(tests)
    predict,_ = model(test_batch,3,4)
    print("predict:")
    print(predict)
    predict = predict.data.max(1, keepdim=True)[1]
    print("predict:",predict[0][0].numpy())
    print("real data:")
    print(sentences[0])
    print(index[0])
    print(labels[0])
    all_true=0
    cor=0
    for i in range(len(sentences)):
        test_text=sentences[i]
        sub_lis = []
        for n in test_text.split():
            if n in word2id.keys():
                sub_lis.append(word2id[n])
            else:
                sub_lis.append(word2id['<unk>'])
        tests = [np.asarray(sub_lis)]
        test_batch = torch.LongTensor(tests)
        for j in range(len(labels[i])):
            all_true+=1
            left=index[i][j][0]
            right = index[i][j][1]
            # print("left:",left)
            # print("right:",right)
            # print("j:",j)
            # print("i:",i)
            predict,_ = model(test_batch,left,right)
            predict = predict.data.max(1, keepdim=True)[1][0][0].numpy()
            if predict==labels[i][j]:
                cor+=1
    print("cor:",cor)
    print("all_true:",all_true)
    precision=cor/all_true
    print("precision:",precision)
    recall = cor/all_true
    print("recall:",recall)
    f1 = 2*precision*recall/(precision+recall)
    print("f1:",f1)

    # data=[]
    # for i in range(len(sentences)):
    #     test_text=sentences[i]
    #     sub_lis = []
    #     for n in test_text.split():
    #         if n in word2id.keys():
    #             sub_lis.append(word2id[n])
    #         else:
    #             sub_lis.append(word2id['<unk>'])
    #     tests = [np.asarray(sub_lis)]
    #     test_batch = torch.LongTensor(tests)
    #     for j in range(len(labels[i])):
    #         one_data=[]
    #         one_data.append(all_label[i][j][3])
    #         left=index[i][j][0]
    #         right = index[i][j][1]
    #         predict,_ = model(test_batch,left,right)
    #         one_data.append(list(predict.detach().numpy()[0]))
    #         one_data.append(cadec_lebal_dict[int(predict.data.max(1, keepdim=True)[1][0][0].numpy())])
    #         one_data.append(all_label[i][j][2])
    #         data.append(one_data)
    #         if one_data[2]==one_data[3]:
    #             one_data.append(1)
    #         else:
    #             one_data.append(0)
            
    #         # break
    #     # break
    # print(len(data))
    # print(data[0])
    # with open('cadec_results.csv', 'w', newline='') as file:
    #     fields = ['entity content', 'predict numbers', 'predict label','real label','predict true or not']
    #     writer = csv.DictWriter(file, fieldnames=fields)
    #     writer.writeheader()
    #     for item in data:
    #         writer.writerow({'entity content': item[0], 'predict numbers': item[1], 'predict label': item[2],'real label':item[3],'predict true or not':item[4]})
    
    
    # #含discontinuous的f1score计算方式
    # discontinuous_sample=[]
    # for i in range(len(test_data)):
    #     if 'discontinuous' in test_data[i]:
    #         discontinuous_sample.append(test_data[i])
    # # print(len(discontinuous_sample))
    # i=0
    # dis_sample=0
    # dis_sample_true=0
    # for sentence in discontinuous_sample:
    #     sub_lis=[]
    #     for n in sentence['text'].split():
    #         if n in word2id.keys():
    #             sub_lis.append(word2id[n])
    #         else:
    #             sub_lis.append(word2id['<unk>'])
    #     tests = [np.asarray(sub_lis)]
    #     test_batch = torch.LongTensor(tests)
    #     # print("第",i,"个:sentence")
    #     # print("i:",i)
    #     for item in sentence['discontinuous']:
    #         # print("当前item:",item)
    #         flag=0
    #         for ite in item:
    #             # print(ite)
    #             predict,_ = model(test_batch,ite[0],ite[1])
    #             predict = predict.data.max(1, keepdim=True)[1][0][0].numpy()
    #             # print("predict:",predict)
    #             real_label=cadec_dict[ite[2]]
    #             # print("real label:",real_label)
    #             if predict!=real_label:
    #                 flag=1
    #         dis_sample+=1
    #         if flag!=1:
    #             dis_sample_true+=1
    #     i+=1
    # print(dis_sample)
    # print(dis_sample_true)
    # # precision=(cor+dis_sample_true)/(all_true+dis_sample_true)
    # precision=cor/all_true
    # print("precision:",precision)
    # recall=(cor)/(all_true+dis_sample-dis_sample_true)
    # print("recall:",recall)
    # f1_score=2*precision*recall/(precision+recall)
    # print("f1 score:",f1_score)

    # 30 epoch MSRA pre recall f1 0.8686 | 50 epoch MSRA pre recall f1 90.55 | 100 epoch MSRA pre recall f1 94.29 | 200 epoch MSRA pre recall f1 96.00 | 200 epoch roberta_static MSRA pre recall f1 95.97 | 500 epoch MSRA pre recall f1 0.9686 | 2000 epoch MSRA pre recall f1 0.9676 | 200 epoch electra pre recall f1 0.9572 ｜ 1000 epoch MSRA pre recall f1 0.9684 | 600 epoch MSRA pre 0.9687 recall 1 f1 98.41
    # 30 epoch litbank pre recall f1 0.76998 | 200 epoch pre recall f1 0.8725 | 600 epoch pre 0.8793 recall 1 f1 0.9358 | 800 epoch pre recall f1 0.8680 | 1000 epoch pre recall f1 0.8733
    # 200 epoch genia pre recall f1 0.9072 | 600 epoch pre 0.9176 recall 1 f1 0.9570
    # 600 epoch germeval pre 0.6947 recall 1 f1 0.8199
    # 600 epoch cleananer pre 0.7273 recall 1 f1 0.8421
    # 600 epoch pt_selective pre 0.6498 recall 1 f1 0.7877
    # 600 epoch pt_total pre recall f1 0.5179
    # 1500 epoch nerel pre 0.6875 recall 0.6864 f1 0.6869 acc 0.6866
    # 600 epoch cadec pre 0.8905 recall 0.8809 f1 0.88.57 acc 0.8865