import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import time

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
    # test_file_name="./dataset/MSRA/test.json"
    # word_vec_file="./dataset/MSRA/MSRA_vectors.txt"

    # train_file_name="./dataset/litbank/train.json"
    # test_file_name="./dataset/litbank/test.json"
    # word_vec_file="./dataset/litbank/litbank_vectors.txt"

    # train_file_name="./dataset/Genia/train.json"
    # test_file_name="./dataset/Genia/test.json"
    # word_vec_file="./dataset/Genia/Genia_vectors.txt"

    # train_file_name="./dataset/Germeval14/train.json"
    # test_file_name="./dataset/Germeval14/test.json"
    # word_vec_file="./dataset/Germeval14/germeval_electra_vectors.txt"

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
    # embedding_dim = 1024
    n_hidden=200
    # num_classes = 3 # ns:0,nt:1,nr:2    MSRA
    # num_classes = 6 # 'PER':0,'FAC':1,'LOC':2,'GPE':3,'ORG':4,'VEH':5  litbank
    # num_classes = 5 # 'protein':0, 'DNA':1, 'cell_line':2, 'RNA':3, 'cell_type':4 Genia
    # num_classes = 12 # {'LOCderiv', 'ORG', 'OTH', 'LOCpart', 'PERderiv', 'PERpart', 'PER', 'OTHderiv', 'OTHpart', 'ORGderiv', 'LOC', 'ORGpart'} Germeval
    # num_classes = 4 # {'LOC', 'PERS', 'ORG', 'MISC'} cleananer
    # num_classes = 5 # {'PER', 'ORG', 'TMP', 'LOC', 'VAL'} pt_selective
    # num_classes = 10 # {'ORG', 'OTR', 'TMP', 'ABS', 'ACO', 'COI', 'LOC', 'OBR', 'VAL', 'PER'} pt_total
    # num_classes=29 # {'NATIONALITY', 'CITY', 'EVENT', 'PERCENT', 'PRODUCT', 'PENALTY', 'NUMBER', 'PROFESSION', 'STATE_OR_PROVINCE', 'AGE', 'LOCATION', 'LANGUAGE', 'IDEOLOGY', 'ORGANIZATION', 'TIME', 'COUNTRY', 'DISTRICT', 'DATE', 'DISEASE', 'PERSON', 'CRIME', 'AWARD', 'MONEY', 'RELIGION', 'FACILITY', 'ORDINAL', 'LAW', 'FAMILY', 'WORK_OF_ART'} nerel
    num_classes=5 # {'ADR':0, 'Disease':1, 'Drug':2, 'Finding':3, 'Symptom':4}
    vocab_size = len(word2id)

    print(len(train_data))
    sentences = []
    index=[]
    labels=[]
    # MSRA_label_dict={'ns':0,'nt':1,'nr':2}
    # litbank_label_dict={'PER':0,'FAC':1,'LOC':2,'GPE':3,'ORG':4,'VEH':5}
    # genia_label_dict={'protein':0, 'DNA':1, 'cell_line':2, 'RNA':3, 'cell_type':4}
    # Germeva14_label_dict={'LOCderiv':0, 'ORG':1, 'OTH':2, 'LOCpart':3, 'PERderiv':4, 'PERpart':5, 'PER':6, 'OTHderiv':7, 'OTHpart':8, 'ORGderiv':9, 'LOC':10, 'ORGpart':11}
    # cleananer_dict={'LOC':0, 'PERS':1, 'ORG':2, 'MISC':3}
    # pt_selective_dict={'PER':0, 'ORG':1, 'TMP':2, 'LOC':3, 'VAL':4} 
    # pt_total_dict={'ORG':0, 'OTR':1, 'TMP':2, 'ABS':3, 'ACO':4, 'COI':5, 'LOC':6, 'OBR':7, 'VAL':8, 'PER':9}
    # nerel_dict={'NATIONALITY':0, 'CITY':1, 'EVENT':2, 'PERCENT':3, 'PRODUCT':4, 'PENALTY':5, 'NUMBER':6, 'PROFESSION':7, 'STATE_OR_PROVINCE':8, 'AGE':9, 'LOCATION':10, 'LANGUAGE':11, 'IDEOLOGY':12, 'ORGANIZATION':13, 'TIME':14, 'COUNTRY':15, 'DISTRICT':16, 'DATE':17, 'DISEASE':18, 'PERSON':19, 'CRIME':20, 'AWARD':21, 'MONEY':22, 'RELIGION':23, 'FACILITY':24, 'ORDINAL':25, 'LAW':26, 'FAMILY':27, 'WORK_OF_ART':28}
    cadec_dict={'ADR':0, 'Disease':1, 'Drug':2, 'Finding':3, 'Symptom':4}
    for i in range(len(train_data)):
        sub_label=[]
        sub_index=[]
        # sentences.append(" ".join(train_data[i]['text'])) # for MSRA
        sentences.append("".join(train_data[i]['text'])) # for litbank
        for j in range(len(train_data[i]['label'])):
            sub_index.append([train_data[i]['label'][j][0],train_data[i]['label'][j][1]])
            sub_label.append(cadec_dict[train_data[i]['label'][j][2]])
        index.append(sub_index)
        labels.append(sub_label)
    
    model = BiLSTM_Attention()
    # model = torch.load('bilstm_attention_MSRA.pth')

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    model.embedding.weight.requires_grad = False
    for i in range(len(word2id)):
        for j in range(300):
            model.embedding.weight[i][j]=float(word2vec[id2word[i]][j])
    
    inputs =[]
    # print(len(word2id))
    # print(word2id)
    for item in sentences:
        sub_lis=[]
        # print(item)
        for n in item.split():
            # print(n)
            if n in word2id.keys():
                sub_lis.append(word2id[n])
            else:
                sub_lis.append(word2id['<unk>'])
        inputs.append(torch.LongTensor([np.asarray(sub_lis)]))
        # break
    # print(inputs[0])
    # print(sentences[0])
    # print("====")
    targets=[]
    for out in labels:
        sub_lis=[]
        for item in out:
            sub_lis.append(torch.LongTensor([item]))
        targets.append(sub_lis)
    # print(targets[0])
    print("start training...")
    start = time.time()
    for epoch in range(600):
        optimizer.zero_grad()
        loss_total=0
        for i in range(len(inputs)):
            for j in range(len(targets[i])):
                output, attention = model(inputs[i],index[i][j][0],index[i][j][1])
                loss = criterion(output,targets[i][j])
                loss_total+=loss.detach().numpy()
                # if (epoch+1)%2==0:
                #     print('Epoch','%04d'%(epoch+1),'loss = ','{:.6f}'.format(loss))
                loss.backward()
        loss_average=loss_total/len(inputs)
        print("epoch:",epoch,"loss:",loss_average)
        optimizer.step()
    end=time.time()
    print("finished time:",end-start,"s")
    # torch.save(model,'bilstm_attention_MSRA_600.pth') # time:18600.865371227264 s(50 epoch) 34785.92405104637 s(100 epoch) 72019.27129721642 s(200 epoch) 177937.0099067688 s (500 epoch) 585776.201359272 s(2000 epoch) time: 307747.4259970188 s (1000 epoch) time: 179908.06247019768 s (600 epoch)
    # torch.save(model,'bilstm_attention_litbank_200.pth')# 11889.38121(200 epoch)
    # torch.save(model,'bilstm_attention_genia_600.pth') # 46777.850148916245 s (200 epoch) time: 117035.64414906502 s(600 epoch)
    # torch.save(model,'bilstm_attention_germeval_electra_600.pth')  # (600 glove epoch) time: 78343.86027908325 s (600 electra epoch) time: 184294.23410606384 s 
    # torch.save(model,'bilstm_attention_cleananer_600.pth') # 600 epoch time:31676.390909910202 s
    # torch.save(model,'bilstm_attention_pt_selective_600.pth') # 600 epoch time: 10231.074682950974 s
    # torch.save(model,'bilstm_attention_pt_total_600.pth') # 600 epoch time: 12690.883929014206 s
    # torch.save(model,'bilstm_attention_nerel_1500.pth') # 1500 epoch time: 289014.5947341919 s
    torch.save(model,'bilstm_attention_cadec_600.pth') # 600 epoch time: 24173.21853017807 s

    # torch.save(model,'bilstm_attention_MSRA_roberta_200.pth') # time:155362.25568795204 s (200 epoch)
    # torch.save(model,'bilstm_attention_MSRA_electra_500.pth') # time: 140716.29056811333 s (200 epoch) 
    # torch.save(model,'bilstm_attention_litbank_1000.pth') # time:49989.53783392906 s (1000 epoch)
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
    test_batch = torch.LongTensor(tests)
    predict,_ = model(test_batch,4,4)
    # predict,_ = model(test_batch,13,16)
    print("predict:")
    print(predict)
    predict = predict.data.max(1, keepdim=True)[1]
    print("predict:",predict[0][0])
