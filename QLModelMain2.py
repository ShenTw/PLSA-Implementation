import QLModle2
table = QLModle2.QLM()


# include BGLM
with open('BGLM.txt','r') as BGLM:
    for line0 in BGLM :
        temp = []
        line0 = line0.strip('\n')
        temp = temp + line0.split()
        table.corpus_dict[temp[0]] = temp[1]
    #測試BGLM
    #print(table.corpus_dict)
#open doc_list
with open('doc_list.txt','r' ) as L:
    
    for line0 in L:
        line0 = line0.strip('\n')
        path = "C:/Users/shen/.spyder-py3/Document3/"
        file = path+line0
        with open(file,'r') as f:
            
            listOfWords=[]
            temp= []
            #每篇doc的初始化材料
            for line in f.readlines() :
                #    data=f.readline()
                line=line.strip('\n')
                line=line.strip('-1')
                temp = temp + line.split()
                #temp為該doc的字典(只有字)
                #print("doc : \n" ,line0, "temp : ", temp)
            for i in range(5,len(temp),+1):
                    #扣除前三行不需要的資訊(經切割後為五項)
                listOfWords.append(temp[i])
                 
                #listOfwords為該doc的字典 (字:次數)
                
                #print("Doc: ", line0, "words",listOfWords)
    
            table.add_document(line0, listOfWords)
            #存成一個documents 文件集 {doc:{字:字數}}
            #length = len(result)
            #總文件數量
  #          numpy.array([for word in result])

#open collection and prepare to train
with open('doc_list.txt','r' ) as f:
    print("collecting...")
    Corpus = QLModle2.Corpus()
    count = 1
        #每篇doc的初始化材料
    for line in f.readlines() :
        #print("training corpus "+str(count))
        temp= []
        #    data=f.readline()
        line=line.strip('\n')
        temp = temp + line.split() # temp為切好的字的list
        doc = QLModle2.Document(temp)
        count = count +1 
        Corpus.add_document(doc)
        #print(self.trainedCorpus)
    print("collecting corpus over")
    #結束corpus建立，建立字典
    Corpus.build_vocabulary()            

table.initialTrain(Corpus)
table.train() # train for p_w_z

            
#open and save query
with open('query_list3.txt','r') as Q:

    for line0 in Q:
        line0 = line0.strip('\n')
        path = "C:/Users/shen/.spyder-py3/Query3/"
        file = path+line0
        with open(file,'r') as f:
            
            listOfWords = []
            temp = []
            line = f.readline()
            line = line.strip('-1')
            temp = temp + line.split()
                #不需要最後一個-1 所以只做到len -1
            for i in range(0,len(temp)-1,+1):
                listOfWords.append(temp[i])
 



"""         
        #table.likelihood(listOfWords,line0)
with open('HW3_M10615103.txt','w') as W:
    W.write("Query,RetrievedDocuments")
    for query in table.sims:
        W.write("\n")
        W.write(query)
        W.write(",")
        #print(query)
        for doc,value in table.sims[query]:
            W.write(doc)
            W.write(" ")
           # print ("query :"+query +" doc : "+doc + " score: "+ str(value)+"\n")
"""    
        #print("Query: ", line0,"is ",listOfWords) 

#print("prepare to finalize!")     


"""
count =0
for doc in table.sims['20002.query']:
    if count<13:
        print("result is : ", doc)
        count=count+1
    else:
        break
"""
#print("result is : \n",table.sims['20002.query'])          
#print ("corpus is : ",table.corpus_dict)
            # total times
#print ("\ndocument is : " ,table.documents['VOM19980220.0700.0166']) 
        # documents : tf 
