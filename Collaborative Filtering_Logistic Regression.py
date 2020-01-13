#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering

# 1. Connecting DB(oracle) to Python

# 2. Data Preprocessing

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# # Collaborative Filtering Not Using Package

# ## 은행별 분류전 과정명, 수료구분 preprocessing

# In[3]:


# sql query에서 수료구분 부분을 수정한 데이터
df_TrainMerged_new = pd.read_csv('수강정보_수정본.csv', encoding = 'ms949', low_memory=False)


# In[5]:


#Generally Mendatory Courses Exclusion
wordsToExclude = ['펀드투자상담사 등록교육', '펀드투자상담사등록', '채무증권 투자권유 실무', 
                  '사원기관 HRD과정', 'ATD 사원기관 연수담당 책임자 과정', '펀드투자상담사(부동산,파생상품)',
                 '펀드투자상담사(부동산)', '펀드투자상담사(파생상품)', '펀드투자상담사 자격시험대비',
                 '펀드투자권유자문인력 사전교육(주말)(펀드투자자보호 교육)']


# In[6]:


addWordsToExc = ["Banking Business English",
"Big Data Analytics : Machine Learning",
"Impact of Information Technology on Finance",
"KBI지식콘텐츠 서비스(테스트용)",
"SC TEST",
"Understanding FX and Money Markets",
"Understanding of Derivatives - Forward/Futures",
"Understanding of Investment Banking - DCM",
"격파! 파생상품투자권유자문인력",
"대출상담사 등록교육",
"대출상담사 등록교육(신)",
"대출상담사 정기교육",
"대출상담사 정기교육(신)",
"매경 TEST",
"베트남 금융연수원 연수",
"베트남 은행구조조정 및 부실채권정리 과정",
"은퇴설계전문가-Core 등록교육",
"채무증권 투자권유 실무",
"채무증권 투자권유 실무 보수교육",
"FP보수",
"PRMTP(CIFO, Chicago)",
"TEST",
"간접투자상품판매",
"간접투자상품판매",
"간접투자상품판매 보수교육",
"간접투자상품판매 보수교육(예산용)",
"간접투자상품판매(단가차액에따른금액)",
"간접투자상품판매실무(단가차액에따른금액)",
"광주지역 간접투자상품판매실무",
"국제금융역 보수",
"기술신보테스트과정",
"기타과정",
"대구지역 간접투자상품판매실무",
"대출심사역보수",
"부동산펀드투자상담사",
"부산지역 간접투자상품판매실무",
"수탁",
"신용분석사 보수",
"여신심사역 보수",
"외국환업무t",
"자금운용역보수",
"자산관리사(FP) 보수",
"정신교육(남자)",
"정신교육(여자)",
"증권펀드투자상담사",
"증권펀드투자상담사 보수교육",
"증권펀드투자상담사 보수교육",
"집합 모의",
"투자상담관리인력 등록교육",
"투자상담관리인력 보수교육",
"파생상품투자권유자문인력 전문성교육",
"파생상품투자상담사 보수교육",
"파생상품펀드투자상담사",
"펀드투자상담사 등록교육",
"펀드투자상담사 보수교육(하)",
"평가BMT",
"모바일평가 검수",
"JB 리더스클럽",
"펀드투자권유자문인력 전문성교육",
"펀드투자권유자문인력 투자자보호",
"창의적인 금융인을 꿈꾸는 청소년을 위한 금융교육",
"펀드투자상담사(부동산,파생상품)",
"FinTech Business Models"]


# In[7]:


wordsExTotalList = wordsToExclude + addWordsToExc
wordsExTotalList


# In[8]:


wordList_기관 = ['IBK', '우리은행', '농협', '농협은행', '부산은행', 
               'KEB하나은행', '하나은행', 'KEB/HANA', 'HANA/KEB', 'KEB', 'Hanabank', '외환은행', 'NH','수협',
              '대구은행', '광주은행', '산업은행', '경남은행', 'KDB', '스탠다드차타드', '전북', '씨티', 'Citi', '제주',
              '수출입은행', '신용보증기금', '기술보증기금', '제일은행', '상호저축은행', '신한은행',
              '국민은행', 'KB', 'L1', 'L2', 'L3', 'BDC', 'BNK', 'CSR', 'DGB', 'KDB', 'KJB', 'PFC',
              'SC', 'Teller', 'Woori', '국민주택기금', '금융감독원', '기술신보', '기업은행', '대한투자신탁',
              '부은 사관학교', '부은금융사관학교', '삼성생명', '새마을금고', '서울신탁은행', '서울은행', '성업공사',
              '신보', '예금보험공사', '우리 FPM', '우리론컨설턴트', '은행감독원', '재경부', '전문건설공제조합', '전북, 광주은행',
              '전북은행', '부은 금융사관학교', '제주은행', '조흥은행', '케이뱅크', '하나금융그룹', '한국선물거래',
              '한국신용정보원', '한미은행', '한빛은행', '현대투자신탁', '회원수협', '중소기업진흥공단']


# In[12]:


df_TrainMerged_new_edited = df_TrainMerged_new[['D_OPN_YY','I_TRAN_TP_NAME', 'I_EXE_ST', 'O_REG', 'N_COUR', 'I_TRAN_NAME', 'N_CSTM']]


# In[8]:


df_TrainMerged_new.I_TRAN_NAME.unique()


# In[13]:


df_TrainMerged_new.I_TRAN_TP_NAME.value_counts(normalize=True)


# ### Data Preprocessing

# In[19]:


def datapreprocessing(df, year1, wordsToExList, instituteList):
    df_post2010 = df[df['D_OPN_YY']>=year1]
    df_post2010 = df_post2010[df_post2010['D_OPN_YY']<2019]
    df_post2010 = df_post2010[df_post2010['I_TRAN_TP_NAME']=='정규연수']
    df_post2010.dropna(subset=['N_COUR'], inplace = True)
    #미수료, 퇴교된 과정 제외
    df_post2010 = df_post2010[df_post2010['I_EXE_ST']!='미수료']
    df_post2010 = df_post2010[df_post2010['I_EXE_ST']!='퇴교']
    #집합, 통신, 사이버만
    df_post2010 = df_post2010[df_post2010['I_TRAN_NAME']!='일반(기타)연수']
    df_post2010 = df_post2010[df_post2010['I_TRAN_NAME']!='자격실무교육']
    df_post2010 = df_post2010[df_post2010['I_TRAN_NAME']!='해외연수']
    #보수교육 제외
    df_post2010 = df_post2010[~df_post2010['N_COUR'].str.contains('보수')]
    
    df_post2010_NF = df_post2010[['O_REG', 'D_OPN_YY', 'N_COUR', 'I_TRAN_NAME', 'N_CSTM']]
    
    
    for word in wordsToExList:
        df_post2010_NF = df_post2010_NF[df_post2010_NF['N_COUR']!=word]
        df_post2010_NF = df_post2010_NF[~df_post2010_NF['N_COUR'].str.contains(word)]
    
    for word in instituteList:
        df_post2010_NF = df_post2010_NF[~df_post2010_NF['N_COUR'].str.contains(word)]
        
    #수료했음에도 재수강이력 제외
#     df_post2010_NF = df_post2010_NF.drop_duplicates(subset=['O_REG', 'N_COUR'], keep = 'first')
    # 5개 이상의 수업을 들은 수강생들만 남김.
#     df_post2010_NF = df_post2010_NF.groupby('O_REG').filter(lambda x: len(x) > 4)
    #9999로 시작하는 test id 제외
    df_post2010_NF = df_post2010_NF[~df_post2010_NF['O_REG'].str.contains("99999")]
    
    
    return df_post2010_NF


# In[33]:


def datapreprocessing집합(df, year1, wordsToExList, instituteList):
    df_post2010 = df[df['D_OPN_YY']>=year1]
    df_post2010 = df_post2010[df_post2010['D_OPN_YY']<2019]
    df_post2010 = df_post2010[df_post2010['I_TRAN_TP_NAME']=='정규연수']
    df_post2010.dropna(subset=['N_COUR'], inplace = True)
    #미수료, 퇴교된 과정 제외
    df_post2010 = df_post2010[df_post2010['I_EXE_ST']!='미수료']
    df_post2010 = df_post2010[df_post2010['I_EXE_ST']!='퇴교']
    #집합 만
    df_post2010 = df_post2010[df_post2010['I_TRAN_NAME']=='집합연수']
#     df_post2010 = df_post2010[df_post2010['I_TRAN_NAME']!='자격실무교육']
#     df_post2010 = df_post2010[df_post2010['I_TRAN_NAME']!='해외연수']
    #보수교육 제외
    df_post2010 = df_post2010[~df_post2010['N_COUR'].str.contains('보수')]
    
    df_post2010_NF = df_post2010[['O_REG', 'D_OPN_YY', 'N_COUR', 'I_TRAN_NAME', 'N_CSTM']]
    
    
    for word in wordsToExList:
        df_post2010_NF = df_post2010_NF[df_post2010_NF['N_COUR']!=word]
        df_post2010_NF = df_post2010_NF[~df_post2010_NF['N_COUR'].str.contains(word)]
    
    for word in instituteList:
        df_post2010_NF = df_post2010_NF[~df_post2010_NF['N_COUR'].str.contains(word)]
        
    #수료했음에도 재수강이력 제외
    df_post2010_NF = df_post2010_NF.drop_duplicates(subset=['O_REG', 'N_COUR'], keep = 'first')
    # 5개 이상의 수업을 들은 수강생들만 남김.
    df_post2010_NF = df_post2010_NF.groupby('O_REG').filter(lambda x: len(x) > 4)
    #9999로 시작하는 test id 제외
    df_post2010_NF = df_post2010_NF[~df_post2010_NF['O_REG'].str.contains("99999")]
    
    
    return df_post2010_NF


# In[20]:


dfdf = datapreprocessing(df_TrainMerged_new_edited, 2010, wordsExTotalList, wordList_기관)


# In[46]:


dfdf_집합 = datapreprocessing집합(df_TrainMerged_new, 2010, wordsExTotalList, wordList_기관)


# In[232]:


dfdf_2012 = datapreprocessing(df_TrainMerged_new, 2012, wordsExTotalList, wordList_기관)


# In[15]:


exceptionList = ['신용분석기초', '신용분석기초(야)', '회계원리와 재무제표 작성', '회계원리와 재무제표 분석', '회계원리와 재무제표 작성(부산)']


# In[16]:


dfCourseNameChanges = pd.read_excel('과정변경이력.xlsx', encoding = 'ms949')


# In[17]:


def convertCourseNames(df, df_namechange):
    
    #exception 신용분석기초 & 집합연수
    for i in range(len(df)):
        if df.iloc[i].N_COUR == '신용분석기초' and df.iloc[i].I_TRAN_NAME == '집합연수':
            df.iloc[i].N_COUR = '회계원리와 재무제표 작성'
            
#     courseNameChanges = df_namechange[df_namechange.N_COUR != df_namechange.UNIQUE_NAME]
    courseNameChanges_2 = df_namechange.set_index('N_COUR')
    courseNameChanges_2 = courseNameChanges_2.loc[~courseNameChanges_2.index.duplicated(keep='first')]

    
    
    oldCourseName = courseNameChanges_2.index
    oldNameList = oldCourseName.tolist()
    
    uniqueNameList = courseNameChanges_2.UNIQUE_NAME.tolist()
    
    namesDict = {}
    for i in range(len(oldNameList)):
        namesDict[oldNameList[i]] = uniqueNameList[i]
        
    allCourseNameSet = set(df['N_COUR'])
    necessaryCourseNameSet = set(oldNameList)
    
    exceptionCourseNameSet = allCourseNameSet - necessaryCourseNameSet
    exceptionCourseNameList = list(exceptionCourseNameSet)
    
    for course in oldNameList:
        df['N_COUR'] = df['N_COUR'].replace(course, namesDict[course])
    
    for course in exceptionCourseNameList:
        df = df[df['N_COUR']!=course]
    
    return df


# In[18]:


df_namechanges = convertCourseNames(dfdf, dfCourseNameChanges)


# In[34]:


df_namechanges.shape


# In[47]:


df_집합 = convertCourseNames(dfdf_집합, dfCourseNameChanges)


# In[233]:


df_2012 = convertCourseNames(dfdf_2012, dfCourseNameChanges)


# In[36]:


def preprocessing2(df):
    #(1) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(1\)","")
    #(비) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(비\)","")
    #(부산), (광주) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(부산\)","")
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(광주\)","")
    #(금토) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(금토\) ","")
    #(B) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(B\)","")
    #(구), (통), (사) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(구\)","")
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(통\)","")
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(사\)","")
    #(부제:..) 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(부제.*\)","")
    #주말 제거
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(주말\) ","")
    df['N_COUR'] = df['N_COUR'].str.replace(r"주말 ","")
    df['N_COUR'] = df['N_COUR'].str.replace(r"\(주말\)","")
    
    df['N_COUR'] = df['N_COUR'].str.rstrip()
    df['N_COUR'] = df['N_COUR'].str.lstrip()
    
    return df


# In[37]:


df_processing2 = preprocessing2(df_namechanges)


# In[38]:


df_processing2.shape


# In[48]:


df_집합 = preprocessing2(df_집합)


# In[50]:


df_집합.I_TRAN_NAME.unique()


# In[234]:


df_2012 = preprocessing2(df_2012)


# In[39]:


def preprocessing3(df, wordsToExList):
    
    for word in wordsToExList:
        df = df[df['N_COUR']!=word]
        df = df[~df['N_COUR'].str.contains(word)]


    #수료했음에도 재수강이력 제외
    df = df.drop_duplicates(subset=['O_REG', 'N_COUR'], keep = 'first')
    # 5개 이상의 수업을 들은 수강생들만 남김.
#     df = df.groupby('O_REG').filter(lambda x: len(x) > 4)
    
    return df


# In[40]:


df_processing3 = preprocessing3(df_processing2, wordsExTotalList)


# In[41]:


df_processing3.shape


# In[51]:


df_집합 = preprocessing3(df_집합, wordsExTotalList)


# In[235]:


df_2012 = preprocessing3(df_2012, wordsExTotalList)


# df_processing3.to_excel('datapreprocessed.xlsx')

# df_processing3.shape

# In[201]:


df_processing3 = pd.read_excel('datapreprocessed.xlsx', encoding = "ms949", index_col=0)


# In[202]:


df_processing3.shape


# #개인정보 데이터
# df_PI = pd.read_csv('Personal_InfoData.csv', encoding = "ms949", low_memory=False)

# df_PI_NEW = df_PI[['N_CSTM', 'I_DMND_NAME']]

# df_PI_NEW = df_PI_NEW.drop_duplicates()

# df_PI_NEW.shape

# df_PI_NEW.to_excel('InstDist.xlsx')

# In[179]:


df_Inst = pd.read_excel('InstDist.xlsx', encoding = "ms949")


# In[180]:


df_Inst.shape


# In[181]:


비사원기관List = np.unique(df_Inst[df_Inst['I_DMND_NAME']=='비사원기관'].N_CSTM).tolist()


# In[183]:


사원기관List = np.unique(df_Inst[df_Inst['I_DMND_NAME']=='사원기관'].N_CSTM).tolist()


# In[203]:


df_processing3_사원 = df_processing3[df_processing3['N_CSTM'].isin(사원기관List)]


# In[212]:


df_processing3_사원_CF = df_processing3_사원[['O_REG', 'N_COUR']]


# In[213]:


df_processing3_사원_CF_over5 = df_processing3_사원_CF.groupby('O_REG').filter(lambda x: len(x) > 4)


# In[214]:


df_processing3_사원_CF_over5.shape


# In[ ]:





# In[ ]:





# ### Data Segmentation

# ## 기관별

# In[91]:


df.N_CSTM.value_counts()


# In[92]:


def segmentByInst(df, inst):
    df = df[df['N_CSTM']==inst]
    return df


# In[93]:


df_부산 = segmentByInst(df, '부산은행')
df_국민 = segmentByInst(df, '국민은행')
df_KEB = segmentByInst(df, 'KEB하나은행')
df_신한 = segmentByInst(df, '신한은행')
df_개인 = segmentByInst(df, '개인고객')


# ## 사전사후과정 정리

# In[21]:


#사전사후과정
coursePreReq = pd.read_excel('사전_사후_과정.xlsx', encoding = "ms949")


# In[22]:


coursePreReq = coursePreReq[['후과정', '전과정']]


# In[23]:


coursePreReq_unique = coursePreReq.drop_duplicates(subset=['후과정', '전과정'], keep = 'first')


# In[24]:


coursePreReq_unique.head()


# In[25]:


coursePreReq_Ser = coursePreReq_unique.groupby('후과정')['전과정'].apply(lambda x: x.tolist())


# In[26]:


type(coursePreReq_Ser)


# ## Alternating Least Squares(Matrix Factorization)

# In[156]:


import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler


# In[157]:


df = df_new_사원


# In[158]:


data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
data_items = data_mat.reset_index(drop=True)
data_sparse = sparse.csr_matrix(data_items)


# In[165]:


data_index = data_mat.reset_index()
data_index


# In[ ]:





# In[160]:


def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):
    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape
    
    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size = (item_size, features)))
    
    #Precompute I and lambda * I
#     X_I = sparse.eye(user_size)
#     Y_I = sparse.eye(item_size)
    
    I = sparse.eye(features)
    lI = lambda_val * I
    
    # Start main loop. For each iteration we first compute X and then Y
    for i in range(iterations):
#         print ('iteration %d of %d' % (i+1, iterations))
        
        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in range(user_size):

            # Get the user row.
#             u_row = confidence[u,:].toarray() 

            # Calculate the binary preference p(u)
#             p_u = u_row.copy()
#             p_u[p_u != 0] = 1.0
            u_row = sparse_data[u, :].toarray()
    
            p_u = u_row.copy()

            # Calculate Cu and Cu - I
#             CuI = sparse.diags(u_row, [0])
#             Cu = CuI + Y_I

            # Put it all together and compute the final formula
#             yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_pu = Y.T.dot(p_u.T)
            X[u] = spsolve(yTy + lI, yT_pu)

    
        for i in range(item_size):

            # Get the item column and transpose it.
            i_row = sparse_data[:,i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
#             p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
#             CiI = sparse.diags(i_row, [0])
#             Ci = CiI + X_I

            # Put it all together and compute the final formula
#             xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_pi = X.T.dot(p_i.T)
            Y[i] = spsolve(xTx + lI, xT_pi)

    return X, Y


# In[ ]:


def nonzeros(m, row):
    for index in xrange(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]
      
      
def implicit_als_cg(Cui, features=20, iterations=20, lambda_val=0.1):
    user_size, item_size = Cui.shape

    X = np.random.rand(user_size, features) * 0.01
    Y = np.random.rand(item_size, features) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    for iteration in xrange(iterations):
        print 'iteration %d of %d' % (iteration+1, iterations)
        least_squares_cg(Cui, X, Y, lambda_val)
        least_squares_cg(Ciu, Y, X, lambda_val)
    
    return sparse.csr_matrix(X), sparse.csr_matrix(Y)


def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    users, features = X.shape
    
    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in xrange(users):

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in xrange(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

alpha_val = 15
conf_data = (data_sparse * alpha_val).astype('double')
user_vecs, item_vecs = implicit_als_cg(conf_data, iterations=20, features=20)


# In[161]:


user_vecs, item_vecs = implicit_als(data_sparse, iterations=20, features=20, alpha_val=40)


# In[46]:


user_vecs.toarray().shape


# In[263]:


def makeRecDictionaryMF(userList):
    # indexList = []
    userLikesDict_MF = {}
    recommenDict_MF = {}

    for user in userList:
        index = data_index[data_index.O_REG == user].index.values[0]
        user_interactions = data_sparse[index,:].toarray()

        # We don't want to recommend items the user has consumed. So let's
        # set them all to 0 and the unknowns to 1.
        user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
        user_interactions[user_interactions > 1] = 0

        # This is where we calculate the recommendation by taking the 
        # dot-product of the user vectors with the item vectors.
        rec_vector = user_vecs[index,:].dot(item_vecs.T).toarray()

        # Let's scale our scores between 0 and 1 to make it all easier to interpret.
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
        recommend_vector = user_interactions*rec_vector_scaled

        # Get all the artist indices in order of recommendations (descending) and
        # select only the top "num_items" items. 
        item_idx = np.argsort(recommend_vector)[::-1][:10]

        knownlikesindex = np.where(recommend_vector == 0)
        known_user_likes = data_items.columns.values[knownlikesindex]

        userLikesDict_MF[user] = known_user_likes.tolist()
        
        for i in range(len(coursePreReq_Ser.index)):
            if(coursePreReq_Ser.index[i] in known_user_likes.tolist()): 
                recSeries = pd.Series(index = data_mat.columns.values[item_idx]).drop(coursePreReq_Ser[i], errors = 'ignore').index.values
        
        recommenDict_MF[user] = recSeries.tolist()

    dfRec_MF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in recommenDict_MF.items() ])).T
    userLikesSeries = pd.Series(userLikesDict_MF)
    dfRec_MF['userLikes'] = userLikesSeries
        
    return dfRec_MF


# In[264]:


dfRec_MF = makeRecDictionaryMF(userList)


# In[265]:


dfRec_MF.to_excel("MF_사원_2010.xlsx")


# Sampling Users

# In[32]:


def sampleUserList(df, nsamples, seed):
    sampleUsersList = df.O_REG.sample(n=nsamples, random_state=seed).values.tolist()
    return sampleUsersList


# In[57]:


sampleUsersList = sampleUserList(df_new_사원_집합, 50, 2)


# ## item-based recommendations

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


import sklearn


# In[24]:


df_new_사원.shape


# In[30]:


df_NEW_사원.shape


# sampleUsersList = ['2012000338346',
# '2012000346895',
# '2012000473368',
# '2012000253280',
# '2012000488606',
# '2012000369707',
# '2012000214743',
# '2012000256870',
# '2015000653884',
# '2012000355436']
# 

# In[35]:


def calculate_similarity(data_items):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return sim


# In[44]:


def makeRecDictionaryItemBased(usersList, df, df_items, df_neighbours, data_matrix, topN):
    userLikesDict_itembased_nei = {}
    recommenDict_itembased_nei = {}
    for user in usersList:
        user_index = df[df.O_REG == user].index.tolist()[0] # Get the frame index

        known_user_likes = df_items.loc[user_index]
        known_user_likes = known_user_likes[known_user_likes >0].index.values

        most_similar_to_likes = df_neighbours.loc[known_user_likes]
        similar_list = most_similar_to_likes.values.tolist()
        similar_list = list(set([item for sublist in similar_list for item in sublist]))
        neighbourhood = data_matrix[similar_list].loc[similar_list]

        user_vector = df_items.loc[user_index].loc[similar_list]

        score = neighbourhood.dot(user_vector).div(neighbourhood.sum(axis=1))
        score = score.drop(known_user_likes)
        
#         print(type(score))

#         recommenDict_itembased_nei[user] = score.nlargest(10).index.tolist()
        known_user_list = known_user_likes.tolist()
        
        for i in range(len(coursePreReq_Ser.index)):
            if(coursePreReq_Ser.index[i] in known_user_list):
                score = score.drop(coursePreReq_Ser[i], errors = 'ignore')

#         if('자금세탁방지 핵심요원(전문)' in known_user_list):
#             score = score.drop('자금세탁방지 핵심요원(기초)')
#         if('여신법률' in known_user_list):
#             score = score.drop('여신법률(담보관리)기초')
    
        userLikesDict_itembased_nei[user] = known_user_likes.tolist()
        recommenDict_itembased_nei[user] = score.nlargest(topN).index.tolist()
    
    
    dfRec_itembased_nei = pd.DataFrame(recommenDict_itembased_nei).T
    userLikesSeries_nei = pd.Series(userLikesDict_itembased_nei)
    dfRec_itembased_nei['userLikes'] = userLikesSeries_nei
    
    return dfRec_itembased_nei


# In[60]:


def makeRecDictionaryItemBased빈칸(usersList, df, df_items, df_neighbours, data_matrix):
    userLikesDict_itembased_nei = {}
    recommenDict_itembased_nei = {}
    for user in usersList:
        user_index = df[df.O_REG == user].index.tolist()[0] # Get the frame index

        known_user_likes = df_items.loc[user_index]
        known_user_likes = known_user_likes[known_user_likes >0].index.values

        most_similar_to_likes = df_neighbours.loc[known_user_likes]
        similar_list = most_similar_to_likes.values.tolist()
        similar_list = list(set([item for sublist in similar_list for item in sublist]))
        neighbourhood = data_matrix[similar_list].loc[similar_list]

        user_vector = df_items.loc[user_index].loc[similar_list]

        score = neighbourhood.dot(user_vector).div(neighbourhood.sum(axis=1))
        score = score.drop(known_user_likes)
        
#         print(type(score))

#         recommenDict_itembased_nei[user] = score.nlargest(10).index.tolist()
        known_user_list = known_user_likes.tolist()
        
#         for i in range(len(coursePreReq_Ser.index)):
#             if(coursePreReq_Ser.index[i] in known_user_list):
#                 score = score.drop(coursePreReq_Ser[i], errors = 'ignore')

#         if('자금세탁방지 핵심요원(전문)' in known_user_list):
#             score = score.drop('자금세탁방지 핵심요원(기초)')
#         if('여신법률' in known_user_list):
#             score = score.drop('여신법률(담보관리)기초')
    
        userLikesDict_itembased_nei[user] = known_user_likes.tolist()
        score_top10 = score.nlargest(10)
        
        for i in range(len(coursePreReq_Ser.index)):
            if(coursePreReq_Ser.index[i] in known_user_list):
                score_top10 = score_top10.drop(coursePreReq_Ser[i], errors = 'ignore')
        
        recommenDict_itembased_nei[user] = score_top10.index.tolist()
    
    
    dfRec_itembased_nei = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in recommenDict_itembased_nei.items() ])).T
    userLikesSeries_nei = pd.Series(userLikesDict_itembased_nei)
    dfRec_itembased_nei['userLikes'] = userLikesSeries_nei
    
    return dfRec_itembased_nei


# In[41]:


def recommendItemBased(df, usersList, neighbour):

    data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    data = data_mat.reset_index()
#     data_mat.reset_index(inplace = True)
#     data_mat_train, data_mat_test = train_test_split(data, test_size = 0.33, random_state=42)
    data_items = data_mat.reset_index(drop=True)
#     data_items_train, data_items_test = train_test_split(data_items, test_size = 0.20, random_state=42)
    #normalize
    magnitude_total = np.sqrt(np.square(data_items).sum(axis=1))
    data_items = data_items.divide(magnitude_total, axis='index')
    
    data_matrix = calculate_similarity(data_items)
    
    data_neighbours = pd.DataFrame(index=data_matrix.columns, columns=range(1,neighbour+1))
    for i in range(0, len(data_matrix.columns)):
        data_neighbours.iloc[i,:neighbour] = data_matrix.iloc[0:,i].sort_values(ascending=False)[:neighbour].index
        
    dfRec = makeRecDictionaryItemBased빈칸(usersList, data, data_items, data_neighbours, data_matrix)
    return dfRec


# In[61]:


dfRec_itembased_nei = recommendItemBased(df_new_사원_집합, sampleUsersList, 50)


# In[62]:


dfRec_itembased_nei.to_excel("itemBasedRec_사원_2010_집합_0709.xlsx")


# In[ ]:





# In[36]:


def splitingKnownUnknown(testDat, givenNum):
    
    for i in range(testDat.shape[0]):

        user = testDat[i]
        allonesarr = np.where(testDat[i] > 0)[0]
        sampleonesList = np.random.choice(allonesarr, givenNum, replace = False).tolist()
        if i==0:
            known = np.zeros(testDat.shape[1])
            np.put(known, sampleonesList, 1)
            unknown = user - known
        else:
            known_new = np.zeros(testDat.shape[1])
            np.put(known_new, sampleonesList, 1)
            known = np.vstack((known, known_new))
            unknown_new = user - known_new
            unknown = np.vstack((unknown, unknown_new))
    
    return known, unknown


# In[37]:


def normalize(data_items, test_index, test_columns):
    data_items = pd.DataFrame(data_items, index = test_index, columns = test_columns)
    magnitude_total = np.sqrt(np.square(data_items).sum(axis=1))
    data_items_normalized = data_items.divide(magnitude_total, axis='index')
    return data_items_normalized


# In[38]:


def predictItemBasedTrainTest(df_raw, given):
    data_mat = pd.get_dummies(df_raw.N_COUR).groupby(df_raw.O_REG).apply(max)
    data = data_mat.reset_index()
    data_items = data_mat.reset_index(drop=True)
    #splist the data into train and test
    data_items_train, data_items_test = train_test_split(data_items, test_size = 0.20, random_state=42)
    #test dataset index and columns
    test_index = data_items_test.index
    test_columns = data_items_test.columns
    #split test dataset into known and unknown for testing
    test_known, test_unknown = splitingKnownUnknown(data_items_test.values,given)
    #normalize the data
    test_known_norm = normalize(test_known, test_index, test_columns)
    test_unknown_norm = normalize(test_unknown, test_index, test_columns)
    magnitude_total = np.sqrt(np.square(data_items_train).sum(axis=1))
    data_train_norm = data_items_train.divide(magnitude_total, axis='index')
    #calculate Similarity with training dataset
    data_matrix = calculate_similarity(data_train_norm)
        
    #calculate rating matrix(weighted average)
    scoreMat = np.matmul(test_known_norm, data_matrix)
    scoreMatNumer = data_matrix.sum(axis=1)
    scoreMatWeighted = scoreMat.div(scoreMatNumer, axis = 'columns')
    
#     print(scoreMatWeighted)
    
    #index of test dataset users' purchase record
    knownlikesIndex = np.where(test_known_norm.values!=0)
    #drop knownlikesvalues by index
    scoreMatWeighted_values = scoreMatWeighted.values
    scoreMatWeighted_values[knownlikesIndex] = 0
    
    scoreMatWeighted_values_df = pd.DataFrame(scoreMatWeighted_values, index = scoreMat.index, columns = scoreMat.columns)
    
    return scoreMatWeighted_values_df, test_unknown


# In[29]:


def predictItemBasedTrainTestKNN(df_raw, given):
    data_mat = pd.get_dummies(df_raw.N_COUR).groupby(df_raw.O_REG).apply(max)
    data = data_mat.reset_index()
    data_items = data_mat.reset_index(drop=True)
    #splist the data into train and test
    data_items_train, data_items_test = train_test_split(data_items, test_size = 0.20, random_state=42)
    #test dataset index and columns
    test_index = data_items_test.index
    test_columns = data_items_test.columns
    #split test dataset into known and unknown for testing
    test_known, test_unknown = splitingKnownUnknown(data_items_test.values,given)
    #normalize the data
    test_known_norm = normalize(test_known, test_index, test_columns)
    test_unknown_norm = normalize(test_unknown, test_index, test_columns)
    magnitude_total = np.sqrt(np.square(data_items_train).sum(axis=1))
    data_train_norm = data_items_train.divide(magnitude_total, axis='index')
    #calculate Similarity with training dataset
    data_matrix = calculate_similarity(data_train_norm)
    
    data_neighbours = pd.DataFrame(index=data_matrix.columns, columns=range(1,50+1))
    for i in range(0, len(data_matrix.columns)):
        data_neighbours.iloc[i,:50] = data_matrix.iloc[0:,i].sort_values(ascending=False)[:50].index
    
    
    
    #calculate rating matrix(weighted average)
    scoreMat = np.matmul(test_known_norm, data_matrix)
    scoreMatNumer = data_matrix.sum(axis=1)
    scoreMatWeighted = scoreMat.div(scoreMatNumer, axis = 'columns')
    
#     print(scoreMatWeighted)
    
    #index of test dataset users' purchase record
    knownlikesIndex = np.where(test_known_norm.values!=0)
    #drop knownlikesvalues by index
    scoreMatWeighted_values = scoreMatWeighted.values
    scoreMatWeighted_values[knownlikesIndex] = 0
    
    scoreMatWeighted_values_df = pd.DataFrame(scoreMatWeighted_values, index = scoreMat.index, columns = scoreMat.columns)
    
    return scoreMatWeighted_values_df


# In[50]:


import copy


# In[73]:


def evaluateModel(df_score, test_unknown, given, topN):
    # df = scoreMat_values_df
    #record topN courses as 1 and rest of the courses as 0
    #optimize required
    df_scoreMat = copy.deepcopy(df_score)
    test_unknown_func = test_unknown
    for i in range(4774):
        df_scoreMat.iloc[i][df_scoreMat.iloc[i].nlargest(topN).index.tolist()] = 1
    df_scoreMat[df_scoreMat!=1] = 0
    
    #calculate metrics
    TP = np.sum(np.multiply(df_scoreMat.values, test_unknown_func), axis=1) # element-wise multiplication
#     print(TP)
    TP_FN = test_unknown_func.sum(axis=1)
    TP_FP = df_scoreMat.values.sum(axis=1)
#     print(TP_FP)
    FP = np.subtract(TP_FP, TP)
#     print(FP)
    FN = np.subtract(TP_FN, TP)
    TN = test_unknown_func.shape[1] - given - TP - FP - FN
#     print('given number is', given)
#     print('TopN is ', topN)
    precision = np.divide(TP,np.add(TP, FP))
    recall = np.divide(TP,np.add(TP, FN))
    TPR = recall
    FPR = np.divide(FP,np.add(FP, TN))
    
    metrics = [{'precisionAvg' : np.mean(precision), 'recallAvg' : np.mean(recall), 'TPRAvg' : np.mean(TPR), 'FPRAvg' : np.mean(FPR)}]
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df
    
    
    
    


# In[48]:


scoreMat, test_unknown_df = predictItemBasedTrainTest(df_NEW_사원, 4)


# In[49]:


scoreMat


# In[74]:


metricsdf_1 = evaluateModel(scoreMat, test_unknown_df, 4, 1)


# In[75]:


metricsdf_1


# In[76]:


metricsdf_3 = evaluateModel(scoreMat, test_unknown_df, 4, 3)


# In[77]:


metricsdf_3


# In[78]:


metricsdf_5 = evaluateModel(scoreMat, test_unknown_df, 4, 5)


# In[79]:


metricsdf_5


# In[80]:


metricsdf_10 = evaluateModel(scoreMat, test_unknown_df, 4, 10)


# In[81]:


metricsdf_10


# ## CF - User-Based

# In[195]:


def calculateUserSim(data_items):
    
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse)
    sim = pd.DataFrame(data=similarities, index= data_items.index, columns= data_items.index)
    return sim


# In[ ]:


def normalize(data_items, test_index, test_columns):
    data_items = pd.DataFrame(data_items, index = test_index, columns = test_columns)
    magnitude_total = np.sqrt(np.square(data_items).sum(axis=1))
    data_items_normalized = data_items.divide(magnitude_total, axis='index')
    return data_items_normalized


# In[ ]:


def predictUserBasedTrainTest(df, given):
    data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    data = data_mat.reset_index()
    data_items = data_mat.reset_index(drop=True)
    #splist the data into train and test
    data_items_train, data_items_test = train_test_split(data_items, test_size = 0.20, random_state=42)
    #test dataset index and columns
    test_index = data_items_test.index
    test_columns = data_items_test.columns
    #split test dataset into known and unknown for testing
    test_known, test_unknown = splitingKnownUnknown(data_items_test.values,given)
    #normalize the data
    test_known_norm = normalize(test_known, test_index, test_columns)
    test_unknown_norm = normalize(test_unknown, test_index, test_columns)
    magnitude_total = np.sqrt(np.square(data_items_train).sum(axis=1))
    data_train_norm = data_items_train.divide(magnitude_total, axis='index')
    #calculate Similarity with training dataset
    data_matrix = calculateUserSim(data_train_norm)
        
    #calculate rating matrix(weighted average)
    scoreMat = np.matmul(test_known_norm, data_matrix)
    scoreMatNumer = data_matrix.sum(axis=1)
    scoreMatWeighted = scoreMat.div(scoreMatNumer, axis = 'columns')
    
#     print(scoreMatWeighted)
    
    #index of test dataset users' purchase record
    knownlikesIndex = np.where(test_known_norm.values!=0)
    #drop knownlikesvalues by index
    scoreMatWeighted_values = scoreMatWeighted.values
    scoreMatWeighted_values[knownlikesIndex] = 0
    
    scoreMatWeighted_values_df = pd.DataFrame(scoreMatWeighted_values, index = scoreMat.index, columns = scoreMat.columns)
    
    return scoreMatWeighted_values_df


# In[ ]:





# In[152]:


def makeRecDictionaryUserBased(usersList, df, df_items, user_pred):
    userLikesDict_userbased = {}
    recommenDict_userbased = {}
    for user in usersList:
        user_index = df[df.O_REG == user].index.tolist()[0] # Get the frame index

        # # Get the artists the user has likd.
        known_user_likes = df_items.loc[user_index]
        #print(type(known_user_likes))
        known_user_likes = known_user_likes[known_user_likes >0].index.values

        user_pred_df_user = user_pred.iloc[user_index]

        # # Remove the known likes from the recommendation.
        user_pred_df_score = user_pred_df_user.drop(known_user_likes)
        
        known_user_list = known_user_likes.tolist()
        
#         for i in range(len(coursePreReq_Ser.index)):
#             if(coursePreReq_Ser.index[i] in known_user_list):
#                 score = score.drop(coursePreReq_Ser[i], errors = 'ignore')
        
#         if('자금세탁방지 핵심요원(전문)' in known_user_list):
#             user_pred_df_score = user_pred_df_score.drop('자금세탁방지 핵심요원(기초)')
#         if('여신법률' in known_user_list):
#             user_pred_df_score = user_pred_df_score.drop('여신법률(담보관리)기초')

        recommenDict_userbased[user] = user_pred_df_score.nlargest(10).index.tolist()
        userLikesDict_userbased[user] = known_user_likes.tolist()
    
    dfRec_userbased = pd.DataFrame(recommenDict_userbased).T
    userLikesSeries = pd.Series(userLikesDict_userbased)
    dfRec_userbased['userLikes'] = userLikesSeries
    
    return dfRec_userbased


# In[153]:


def recommendUserBased(df, usersList):

    data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    data_items = data_mat.reset_index(drop=True)
    #normalize
    magnitude_total = np.sqrt(np.square(data_items).sum(axis=1))
    data_items = data_items.divide(magnitude_total, axis='index')
        
    user_pred = calculateUserSim(data_mat)
    data = data_mat.reset_index()

    dfRec = makeRecDictionaryUserBased(usersList, data, data_items, user_pred)
    return dfRec


# In[149]:


sampleUsersList2 = sampleUserList(df_new_사원_2015, 50, 2)


# In[155]:


dfRec_userbased = recommendUserBased(df, userList)


# In[430]:


dfRec_userbased.to_excel("userBasedRec_사원_2015.xlsx")


# # Logistic Regression

# In[127]:


from sklearn.model_selection import train_test_split


# In[355]:


user_item_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
user_item_mat


# In[356]:


user_item_mat_train, user_item_mat_test = train_test_split(user_item_mat, test_size = 0.15, random_state=42)


# In[130]:


data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    
train_df, test_df = train_test_split(data_mat, test_size = 0.15, random_state=42)
    
userList = pd.Series(test_df.index).sample(n=50, random_state=2).values.tolist()


# In[366]:


def recLogistic(df):
    
    data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    
    train_df, test_df = train_test_split(data_mat, test_size = 0.15, random_state=42)
    
    userList = pd.Series(test_df.index).sample(n=50, random_state=2).values.tolist()
    
    num_users = len(userList)
    num_items = num_features + 1
    
    user_pred = np.zeros((num_users, num_items))
        
    for i in range(num_items):
        x_train = train_df.drop(train_df.columns[i],axis=1)
        y_train = train_df.iloc[:,i]
        x_test = test_df.drop(test_df.columns[i],axis=1)
        x_test = x_test.loc[userList]
#         y_test = test_df.iloc[:,i]
        
        logmodel = LogisticRegression(solver='lbfgs', max_iter = 200)
#         print(i)
        logmodel.fit(x_train,y_train)
        
        purchaseProb = logmodel.predict_proba(x_test)[:,1] # array (#examples, )
        
#         active_user_features = active_user_ratings.drop(active_user_ratings.columns[i], axis=1)
#         active_user_features = active_user_features.values # 10 X 373
# #         purchase_prob = sigmoid(np.inner(W, active_user_features)) # 10 X 1
        
        user_pred[:,i] = purchaseProb
        
        ### delete known userlikes in each row(user) or make it zero and select most probable items for each row
    user_pred_df = pd.DataFrame(user_pred)
    user_pred_df.columns = data_mat.columns.tolist()     
    
    userLikesDict_logistic = {}
    recommenDict_logistic = {}
    
    for i in range(num_users):
        
        # # Get the artists the user has likd.
        useri_history = test_df.loc[[userList[i]]]
        known_user_likes = np.where(useri_history == 1)[1].tolist()

        user_pred_df_score = user_pred_df.iloc[[i]].drop(user_pred_df.columns[known_user_likes], axis=1)
        
        recommenDict_logistic[userList[i]] = user_pred_df_score.iloc[0].nlargest(10).index.tolist()
        userLikesDict_logistic[userList[i]] = user_pred_df.columns[known_user_likes].values.tolist()
    
    dfRec_logistic = pd.DataFrame(recommenDict_logistic).T
    userLikesSeries = pd.Series(userLikesDict_logistic)
    dfRec_logistic['userLikes'] = userLikesSeries
    
    return dfRec_logistic


# In[367]:


dfRec_Logistic = recLogistic(df)


# In[372]:


dfRec_Logistic.to_excel("logistic_사원_2010.xlsx")


# In[ ]:





# In[381]:


num_features = user_item_mat_values.shape[1] - 1
w = np.zeros(num_features)
N = user_item_mat_train.shape[0] #number of training examples


# In[139]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[140]:


def loss(x,y,w):
    loss=0
    for i in range(N):
        loss += -y[0,i]*np.log(sigmoid(np.dot(w.T, x[:,i]))) - (1 - y[0,i])*np.log(1 - np.dot(w.T, x[:,i]))
    return loss


# In[141]:


def gradient(x,y,w):
    lamda = 10
    gradient = np.zeros(num_features)
    for i in range(N):
        D = sigmoid(np.dot(w.T,x[:,i])) - y[0,i]
        G = D * x[:,i]
        gradient += G
    gradient += 2*lamda*w
    return gradient


# In[142]:


def hessian(x,y,w):
    lamda = 10
    hessian = np.zeros((num_features,num_features))
    I = np.identity(num_features)
    for i in range(N):
        xxt = np.outer(x[:,i],x[:,i])
        sigmoidsquared = sigmoid(np.dot(w.T,x[:,i]))*(1 - sigmoid(np.dot(w.T, x[:,i])))
        H = xxt * sigmoidsquared
        hessian += H
    hessian += 2*lamda*I
    return hessian


# In[143]:


def Newton(x, y, tol):
    
    w = np.zeros(num_features)
    lamda = 10
    error = 10
    
    while error>tol :
        
        l1_train = loss( x, y, w)
        norm = np.linalg.norm(w)
        l1_train+= lamda*(norm**2)
        
        G = gradient(x, y, w)
        
        H = hessian(x, y, w)
        
        w = w - np.dot(np.linalg.pinv(H), G)
        
        l2_train = loss(x, y, w)
        norm = np.linalg.norm(w)
        l2_train += lamda*(norm**2)
        error = np.abs(l2_train - l1_train)
        
    return w


# In[388]:


def recLogisticMyOwn(df):
    
    data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    
    train_df, test_df = train_test_split(data_mat, test_size = 0.15, random_state=42)   
    userList = pd.Series(test_df.index).sample(n=50, random_state=2).values.tolist()
    
    num_users = len(userList)
    num_items = num_features + 1
    
    user_pred = np.zeros((num_users, num_items))
    
    for i in range(num_items):
        x_train = train_df.drop(train_df.columns[i],axis=1)
#         x_train = x_train.values
        x_train = x_train.T
        y_train = train_df.iloc[:,i]
        y_train = y_train.T
        x_test = test_df.drop(test_df.columns[i],axis=1)
        x_test = x_test.loc[userList]
        x_test = x_test.T
#         x_test = np.delete(user_item_mat_test, i, axis=1)
#         x_test = x_test.T
#         y_test = user_item_mat_test[:,i]
#         y_test = y_test.reshape(-1,1)
#         y_test = y_test.T

        W = Newton(x_train.values, y_train.values, 10**(-6)) # 1 X 373

        
#         active_user_features = active_user_ratings.drop(active_user_ratings.columns[i], axis=1)
#         active_user_features = active_user_features.values # 10 X 373
        purchase_prob = sigmoid(np.inner(W, x_test.values)) # 10 X 1
        
        user_pred[:,i] = purchase_prob
        
        ### delete known userlikes in each row(user) or make it zero and select most probable items for each row
    user_pred_df = pd.DataFrame(user_pred)
    user_pred_df.columns = df.columns.tolist()     
    
    userLikesDict_logistic = {}
    recommenDict_logistic = {}
    
    for i in range(num_users):
        
        # # Get the artists the user has likd.
        useri_history = test_df.loc[[userList[i]]]
        known_user_likes = np.where(useri_history == 1)[1].tolist()

        user_pred_df_score = user_pred_df.iloc[[i]].drop(user_pred_df.columns[known_user_likes], axis=1)
        
        recommenDict_logistic[userList[i]] = user_pred_df_score.iloc[0].nlargest(10).index.tolist()
        userLikesDict_logistic[userList[i]] = user_pred_df.columns[known_user_likes].values.tolist()
    
    dfRec_logistic = pd.DataFrame(recommenDict_logistic).T
    userLikesSeries = pd.Series(userLikesDict_logistic)
    dfRec_logistic['userLikes'] = userLikesSeries
    
    return dfRec_logistic
        


# In[118]:


def recLogistic(df, userList):
    
    num_users = len(userList)
    num_items = num_features + 1
    
    user_pred = np.zeros((num_users, num_items))
    
    data_mat = pd.get_dummies(df.N_COUR).groupby(df.O_REG).apply(max)
    
    data = data_mat.reset_index()
    
    data_items = data_mat.reset_index(drop=True)
    data_items_values = data_items.values
    user_item_mat_train, user_item_mat_test = train_test_split(data_items_values, test_size = 0.15, random_state=42) #numpy ndarray
    
    user_index_list = []
    for user in userList:
        user_index = data[data.O_REG == user].index.tolist()[0]
        user_index_list.append(user_index)
        
    active_user_ratings = data_items.iloc[user_index_list,:] #dataframe
    
    for i in range(num_items):
        x_train = np.delete(user_item_mat_train, i, axis=1) ## #users X #features
        x_train = x_train.T  ## #features X #users
        y_train = user_item_mat_train[:,i]
        y_train = y_train.reshape(-1,1) ## #users X 1
        y_train = y_train.T  ## 1 X #users
#         x_test = np.delete(user_item_mat_test, i, axis=1)
#         x_test = x_test.T
#         y_test = user_item_mat_test[:,i]
#         y_test = y_test.reshape(-1,1)
#         y_test = y_test.T

        W = Newton(x_train, y_train, 10**(-6)) # 1 X 373

        
        active_user_features = active_user_ratings.drop(active_user_ratings.columns[i], axis=1)
        active_user_features = active_user_features.values # 10 X 373
        purchase_prob = sigmoid(np.inner(W, active_user_features)) # 10 X 1
        
        user_pred[:,i] = purchase_prob
        
        ### delete known userlikes in each row(user) or make it zero and select most probable items for each row
    user_pred_df = pd.DataFrame(user_pred)
    user_pred_df.columns = df.columns.tolist()     
    
    userLikesDict_logistic = {}
    recommenDict_logistic = {}
    
    for i in range(num_users):
        
        # # Get the artists the user has likd.
        known_user_likes = data_items.loc[user_index_list[i]]
        known_user_likes = known_user_likes[known_user_likes >0].index.values #numpy ndarray

        user_pred_df_score = user_pred_df.iloc[i].drop(known_user_likes)
        
        recommenDict_logistic[userList[i]] = user_pred_df_score.nlargest(10).index.tolist()
        userLikesDict_logistic[userList[i]] = known_user_likes.tolist()
    
    dfRec_logistic = pd.DataFrame(recommenDict_logistic).T
    userLikesSeries = pd.Series(userLikesDict_logistic)
    dfRec_logistic['userLikes'] = userLikesSeries
    
    return dfRec_logistic
        


# In[173]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(classification_report(y_test,predictions))
print("Accuracy:",accuracy_score(y_test, predictions))


# ### Example.. no need to run due to repetitiveness

# In[94]:


data_matrix['(금토) 비즈니스 매너와 의전 사례연구'].nlargest(4, keep='all').index.tolist()[1:]


# In[86]:


sampleDict = {}
sampleDict['전략적블라블라'] = data_matrix.iloc[1].nlargest(4, keep='all').index.tolist()[1:]


# In[87]:


sampleDict


# In[61]:


pd.DataFrame(data_matrix.index).to_excel('courseName.xlsx')


# In[65]:


type(data_matrix.index.tolist())


# ## Dictionary with Similar Courses

# 겹치는 목록까지..

# In[92]:


simDict = {}

for courseName in data_matrix.index.tolist():
    simDict[courseName] = data_matrix[courseName].nlargest(4, keep='all').index.tolist()[1:]


# 겹치는것중 첫번째 것만

# In[94]:


simDict2 = {}

for courseName in data_matrix.index.tolist():
    simDict2[courseName] = data_matrix[courseName].nlargest(4, keep='last').index.tolist()[1:]


# In[27]:


simDict2_euclidean = {}

for courseName in data_matrix_euclidean.index.tolist():
    simDict2_euclidean[courseName] = data_matrix_euclidean[courseName].nlargest(4, keep='last').index.tolist()[1:]


# In[ ]:





# In[95]:


import csv


# In[95]:


with open('simCourses_1_nonormalize.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for key, val in simDict.items():
        writer.writerow([key,val])
    


# In[35]:


with open('simCourses_2_nonormalize_euclidean.csv', 'w') as outfile2:
    writer2 = csv.writer(outfile2)
    for key, val in simDict2_euclidean.items():
        writer2.writerow([key,val])


# In[ ]:





# In[ ]:





# ### Real EDA

# In[52]:


df_post2010_NF.columns


# In[63]:


df_post2010.N_COUR.value_counts()


# In[57]:


df_post2010.N_CSTM.value_counts()


# In[58]:


top10dict_pre = {}
for company in df_post2010.N_CSTM.value_counts().head(10).index.tolist():
    top10dict_pre[company] = df_post2010[df_post2010.N_CSTM == company].N_COUR.value_counts().index[:10].tolist()


# In[59]:


top10dict_pre


# In[61]:


import csv


# In[62]:


with open('top10pre.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for key, val in top10dict_pre.items():
        writer.writerow([key,val])


# In[103]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("은행실무")==True].N_COUR.value_counts()


# In[102]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("은행실무")==True].N_CSTM.value_counts()


# In[48]:


df_post2010_NF.columns


# In[94]:


df_post2010_NF[df_post2010_NF.N_CSTM=='개인고객'].N_COUR.value_counts()


# In[88]:


top10dict = {}
for company in df_post2010_NF.N_CSTM.value_counts().head(10).index.tolist():
    top10dict[company] = df_post2010_NF[df_post2010_NF.N_CSTM == company].N_COUR.value_counts().index[:10].tolist()


# In[90]:


top10dict['개인고객'] = df_post2010_NF[df_post2010_NF.N_CSTM == '개인고객'].N_COUR.value_counts().index[:10].tolist()


# In[91]:


top10dict


# In[96]:


with open('top10.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for key, val in top10dict.items():
        writer.writerow([key,val])


# In[74]:


df_post2010_NF[df_post2010_NF.N_CSTM == '한국스탠다드차타드은행'].N_COUR.value_counts().index[:10].tolist()


# In[80]:


df_post2010_NF.N_CSTM.value_counts().head(10).index.tolist()


# In[40]:


colnamesDict = {
    'O_REG' : '고객관리번호',
    'N_COUR' : '과정명',
    'I_COUR' : '과정코드',
    'D_OPN_YY' : '개강년도',
    'O_SIM_NUM' : '기수',
    'N_CSTM' : '기관코드명',
    'I_INST' : '기관코드',
    'N_SUB_BR' : '부지점명',
    'I_EXE_ST' : '수료여부',
    'I_TRAN_NAME' : '연수형태',
    'I_TRAN_FLD_NAME' : '연수분야(대분류)',
    'I_TRAN_FLD_MID_NAME' : '연수분야(중분류)',
    'I_TRAN_TP_NAME' : '정규,맞춤 여부',
    'Y_EXE' : '실시여부(과정)'}
colnamesDict


# In[42]:


df_post2010_NF.I_TRAN_FLD_NAME


# In[37]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("영문금융계약서")==True].N_SUB_BR.value_counts()


# In[53]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("영문금융계약서")==True].N_CSTM.value_counts()


# In[55]:


df_post2010_NF.N_COUR.value_counts()


# In[ ]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("영문금융계약서")==True]


# In[100]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("은행실무")==True].N_CSTM.value_counts()


# In[118]:


df_post2010_NF.N_COUR.value_counts()


# In[68]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("KBI 그")]


# In[71]:


df_post2010_NF[df_post2010_NF['N_COUR'].str.contains("KBI 금융")==True].N_CSTM.value_counts()


# In[73]:


224/1346


# In[ ]:





# In[ ]:





# In[83]:


pd.DataFrame(df_post2010_NF.N_CSTM.unique()).to_excel("기관명.xlsx")


# In[95]:


df_post2010_NF.N_CSTM.value_counts().to_excel("기관명sorted.xlsx")


# In[96]:


type(df_post2010_NF.N_CSTM.value_counts())


# In[98]:


pd.DataFrame(df_post2010_NF.N_COUR.unique()).to_excel("과정명s.xlsx")


# In[16]:


#개인정보 데이터
df_PI = pd.read_csv('Personal_InfoData.csv', encoding = "ms949", low_memory=False)


# In[44]:


df_trainee['O_REG'].isin(df_PI['O_REG']).value_counts()


# In[33]:


df_PI[df_PI['O_REG']=='9999999992056']


# In[6]:


df_PI[df_PI['AGE'] < 20]


# In[52]:


df_exclusive = df_trainee.loc[df_trainee['O_REG'].isin(df_PI['O_REG'])==False].iloc[0:10]


# In[54]:


df_exclusive.to_excel("exclusive.xlsx")


# In[13]:


#연수 과정기수 데이터
df_TRAINING = pd.read_csv('연수과정기수.csv', encoding = 'ms949')


# In[14]:


df_TRAINING.head()


# In[56]:


df_TRAINING.columns


# In[72]:


df_TRAINING['I_TRAN_TP_NAME'].value_counts()


# In[76]:


df_TRAINING['D_OPN_YY'].unique()


# In[78]:


df_TRAINING_post2010 = df_TRAINING[df_TRAINING['D_OPN_YY']>=2000]


# In[79]:


df_TRAINING_post2010['D_OPN_YY'].unique()


# In[81]:


df_TRAINING_post2010.columns


# In[82]:


df_TRAINING_post2010_NF = df_TRAINING_post2010[['I_COUR', 'N_COUR', 'N_SIM_COUR', 'D_OPN_YY', 'O_SIM_NUM','I_TRAN_NAME', 'I_TRAN_TP_NAME']]


# In[83]:


df_TRAINING_post2010_NF


# #수강생정보

# In[11]:


#수강생정보 데이터
df_trainee = pd.read_csv('신청수강생.csv', encoding = 'ms949')


# In[62]:


df_trainee[df_trainee['I_COUR']==266]['O_SIM_NUM'].unique()


# In[18]:


df_trainee_new = pd.read_csv('수강생.csv', encoding = 'ms949')


# In[21]:


df_trainee_new[df_trainee_new.O_REG == 9999999992056].N_CSTM


# In[10]:


df_trainee_new.columns


# In[12]:


df_trainee.columns


# In[84]:


df_trainee_new_NF = df_trainee_new[['O_REG', 'I_COUR', 'D_OPN_YY', 'O_SIM_NUM', 'I_INST', 'N_CSTM', 'N_SUB_BR','I_EXE_ST']]


# In[86]:


df_trainee_new_NF['I_EXE_ST'].unique()


# In[90]:


df_trainee_new_post2010_NF= df_trainee_new_NF[df_trainee_new_NF['D_OPN_YY']>=2010]


# In[94]:


df_trainee_new_post2010_NF['I_COUR'].isin(df_TRAINING_post2010_NF['I_COUR']).value_counts()


# In[95]:


df1 = df_TRAINING_post2010_NF
df2 = df_trainee_new_post2010_NF


# In[96]:


df1.head()


# In[97]:


df2.head()


# In[91]:


df_merged_code = pd.merge(df_TRAINING_post2010_NF, df_trainee_new_post2010_NF, on = 'I_COUR')


# In[ ]:





# In[ ]:





# In[100]:


#연수정보전체데이터
df_TrainMerged = pd.read_csv('수강정보.csv', encoding = 'ms949')


# 연수전체정보데이터 주에 필요한것들..

# In[25]:


df_TrainMerged_NF = df_TrainMerged[['O_REG', 'I_COUR', 'D_OPN_YY', 'O_SIM_NUM', 'N_CSTM', 'I_INST', 'N_SUB_BR','I_CMPL_GBN_NAME', 'I_TRAN_NAME', 'I_TRAN_FLD_NAME', 'I_TRAN_FLD_MID_NAME', 'I_TRAN_TP_NAME']]


# In[26]:


df_TrainMerged_NF.head()


# In[104]:


df_TrainMerged_NF.shape


# In[106]:


len(df_TrainMerged_NF['O_REG'].unique())


# In[111]:


df_TrainMerged['O_SIM_NUM'].unique()


# In[112]:


df_TrainMerged[df_TrainMerged['O_SIM_NUM']=='9-s       ']


# In[110]:


df_TrainMerged[['I_COUR', 'D_OPN_YY', 'O_SIM_NUM']].sort_values('I_COUR')


# In[27]:


df_merged = pd.merge(df_PI_NF, df_TrainMerged_NF, on = 'O_REG' )


# In[29]:


df_PI_NF.shape


# In[30]:


df_TrainMerged_NF.shape


# In[33]:


df_merged.shape


# In[71]:


len(df_merged['O_REG'].unique())


# In[34]:


df_merged[df_merged['O_REG']==2012000010508]


# In[35]:


df_merged['count'] = df_merged.groupby('O_REG')['O_REG'].transform('count')


# In[43]:


df_merged_over5 = df_merged[df_merged['count']>=5]


# In[49]:


companyCount =  df_merged_over5['N_CSTM_x'].value_counts()


# In[52]:


companyCount


# In[45]:


df_companycount = df_merged_over5['N_CSTM_x'].groupby('N_CSTM_x')[]


# In[53]:


df_merged_over5_KB = df_merged_over5[df_merged_over5['N_CSTM_x']=='국민은행']


# In[59]:


df_merged_over5_KB[df_merged_over5_KB['AGE']==20]


# In[62]:


df_TrainMerged[df_TrainMerged['O_REG']==2017000726964]['N_CSTM']


# In[63]:


df_merged_outer = pd.merge(df_PI_NF, df_TrainMerged_NF, on = 'O_REG' , how = 'outer')


# In[64]:


df_merged_outer.shape


# In[66]:


df_PI['O_REG'].isin(df_TrainMerged['O_REG']).value_counts()


# In[67]:


df_PI.shape


# In[69]:


len(df_PI['O_REG'].unique())


# In[72]:


df_TrainMerged['O_REG'].isin(df_PI['O_REG']).value_counts()


#     Association Rule - Market Basket Analysis

# In[1]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[11]:


df_trainee.columns


# In[13]:


len(df_TRAINING)


# In[32]:


df_TRAINING['N_COUR'].value_counts()


# In[25]:


pd.DataFrame(df_TRAINING['I_COUR'].unique()).to_csv("courses.csv")


# In[28]:


pd.DataFrame(df_TRAINING['N_COUR'].unique()).to_csv("course_name.csv", encoding = "ms949")


# In[20]:


len(df_TRAINING['N_COUR'].unique())


# In[33]:


len(df_TRAINING['I_COUR'].unique())


# In[21]:


len(df_trainee['I_COUR'].unique())


# In[65]:


df_trainee


# In[ ]:




