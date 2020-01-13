#!/usr/bin/env python
# coding: utf-8

# 1. conda 설치
# 
#     https://www.anaconda.com/distribution/#download-section
# 
# 2. conda 환경설정
# 
#     CONDA_HOME
#         C:\ProgramData\Anaconda3
#     Path 추가
#         %CONDA_HOME%\;
#         %CONDA_HOME%\condabin\;
# 
# 3. conda 생성
# 
#     terminal 명령어
#     conda 환경 활성화
#         conda activate condaenv_name
#         
# 4. 모듈 설치 방법
# 
#     conda로 설치하는 경우
#         conda install module
#     python으로 설치하는 경우
#         pip install module
# 
# 5. dbconnect3.py 실행 관련
# 
#     jdk 설치 및 JAVA_HOME 설정
#         https://prolite.tistory.com/975
#     ojdbc8.jar 파일을 startJVM에서 지정된 class path 위치로 복사
#     jaydebeapi.connect 메소드의 parameter DB 주소로 맞게 변경
#     터미널에서 dbconnect3.py 있는 위치로 가서 아래 명령어 차례로 수행(conda 환경이 activate 되어있는 상태여야 함)
#         pip install JPype1
#         pip install jaydebeapi
#         python dbconnect3.py
# 
# 6. jupyter notebook 실행
#     anaconda prompt(터미널)에서 jupyter notebook 입력.

# In[1]:


get_ipython().system('pip install xlrd')
get_ipython().system('pip install openpyxl')


# In[ ]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')


# In[1]:


import jpype
import jaydebeapi
import pandas as pd
import numpy as np


# In[2]:


import time


# In[3]:


path = r"-Djava.class.path=C:\Users\HP\Desktop\sqldeveloper-17.4.0.355.2349-x64\sqldeveloper\jdbc\lib\ojdbc8.jar"
driver = 'oracle.jdbc.driver.OracleDriver'
database = 'jdbc:oracle:thin:xcom/XCOM@//172.16.2.178:2483/KBIDB2'


# ### 연결해제시키는 함수들인데 unnecessary?
# jpype.shutdownJVM()
# curs.close()
# self.conn.close()

# In[4]:


class SqlData:
    
    # sql서버와 연결하는 함수.
    def sqlconnect(self, path, driver, database):
        
        jHome = jpype.getDefaultJVMPath()
        jpype.startJVM(jHome, path)
        self.conn = jaydebeapi.connect(driver, database)
        
#         jpype.shutdownJVM()
    
    # sql 쿼리를 이용하여 데이터프레임을 만드는 함수.
    def createDF(self, query):
        
        curs = self.conn.cursor()
        curs.execute(query)
        
        data = curs.fetchall()
        columnNames = [curs.description[i][0] for i in range(len(curs.description))]
        
        self.df = pd.DataFrame(data, columns = columnNames)
        
        print("shape of the dataframe is ", self.df.shape)
        print("column names(features) are ", self.df.columns.values)
        
        return self.df


# In[5]:


purchaseData = SqlData()
purchaseData.sqlconnect(path, driver, database)


# In[6]:


df_과정_raw = purchaseData.createDF('''select a.o_reg,
       a.i_cour, 
       a.d_opn_yy,
       a.o_sim_num,
       (select n_cstm from xcom.XCOM_CSTM b where c.i_inst = b.i_cstm) as n_cstm,
       a.N_SUB_BR,
       (select n_dtl from XCOM.XCOD_COMMON where o_dtl = a.I_JOB_GRD) as I_JOB_GRD_name,
       (select n_dtl from XCOM.XCOD_COMMON where o_dtl = a.I_exe_st) as I_exe_st,
       k.N_COUR,
       k.Y_EXE,
       k.I_TRAN_name,
       k.I_TRAN_FLD_name,
       k.I_TRAN_FLD_MID_name,
       k.I_TRAN_TP_name
  from inst.TXXM_TKLECTPSN a
        left outer join inst.TXXM_TKLECTAPLC c 
            on a.i_cour = c.i_cour and a.o_sim_num = c.o_sim_num and a.d_opn_yy = c.d_opn_yy and a.o_reg = c.o_reg and a.o_tk_lect = c.o_tk_lect
        left outer join (select a.i_cour, c.N_COUR, a.D_OPN_YY, a.O_SIM_NUM,
                            a.Y_EXE,
                            (select n_dtl from XCOM.XCOD_COMMON where o_dtl = c.I_TRAN) as I_TRAN_name,
                            (select n_dtl from XCOM.XCOD_COMMON where o_dtl = c.I_TRAN_FLD) as I_TRAN_FLD_name,
                            (select n_dtl from XCOM.XCOD_COMMON where o_dtl = c.I_TRAN_FLD_MID) as I_TRAN_FLD_MID_name,
                            (select n_dtl from XCOM.XCOD_COMMON where o_dtl = c.I_TRAN_TP) as I_TRAN_TP_name
                    from inst.TXXM_OPNCOUR a 
                        left outer join nlms.TB_LMS_OPNCOUROPTION b on a.i_cour = b.i_cour and a.D_OPN_YY = b.D_OPN_YY and a.o_sim_num = b.O_SIM_NUM,
                        inst.TXXM_COURCD c
                    where a.i_cour = c.i_cour) k on a.i_cour = k.i_cour and a.o_sim_num = k.o_sim_num and a.d_opn_yy = k.d_opn_yy
  where a.d_opn_yy between 2009 and 2018''')


# In[6]:


df_자격_raw = purchaseData.createDF('''select d.o_reg,
       d.I_QLFN,
       c.n_QLFN,
       a.D_YY,
       d.I_EXAM,
       (select n_cstm from xcom.XCOM_CSTM c where d.i_inst = c.i_cstm) as I_INST_name
  from qual.QEXM_ACPTINFO a 
    left outer join qual.QEXM_RESULT d on a.O_ACPT = d.O_ACPT and a.I_QLFN = d.I_QLFN AND a.Q_NUM = d.Q_NUM
    left outer join qual.QCEM_QLFNPSN e on d.o_reg = e.o_reg and d.I_EXAM = e.I_EXAM and d.I_QLFN = e.I_QLFN, 
    qual.QBSM_QLFNCD c
  where a.I_QLFN = c.I_QLFN''')


# In[7]:


df_도서_raw = purchaseData.createDF('''select a.O_ODR,
       f.O_REG,
       a.N_PURC_ID,
       a.I_SALE_PK,
       a.D_ACPT,
       a.I_ODR,
       a.Y_STTL_MTHD,
       a.W_TOT,
       a.W_FACT_SALE,
       b.i_book,
       b.Q_ODR,
       b.W_ODR_CST,
       b.P_DISC,
       b.W_FACT_RC_MONY,
       c.N_BOOK,
       (select n_dtl from XCOM.XCOD_COMMON f where f.o_dtl = c.I_BOOK_GBN) as I_BOOK_GBN_name,
       (select n_dtl from XCOM.XCOD_COMMON f where f.o_dtl = c.I_BOOK_SORT) as I_BOOK_SORT_name,
       c.Y_BEST_SELR,
       (select n_dtl from XCOM.XCOD_COMMON f where f.o_dtl = c.I_RCMD) as I_RCMD_name,
       (select n_dtl from XCOM.XCOD_COMMON f where f.o_dtl = e.I_SND_BK) as I_SND_BK_name,
       (select n_dtl from XCOM.XCOD_COMMON f where f.o_dtl = e.I_SND_BK_TP) as I_SND_BK_TP_name,
       e.N_SND_BK_RSN,
       c.I_QLFN,
       c.I_COUR 
from book.BORM_ORDER a
    left outer join book.BORD_ORDER b on a.O_ODR = b.O_ODR
    left outer join book.BSBM_SNDBK e on b.i_book = e.i_book and b.O_ODR = e.O_ODR
    left outer join book.BINM_BOOK c on B.i_book = c.i_book
    left outer join xcom.XCOM_MEMBER f on a.N_PURC_ID = f.I_ID''')


# In[7]:


class DataPreprocess과정:
    
    # 연도 변수를 string에서 int로 변환.
    def __init__(self, df):
        
        self.df = df
        self.df['D_OPN_YY'] = df['D_OPN_YY'].astype('int64')
        print("shape of the dataframe is ", self.df.shape)
    
    # 연도범위 설정.
    def defineYearRange(self, year1, year2):
        
        self.df = self.df[self.df['D_OPN_YY']>=year1]
        self.df = self.df[self.df['D_OPN_YY']<=year2]
        print("shape of the dataframe is ", self.df.shape)
    
    # TestID 그리고 모의 수업들 제외.
    def removeTestCoursesID(self):
        
        #9999로 시작하는 고객번호(O_REG)는 Test ID 이므로 제외.
        self.df = self.df[~self.df['O_REG'].str.contains("99999", na=False)]
        #모의수업들 제외.
        self.df = self.df[self.df.N_COUR != '통신모의']
        self.df = self.df[self.df.N_COUR != 'KBI모의']
        self.df = self.df[self.df.N_COUR != '집합 모의']
        
        print("shape of the dataframe is ", self.df.shape)
    
    # 과정명에 불필요한 빈칸 제거.
    def stripCourseNames(self):
        
        self.df.loc[:,'N_COUR'] = self.df.loc[:,'N_COUR'].str.rstrip()
        self.df.loc[:,'N_COUR'] = self.df.loc[:,'N_COUR'].str.lstrip()
        
        print("shape of the dataframe is ", self.df.shape)
    
    # 맞춤성격을 가진 정규수업들 맞춤과정으로 재분류.
    def 맞춤재분류(self):
        
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

        for word in wordList_기관:
            self.df.loc[self.df['N_COUR'].str.contains(word), 'I_TRAN_TP_NAME'] = '맞춤연수'
        
        print("shape of the dataframe is ", self.df.shape)
    
    # 사원기관 비사원기관 분류.
    def distinguish사원(self, 사원비사원file = '사원비사원unique.xlsx'):
        
        df_사원비사원 = pd.read_excel(사원비사원file, encoding = "ms949")
        
        비사원기관List = np.unique(df_사원비사원[df_사원비사원['I_DMND_NAME']=='비사원기관'].N_CSTM).tolist()
        사원기관List = np.unique(df_사원비사원[df_사원비사원['I_DMND_NAME']=='사원기관'].N_CSTM).tolist()

        self.df.loc[:,'I_DMND_INST'] = '비사원기관'
        self.df.loc[self.df['N_CSTM'].isin(사원기관List), 'I_DMND_INST'] = '사원기관'
        self.df.loc[self.df['N_CSTM'].str.contains('개인고객', na=False), 'I_DMND_INST'] = '개인'
        self.df.loc[self.df.N_CSTM.isnull(), 'I_DMND_INST'] = np.nan
        
        print("shape of the dataframe is ", self.df.shape)
    
    # 위 함수들을 한번에 작동시키는 함수.(위에 항목들은 무조건적으로 이행해야하는 전처리 작업으로 판단)
    def preprocessMust(self, year1, year2, 사원비사원file = '사원비사원unique.xlsx'):
        
        self.defineYearRange(year1, year2)
        self.removeTestCoursesID()
        self.stripCourseNames()
        self.맞춤재분류()
        self.distinguish사원(사원비사원file)
        
        print("shape of the dataframe is ", self.df.shape)
    
    # 집합,통신,사이버를 제외한 기타에 해당하는 연수들 기타로 재분류. 
    def make집통사기타(self):
        
        기타List = ['자체연수', '일반(기타)연수', '자격보수교육', '자격실무교육', '해외연수']
        self.df.loc[self.df.I_TRAN_NAME.isin(기타List), 'I_TRAN_NAME'] = '기타'
    
    # 분석에 불필요할 수 있는 재취업 과 세미나 수업들 제외.
    def remove재취업세미나(self):
        
        self.df = self.df[self.df.I_TRAN_TP_NAME!='재취업']
        self.df = self.df[self.df.I_TRAN_TP_NAME!='세미나']
    
    # 맞춤연수 제외시키는 함수.(맞춤연수를 제외하고 분석하고 싶을때) 
    def remove맞춤(self):
        
        self.df = self.df[self.df.I_TRAN_TP_NAME!='맞춤연수'] 
        
        print("shape of the dataframe is ", self.df.shape)
        
    
    # 법정자격보수, 보수교육, '보수'가 과정명에 포함된 과정들 제외. 
    def 보수교육제외(self, 법정자격보수file = '법정자격보수.xlsx'):
        
        법정자격보수 = pd.read_excel(법정자격보수file, encoding = "ms949")
        법정자격보수List = 법정자격보수.N_COUR.to_list()
    
        df_not자격보수 = self.df[self.df.I_TRAN_NAME != '자격보수교육']
        df_no보수과정명 = df_not자격보수[~df_not자격보수.N_COUR.str.contains('보수')]
        df_보수모두제외 = df_no보수과정명[~df_no보수과정명['N_COUR'].isin(법정자격보수List)]
        
        self.df = df_보수모두제외
        
        print("shape of the dataframe is ", self.df.shape)   
    

##########################################################################################################################################        
################################# 협업필터링에 필요한 전처리작업들. ######################################################################
################################# 과정명에 대한 수정 및 전처리가 이루어진다. #############################################################
##########################################################################################################################################

    # 정규연수만 남김.
    def remain정규연수(self):
        self.df = self.df[self.df['I_TRAN_TP_NAME']=='정규연수']
        print("shape of the dataframe is ", self.df.shape)
    
    # 미수료와 퇴교 과정들 제외.
    def remove미수료퇴교(self):
        self.df = self.df[self.df['I_EXE_ST']!='미수료']
        self.df = self.df[self.df['I_EXE_ST']!='퇴교']
        print("shape of the dataframe is ", self.df.shape)
    
    #집합,통신, 사이버만 남김. 
    def remain집합통신사이버(self):
        self.df = self.df[self.df['I_TRAN_NAME']!='일반(기타)연수']
        self.df = self.df[self.df['I_TRAN_NAME']!='자격실무교육']
        self.df = self.df[self.df['I_TRAN_NAME']!='해외연수']
        print("shape of the dataframe is ", self.df.shape)
        
    # 접두사 접미사 제외하기.
    def removePrefixSuffix(self):
        
        print("Number of unique coursename before removing prefix and suffix: ", len(self.df.N_COUR.unique()))
        
        #(1) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(1\)","")
        #(비) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(비\)","")
        #(부산), (광주) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(부산\)","")
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(광주\)","")
        #(금토) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(금토\) ","")
        #(B) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(B\)","")
        #(구), (통), (사) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(구\)","")
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(통\)","")
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(사\)","")
        #(부제:..) 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(부제.*\)","")
        #주말 제거
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(주말\) ","")
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"주말 ","")
        self.df['N_COUR'] = self.df['N_COUR'].str.replace(r"\(주말\)","")

        self.df['N_COUR'] = self.df['N_COUR'].str.rstrip()
        self.df['N_COUR'] = self.df['N_COUR'].str.lstrip()
        
        print("Number of unique coursename after removing prefix and suffix: ", len(self.df.N_COUR.unique()))
        
    # 이름이 바뀐 과정들을 현재 과정명으로 통일 시키는 함수.
    def convertCourseNames(self, nameChangeFile = '과정변경이력.xlsx'):
        
        df_namechange = pd.read_excel(nameChangeFile, encoding = 'ms949')
        
        #exception 신용분석기초 & 집합연수
        self.df.loc[((self.df['I_TRAN_NAME']=='집합연수') & (self.df['N_COUR']=='신용분석기초')), 'N_COUR'] = '회계원리와 재무제표 작성'
        
        courseNameChanges_2 = df_namechange.set_index('N_COUR')
        courseNameChanges_2 = courseNameChanges_2.loc[~courseNameChanges_2.index.duplicated(keep='first')]
        
        oldCourseName = courseNameChanges_2.index
        oldNameList = oldCourseName.tolist()

        uniqueNameList_2019 = courseNameChanges_2.UNIQUE_NAME.tolist()

        namesDict = dict(zip(oldNameList, uniqueNameList_2019))

        self.df['N_COUR'] = self.df['N_COUR'].replace(namesDict)   
        
    # 미수료 및 퇴교 제거후 이루어져야하는 function.      
    def remove수료재수강(self):
        self.df = self.df.drop_duplicates(subset=['O_REG', 'N_COUR'], keep = 'first')
        print("shape of the dataframe is ", self.df.shape)
        
    # 5개 이상의 수업을 들은 수강생들만 남김.
    def removeLessThanFive(self):
        self.df = self.df.groupby('O_REG').filter(lambda x: len(x) > 4)
        print("shape of the dataframe is ", self.df.shape)
        
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
         
    # pandas 데이터프레임을 return하는 함수.
    def returndf(self):
        
        print("shape of the dataframe is ", self.df.shape)
        return self.df
        


# In[71]:


class DataPreprocess자격:
    
    # 연도변수를 string에서 int로 변환.
    def __init__(self, df):
        self.df = df
        self.df['D_YY'] = df['D_YY'].astype('int64')
        print("shape of the dataframe is ", self.df.shape)
    
    # 고객번호가 null값인 데이터 제외.
    def removeNullOREG(self):
        self.df = self.df[~self.df.O_REG.isnull()]
        print("shape of the dataframe is ", self.df.shape)
    
    # pandas 데이터프레임을 return하는 함수.
    def returndf(self):
        print("shape of the dataframe is ", self.df.shape)
        return self.df


# In[72]:


class DataPreprocess도서:
    
    # 주문일자를 연도와 월로 나누어서 새로운 열을 만듬.
    def __init__(self, df):
        self.df = df
        self.df.D_ACPT = pd.to_datetime(self.df.D_ACPT)
        self.df['Year'] = self.df.D_ACPT.dt.strftime('%Y').astype('int64')
        self.df['Month'] = self.df.D_ACPT.dt.strftime('%m').astype('int64')
        print("shape of the dataframe is ", self.df.shape)
    
    # 도서코드가 null값인 데이터 제외.
    def removeNullibook(self):
        self.df = self.df[~self.df.I_BOOK.isnull()]
        print("shape of the dataframe is ", self.df.shape)
        
    # 도서이름이 null값인 데이터 제외.
    def removeNullnbook(self):
        self.df = self.df[~self.df.N_BOOK.isnull()]
        print("shape of the dataframe is ", self.df.shape)
        
    # 구매자 id가 null값인 데이터 제외.
    def removeNullid(self):
        self.df = self.df[~self.df.N_PURC_ID.isnull()]
        print("shape of the dataframe is ", self.df.shape)
    
    # pandas 데이터프레임을 return하는 함수.
    def returndf(self):
        print("shape of the dataframe is ", self.df.shape)
        return self.df


# In[9]:


df_과정_processed = DataPreprocess과정(df_과정_raw)
df_과정_processed.preprocessMust(2015,2018)
df_과정_processed.보수교육제외()

# df_과정_보수제외 = df_과정_processed.returndf()


# In[10]:


df_과정_processed.remain정규연수()
df_과정_processed.remove미수료퇴교()
df_과정_processed.remain집합통신사이버()
df_과정_processed.removePrefixSuffix()
df_과정_processed.convertCourseNames()
df_과정_processed.remove수료재수강()
df_과정_processed.removeLessThanFive()


# In[11]:


df_CF = df_과정_processed.returndf()


# In[129]:


df_도서_processed = DataPreprocess도서(df_도서_raw)
df_도서_processed.removeNullibook()
df_도서_processed.removeNullnbook()
df_도서_processed.removeNullid()
df_도서 = df_도서_processed.returndf()


# In[130]:


df_자격_processed = DataPreprocess자격(df_자격_raw)
df_자격_processed.removeNullOREG()
df_자격 = df_자격_processed.returndf()


# In[154]:


class DataAnalysis:
    
    def __init__(self, df, topic):
        
        self.df = df
        self.topic = topic
        print(self.df.shape)
        print("column names(features) are ", self.df.columns.values)
        
        if(self.topic == '과정'):
            self.yearColName = 'D_OPN_YY'
            self.uniqueIDCol = 'O_REG'
            self.uniqueContentCol = 'N_COUR'
        elif(self.topic == '자격'):
            self.yearColName = 'D_YY'
            self.uniqueIDCol = 'O_REG'
            self.uniqueContentCol = 'I_QLFN'
        elif(self.topic == '도서'):
            self.yearColName = 'Year'
            self.uniqueIDCol = 'N_PURC_ID'
            self.uniqueContentCol = 'I_BOOK'
    
    # 선택하는 항목의 항목갯수
    def countVarInCategory(self, feature):
        
        return len(self.df[feature].unique())
    
    # 선택하는 항목별 인원수
    def countValueByFeature(self, feature):
        
        return self.df[feature].value_counts(dropna=False).to_frame()
    
    def groupbyFeatures(self, featureList):
        
        return self.df.groupby(featureList).count().apply(max,axis=1).to_frame()
    
    # 각 연도별 인원수(절대적인 횟수), uniqueID수(개인 수, 주문번호), Content수(과정, 자격, 도서)
    def getAnnualTrend(self):
            
        dfyearly = pd.DataFrame(self.df.groupby(self.yearColName).count().apply(max,axis=1))
        yearList = dfyearly.index.tolist()

        numUniqueIDList = []
        for year in yearList:
            numUniqueIDList.append(len(self.df[self.df[self.yearColName]==year][self.uniqueIDCol].unique()))
        numUniqueCourseList = []
        for year in yearList:
            numUniqueCourseList.append(len(self.df[self.df[self.yearColName]==year][self.uniqueContentCol].unique()))

        dfyearly['uniqueID'] = numUniqueIDList
        dfyearly['Content'] = numUniqueCourseList

        dfyearly.loc['Sum'] = dfyearly.sum(axis=0).values.tolist()
        dfyearly.loc['Sum'][['uniqueID' ,'Content']] = 0

        return dfyearly
    
    # 선택하는 항목(feature)별 인원수(절대적인 횟수), uniqueID수(개인 수, 주문번호), Content수(과정, 자격, 도서) 
    def getdistributionbyfeature(self, feature):
        
        Ser = self.df[feature].value_counts()
        featurenameList = Ser.index.tolist()

        numUniqueIDList = []
        for feat in featurenameList:
            numUniqueIDList.append(len(self.df[self.df[feature]==feat][self.uniqueIDCol].unique()))

        numUniqueCourseList = []
        for feat in featurenameList:
            numUniqueCourseList.append(len(self.df[self.df[feature]==feat][self.uniqueContentCol].unique()))

        df_분포분석 = pd.DataFrame(Ser)
        df_분포분석['uniqueID'] = numUniqueIDList
        df_분포분석['Content'] = numUniqueCourseList
        df_분포분석.loc['Sum'] = df_분포분석.sum(axis=0).values.tolist()

        df_분포분석.loc['Sum'][['uniqueID' ,'Content']] = 0

        return df_분포분석
    
    # 각 연도별 & 선택하는 항목(feature)별 인원수(절대적인 횟수), uniqueID수(개인 수, 주문번호), Content수(과정, 자격, 도서) 
    def getAnnualTrendByFeature(self, feature):
        
        np.set_printoptions(suppress=True)

        yearsList = self.df[self.yearColName].unique().tolist()
        yearsList.sort()
        featureList = self.df[feature].unique().tolist()

        numYears = len(yearsList)

        for j in range(len(featureList)):
            Arr = np.empty([numYears,3])
            for i in range(len(yearsList)):
                year = yearsList[i]
                df_year = self.df[self.df[self.yearColName] == year]
                numOREG = len(df_year[df_year[feature] == featureList[j]][self.uniqueIDCol])
                numUniqueID = len(df_year[df_year[feature] == featureList[j]][self.uniqueIDCol].unique())
                numUniqueCourse = len(df_year[df_year[feature] == featureList[j]][self.uniqueContentCol].unique())
                numList = [numOREG, numUniqueID, numUniqueCourse]
                Arr[i] = numList
            if (j==0):
                Matrix = Arr
            else:
                Matrix = np.concatenate((Matrix, Arr), axis=1)


        columns = pd.MultiIndex.from_product([featureList, ['인원수', 'uniqueID', 'Content']], names=['형태', '연도'])
        index = yearsList

        df_divanalyze = pd.DataFrame(Matrix, index=index, columns=columns)
        df_divanalyze = df_divanalyze.astype('int64')
        df_divanalyze.loc['Sum'] = df_divanalyze.sum(axis=0).values.tolist()

        return df_divanalyze
    
    # 각 연도별 전체 개인(도서는 각 주문번호)의 평균값, 중간값, 최대값을 구하는 함수.
    def getmeanmedianmax(self):
        
        df_feature = self.df.groupby([self.yearColName, self.uniqueIDCol]).count()[[self.uniqueContentCol]]
        df_mean = df_feature.groupby(level=[0]).apply(np.mean)
        df_median = df_feature.groupby(level=[0]).apply(np.median).to_frame()
        df_max = df_feature.groupby(level=[0]).apply(max)

        df_meanmedianmax = pd.concat([df_mean, df_median, df_max], axis =1)
        df_meanmedianmax.columns = ['mean', 'median', 'max']

        return df_meanmedianmax
    
    # 각 연도별 선택하는 항목(feature)의 평균값, 중간값, 최대값을 구하는 함수.  
    def getmeanmedianmaxfeature(self, feature):
        
        df_feature = self.df.groupby([self.yearColName, feature,self.uniqueIDCol]).count()[[self.uniqueContentCol]]
        df_mean = df_feature.groupby(level=[0,1]).apply(np.mean)
        df_median = df_feature.groupby(level=[0,1]).apply(np.median).to_frame()
        df_max = df_feature.groupby(level=[0,1]).apply(max)

        df_meanmedianmax = pd.concat([df_mean, df_median, df_max], axis =1)
        df_meanmedianmax.columns = ['mean', 'median', 'max']

        return df_meanmedianmax
    
    #선택하는 해당 연도(year)에 max값을 가진 고객ID와 max값을 구하는 함수.
    def getMaxCustomersYear(self, year):
        
        dfyear = self.df[self.df[self.yearColName] == year]
        numCourses = dfyear.groupby(self.uniqueIDCol).count()[self.uniqueContentCol].max()
        o_reg = dfyear.groupby(self.uniqueIDCol).count()[self.uniqueContentCol].idxmax()
        print(self.uniqueIDCol,": " , o_reg, "numCourses: ", numCourses)


# In[155]:


도서dataanalysis = DataAnalysis(df_도서, '도서')
자격dataanalysis = DataAnalysis(df_자격, '자격')
과정dataanalysis_보수제외 = DataAnalysis(df_과정_보수제외, '과정')


# In[78]:


class MergedDataAnalysis:
    
    def __init__(self, df_과정, df_자격, df_도서):
        self.과정 = df_과정
        self.자격 = df_자격
        self.도서 = df_도서
        print("dimension of 과정dataframe: ", self.과정.shape)
        print("dimension of 자격dataframe: ", self.자격.shape)
        print("dimension of 도서dataframe: ", self.도서.shape)
        
        #도서에서 oreg가 null값인 데이터 제외.
        self.도서 = self.도서[~self.도서.O_REG.isnull()]
        print("dimension of 도서dataframe(고객번호null값 제외후): ", self.도서.shape)
    
    # 필요한 feature(항목)만 선택.
    def selectFeatures(self, featureList, topic):
        if(topic == '과정'):
            self.과정 = self.과정[featureList]
        if(topic == '자격'):
            self.자격 = self.자격[featureList]
        if(topic == '도서'):
            self.도서 = self.도서[featureList]
    
    #두개의 데이터프레임을 합치는 함수.
    def mergeTwoDataFrames(self, df1, df2, method):
        # method: inner, outer, right, left
        self.mergeddf = pd.merge(df1, df2, on='O_REG', how = method)
        print("shape of the dataframe is ", self.mergeddf.shape)
        return self.mergeddf


# In[79]:


mda = MergedDataAnalysis(df_과정_보수제외, df_자격, df_도서)


# In[80]:


df_merged_과정자격 = mda.mergeTwoDataFrames(mda.과정, mda.자격, 'outer')


# In[81]:


df_merged_all = mda.mergeTwoDataFrames(df_merged_과정자격, mda.도서, 'outer')


# In[20]:


get_ipython().system('pip install sklearn')


# In[12]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


# In[28]:


class CollaborativFiltering:
    
    def __init__(self, df):
        self.df = df
    
    def 사전사후과정정리(self, prereqfilname = '사전_사후_과정.xlsx'):
        coursePreReq = pd.read_excel(prereqfilname, encoding = "ms949")
        coursePreReq = coursePreReq[['후과정', '전과정']]
        coursePreReq_unique = coursePreReq.drop_duplicates(subset=['후과정', '전과정'], keep = 'first')
        self.coursePreReq_Ser = coursePreReq_unique.groupby('후과정')['전과정'].apply(lambda x: x.tolist())

########################## Item Based CF #########################################################################        
        
        
    def calculate_similarity(self, data_items):
        """Calculate the column-wise cosine similarity for a sparse
        matrix. Return a new dataframe matrix with similarities.
        """
        data_sparse = sparse.csr_matrix(data_items)
        similarities = cosine_similarity(data_sparse.transpose())
        sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
        return sim 
    
    def makeRecDictionaryItemBased(self, usersList, df, df_items, df_neighbours, data_matrix, topN):
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

            known_user_list = known_user_likes.tolist()

            # 선후과정 정리
            for i in range(len(self.coursePreReq_Ser.index)):
                if(self.coursePreReq_Ser.index[i] in known_user_list):
                    score = score.drop(self.coursePreReq_Ser[i], errors = 'ignore')

            userLikesDict_itembased_nei[user] = known_user_likes.tolist()
            recommenDict_itembased_nei[user] = score.nlargest(topN).index.tolist()


        dfRec_itembased_nei = pd.DataFrame(recommenDict_itembased_nei).T
        userLikesSeries_nei = pd.Series(userLikesDict_itembased_nei)
        dfRec_itembased_nei['userLikes'] = userLikesSeries_nei

        return dfRec_itembased_nei
    

    def recommendItemBased(self, usersList, neighbour, topN):

        data_mat = pd.get_dummies(self.df.N_COUR).groupby(self.df.O_REG).apply(max)
        data = data_mat.reset_index()
    
        data_items = data_mat.reset_index(drop=True)
    
        #normalize
        magnitude_total = np.sqrt(np.square(data_items).sum(axis=1))
        data_items = data_items.divide(magnitude_total, axis='index')

        data_matrix = self.calculate_similarity(data_items)

        data_neighbours = pd.DataFrame(index=data_matrix.columns, columns=range(1,neighbour+1))
        for i in range(0, len(data_matrix.columns)):
            data_neighbours.iloc[i,:neighbour] = data_matrix.iloc[0:,i].sort_values(ascending=False)[:neighbour].index

        dfRec = self.makeRecDictionaryItemBased(usersList, data, data_items, data_neighbours, data_matrix, topN)
        return dfRec

    
########################## User Based CF #########################################################################        
    
    def calculateUserSim(self, df):
    
        ratings = df.values
        dist_out = 1-sklearn.metrics.pairwise.cosine_distances(ratings)
        user_pred = dist_out.dot(ratings) / np.array([np.abs(dist_out).sum(axis=1)]).T
        user_pred_df = pd.DataFrame(user_pred)
        user_pred_df.columns = df.columns.tolist()

        return user_pred_df

    def makeRecDictionaryUserBased(self, usersList, df, df_items, user_pred, topN):
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
            
            for i in range(len(self.coursePreReq_Ser.index)):
                if(self.coursePreReq_Ser.index[i] in known_user_list):
                    user_pred_df_score = user_pred_df_score.drop(self.coursePreReq_Ser[i], errors = 'ignore')

            recommenDict_userbased[user] = user_pred_df_score.nlargest(topN).index.tolist()
            userLikesDict_userbased[user] = known_user_likes.tolist()

        dfRec_userbased = pd.DataFrame(recommenDict_userbased).T
        userLikesSeries = pd.Series(userLikesDict_userbased)
        dfRec_userbased['userLikes'] = userLikesSeries

        return dfRec_userbased

    def recommendUserBased(self, usersList, topN):

        data_mat = pd.get_dummies(self.df.N_COUR).groupby(self.df.O_REG).apply(max)
        data_items = data_mat.reset_index(drop=True)
        user_pred = self.calculateUserSim(data_mat)
        data = data_mat.reset_index()

        dfRec = self.makeRecDictionaryUserBased(usersList, data, data_items, user_pred, topN)
        return dfRec


# In[29]:


CF_과정 = CollaborativFiltering(df_CF)


# In[30]:


CF_과정.사전사후과정정리()


# In[35]:


userList = ['2012000544990',
'2012000496982',
'2012000308516',
'2012000123374',
'2012000342583',
'2012000215581',
'2012000362323',
'2012000290799',
'2012000463677']


# In[36]:


dfRec_userbased_nei = CF_과정.recommendUserBased(userList)


# In[37]:


dfRec_userbased_nei.to_excel("userbased_top10.xlsx")


# In[13]:


import sklearn

