#스트림 릿 라이브러리
import streamlit as st

# 시각화 패키지
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

#자연어 처리와 관련된 패키지
from wordcloud import WordCloud
from deep_translator import GoogleTranslator

#자연어 처리(전처리) 패키지
from textblob import TextBlob
import neattext as nt
import spacy
from collections import Counter
import re


#텍스트 요약함수
def summarize_text(text, num_sentences=3):
    # 정규표현식으로 알파벳이 아닌 것을 공백으로 치환, 소문자로 변환
    clean_text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    # words라는 리스트에 저장
    words = clean_text.split()
    
    # 공백을 기준으로 분리된 단어(words) -> 단어의 빈도수 계산
    word_freq = Counter(words)
    
    # 정렬(자주 나오는 수가 앞에 오도록)
    sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)
    
    # 상위 n개(많은 수를 기준으로) 단어를 추출
    top_words = sorted_words[:num_sentences]
    
    # 요약문으로 변환
    summary = ' '.join(top_words)
    
    return summary


#텍스트 분석 -> 어간, 표제어 추출 , 토큰화된 단어를 분석
def text_analyzer(text):
    # spacy -> 자연어의 전처리를 수행하는 패키지
    # spacy -> 영어로된 패키지를 사용('en_core_web_sm') -> python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    
    # 텍스트를 nlp객체(영어만 처리)에 넣어서 doc이라는 변수에 결과를 저장
    doc = nlp(text)

    #tokens and lemmas > 토큰화된 단어와 표제어를 추출
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in doc]
    return allData


#
def main():
    subheader = """
    <div style="background-color:rgba(125, 200, 125, 0.8); padding:8px;">
    <h3 style="color:white">텍스트 분석 및 감성분석</h1>
    </div>
    """

    title = "자연어처리프로젝트"
    st.title(title)
    st.markdown(subheader, unsafe_allow_html=True)

    #사이드바를 만들어서/ 사이드바 제목 및 메뉴 선택 적용
    st.sidebar.title("NLP 프로젝트")

    #셀렉트박스의 다양한 결과물을 menu 변수에 저장하여 
    #선택에 따라 sidebar에 text 출력
    menu = st.sidebar.selectbox("메뉴", ["텍스트 분석", "감성 분석", "번역", "이 프로젝트에 대하여"])

    if menu == "텍스트 분석":
        raw_text = st.text_area("TEXT INPUT", 
                             "여기에 영어 텍스트를 입력하세요", height=300)
        #버튼을 누르면 분석을 실행하도록 함
        if st.button("Start"):
            if len(raw_text) == 0:
                st.warning("텍스트가 입력되지 않았습니다.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    #내가 입력한 자연어 대한 기본적인 분석 정보
                    with st.expander("Basic Info for NLP"):
                        word_desc = nt.TextFrame(raw_text).word_stats()
                        result_desc = {"Len of Text :" , word_desc['Length of Text'],
                                    "Num of Vowels :", word_desc['Num of Vowels'],
                                    "Num of Stopwords :", word_desc['Num of Stopwords']}
                        st.write(result_desc)

                with col2:
                    with st.expander("Pre-processed-Text"):
                        processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                        st.write(processed_text)
        
    elif menu == "감성 분석":

        st.subheader("감성 분석")
        raw_text = st.text_area("TEXT INPUT", "여기에 텍스트를 입력하세요", height=300)

        #버튼을 누르면 분석을 실행하도록 함
        if st.button("분석 시작"):
            if len(raw_text) == 0:
                st.warning("텍스트가 입력되지 않았습니다.")
            else:
                blob = TextBlob(raw_text)
                result = blob.sentiment
                st.success(result)

    elif menu == "번역":
        raw_text = st.text_area("TEXT INPUT", "여기에 텍스트를 입력하세요", height=300)

        if len(raw_text) < 3:
            st.warning("텍스트가 너무 짧습니다.")
        else:
            target_language = st.selectbox("번역할 언어를 선택하세요", ["German", 
                                                             "Spanish", "French", 
                                                             "Italian"])
            if target_language == 'German':
                target_lang = 'de'
            elif target_language == 'Spanish':
                target_lang = 'es'
            elif target_language == 'French':
                target_lang = 'fr'
            else:
                target_lang = 'it'

            if st.button("번역 시작"):
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated_text = translator.translate(raw_text)
                st.write(translated_text)




    




if __name__ == '__main__':
    main()