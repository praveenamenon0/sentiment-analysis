from logging import error
import pickle
import streamlit as st 

loaded_model = pickle.load(open('LR movie new.pkl', 'rb'))
loaded_tfidf = pickle.load(open('tf_idf movie.pkl','rb')) 

def UI_Page():
    st.title("Movie Review Sentiment Analysis") 
    review = st.text_input("Enter your Movie Review: ") 
    ok=st.button("predict the class")
    if ok == True:
        if len(review) == 0:
            st.error('Please enter some data')
            return None
        try:
            review = [review] 
            review = loaded_tfidf.transform(review).toarray() 
            result = loaded_model.predict(review) 
            if result[0] == 1:
                st.success("The review is positive") 
            elif result[0] == 0:
                st.error("The review is negative") 
            else:
                st.error("Oops! Something went wrong") 
        except Exception as e:
            st.error('Enter some data: ')            
