import streamlit as st


st.set_page_config(layout="wide")


# Title of the main page
st.title("Efficient Machine Translation Admin App")

# Description of the main page
st.markdown(  # say here that admin can come anc checn the raters performance and also see the samples that are annotated by them.
    """
    This is a simple app to monitor the annotators performance. 

    The admin can come and check the raters performance and also see the samples that are annotated by them. Also admins can label the samples that are not annotated by the annotator but the model is confident about the eror categories as well as able to select the best translation. This
    will help us to have a better dataset for training the model withouth annotating all the samples.
    """
)
