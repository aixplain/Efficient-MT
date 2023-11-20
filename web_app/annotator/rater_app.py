import streamlit as st


st.set_page_config(layout="wide")


st.title("Efficient Machine Translation Rater App")

st.markdown(  # here describe that annotator can come and annotate the translation of multiple systems with provided error categories from LLM and select the best one and post edit if needed.
    """
    This is a simple app to annotate MT systems. 

    The annotator can come and annotate the translation of multiple systems with provided error categories from LLM and select the best one and post edit if needed.
    As a result, we will have a dataset of annotated translations with error categories and post edits. After each submission the new sample is brought based on the 
    severity model's uncertainity or sample with most predicted severe errors (Regression). You can also select a mized approach where the samples are selected based on
    both of these approaches. 
    """
)
