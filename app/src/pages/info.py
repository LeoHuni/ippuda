import streamlit as st

st.text('Hospital Location')
st.map(loc_database.loc[[num_selected],:])
