import streamlit as st
import base64
from PIL import Image

st.title("DOCUMENTATION")
st.markdown("---")

st.header("HOW TO USE DTI PREDICTOR")
st.markdown("---")

st.write(""" welcome to the documentation page of Drug-Target Interaction Predictor. Here, you will find the binding affinity scores 
by giving input of drug candidate (SMILES) and target protein sequence (Amino Acid). This documentation uses various models from DeepPurpose
library to predict the potetial drug canditates that binds with the given protein. 
         """)
st.header("STEPS FOR DTI PREDICTION")
st.markdown("---")

st.header("""USER INTERFACE
            """)

image = Image.open("UI.png")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(image, use_container_width=True, width=300)
    
st.header("**INPUT**")
st.subheader("1- submit a liand")
st.write("You can submit a ligand only by writing a SMILES structure in the allocated box (Drug SMILES String).")

image = Image.open("SMILES.png")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image(image, use_container_width=True, width=300)
st.subheader("2- SUBMIT A TARGET")
st.write("You can submit a target by uploading a PDB file.")

image = Image.open("SEQUENCE.png")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(image, use_container_width=True, width=100)

st.header("PREDICT INTERACTION")
st.write("""Once you have successfully submitted all the parameters required for the prediction, click on predict interaction button.
         It will start the prediction and give you the result within 5 seconds.""")

image = Image.open("RESULT.png")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(image, use_container_width=True, width=300)
    
st.header("VISUALIZATION")
st.write("""After the completion of the prediction, the model will show the structural visualization of drug molecule , target protein and their binding position.""")
st.subheader("1- 2D-STRUCTURE OF DRUG")
image = Image.open("DRUG.png")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(image, use_container_width=True, width=300)
st.subheader("2- 3D-STRUCTURE OF PROTEIN")
image = Image.open("PROTEIN.png")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(image, use_container_width=True, width=300)
st.subheader("3- GENERATED BINDING POSE")
image = Image.open("BINDING_POSE.png")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(image, use_container_width=True, width=300)



