import streamlit as st

st.title("📘 ABOUT THIS WEBSITE")
st.markdown("---")

st.write("""
This website is designed as an intuitive platform for drug-target interaction (DTI), a powerful web-based tool designed to acclerate drug discovery and development processes.
This tool is prepared using DeepPurpose Framework which enables researchers, scientists and students to predict the binding affinity scores between small molecules
and target protein with ease.
""")

st.header("KEY FEATURES")
st.markdown("---")

st.write("""
-**User-Friendly Input**: Simply enter the 'SMILES' notation of your compound and the 'Amino Acid Sequence' of your target protein.""")
st.write("""
-**Instant Prediction**: The site utilizes advanced deep learning models from the DeepPurpose library to provide rapid and reliable predictions of binding affinity scores.""")
st.write("""
-**No Coding Required**: All computations are handled in the backend, making the tool accessible to users without programming experience.""")
st.write("""
-**Research-Grade Output**: Obtain binding affinity scores that can be used for drug discovery, lead optimization, or academic research.""")
st.write("""
-**Secure and Private**: Your input data is processed securely and is not stored or shared.
""")

st.markdown("---")
st.write("""
**This platform aims to bridge the gap between computational drug discovery and practical application,
making advanced DTI prediction accessible to the broader scientific community.**
""")