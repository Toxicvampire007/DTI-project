import streamlit as st

st.title("📚 Frequently Asked Questions (FAQ) - DTI Prediction")

st.markdown("Welcome to the FAQ section. Here you’ll find answers to common questions about how our Drug-Target Interaction prediction system works.")

# FAQ 1
with st.expander("🔬 What is Drug-Target Interaction (DTI) prediction?"):
    st.write("""
    DTI prediction is the computational process of predicting whether a drug molecule will interact with a specific protein target.
    It helps accelerate drug discovery by reducing the need for expensive and time-consuming lab experiments.
    """)

# FAQ 2
with st.expander("🧪 What inputs do I need to make a DTI prediction?"):
    st.write("""
    You need:
    - A **drug** represented as a SMILES string.
    - A **protein** uploaded in PDB format.
    The model processes these formats to extract biochemical features and predict interaction affinity.
    """)

# FAQ 3
with st.expander("🤖 What machine learning model is used?"):
    st.write("""
    Our backend currently uses a DeepPurpose-based framework which combines a **Message Passing Neural Network (MPNN)** for drug encoding and **Convolutional Neural Network (CNN)** for protein encoding.
    This allows the model to learn complex relationships between chemical structure and protein sequence.
    """)

# FAQ 4
with st.expander("📈 What does the output prediction value mean?"):
    st.write("""
    The output is a **binding affinity score**.
    A lower score generally indicates **stronger binding** between the drug and the target, while a higher score suggests **weaker or no interaction**.
    """)

# FAQ 5
with st.expander("🧬 Can I trust the prediction results?"):
    st.write("""
    The predictions are based on trained machine learning models using curated datasets. While they're useful for **early-stage screening**, we always recommend **lab validation** for any critical decision-making.
    """)

# FAQ 6
with st.expander("🌐 Where does the training data come from?"):
    st.write("""The training data was manually curated by using molecualar docking method using **SwissDock**.The SMILES were retrieve from **DRUGBANK** and the protein sequence were retrieved from **PROTEIN DATA BANK (PDB)**.      
    """)

#FAQ 7
with st.expander("📌 What should I do if I get an invalid input error?"):
    st.write("""
    Double-check:
    - SMILES strings for proper syntax (e.g., balanced brackets, valid atoms).
    - The PDB file containing FASTA sequences for only valid amino acid codes (e.g., A, R, N, D...).
    Use online validators like PubChem or UniProt if unsure.
    """)

# FAQ 8
with st.expander("📥 Can I input multiple drug-target pairs at once?"):
    st.write("""
    Currently, the system supports **one pair at a time** for real-time prediction.
    However, batch processing support is planned for future versions — stay tuned!
    """)

# FAQ 9
with st.expander("🧠 What kinds of deep learning methods are used behind the scenes?"):
    st.write("""
    Our model architecture uses:
    - **MPNN** (Message Passing Neural Network) for capturing chemical graph structures.
    - **CNN** (Convolutional Neural Network) for protein sequence feature extraction.
    These layers learn representations automatically without handcrafted features.
    """)

# FAQ 10
with st.expander("📦 Do you store or log my prediction data?"):
    st.write("""
    No. For privacy and security, we do **not store** any input or output data unless you explicitly choose to download it.
    In training mode, local logs are created only for your private use.
    """)

# FAQ 11
with st.expander("🔐 Are there limitations to the model's prediction power?"):
    st.write("""
    Definitely. Limitations include:
    - Poor generalization for rare/novel proteins or drugs not seen in training.
    - Limited interpretability (deep models are often black boxes).
    - Reliance on data quality and feature extraction tools.
    It's a decision-support tool, not a replacement for lab validation.
    """)

# FAQ 12
with st.expander("📬 Who do I contact for support or questions?"):
    st.write("""
    For any technical issues or project-related queries, feel free to reach out to the developers via 'ahanbiswal2003@gmail.com'.
    """)



