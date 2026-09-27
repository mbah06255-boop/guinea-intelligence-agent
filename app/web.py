import os
import streamlit as st

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from agent import Agent

st.set_page_config(page_title="Guinea Intelligence Agent", page_icon="🇬🇳")
st.title("🇬🇳 Guinea Intelligence Agent")

if "user_id" not in st.session_state:
    with st.form("login"):
        username = st.text_input("Ton prénom ou pseudo")
        submitted = st.form_submit_button("Entrer")
    if submitted and username.strip():
        st.session_state.user_id = username.strip()
        st.rerun()
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = Agent(user_id=st.session_state.user_id)
    st.session_state.messages = []

st.caption(f"Connecté en tant que : {st.session_state.user_id}")

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Écris ton message...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            reply = st.session_state.agent.ask(user_input)
        st.markdown(reply)

    st.session_state.messages.append(("assistant", reply))