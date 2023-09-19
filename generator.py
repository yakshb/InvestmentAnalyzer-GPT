import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.llms import OpenAI

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
# from langchain.callbacks import StreamlitCallbackHandler
from langchain.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain, SimpleSequentialChain
# from langchain.memory import ConversationBufferMemory


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]


# App frontend and framework
st.set_page_config(
    page_title="Investment Memo Review (Langchain + LlamaIndex)",
    page_icon="https://api.dicebear.com/7.x/icons/svg?seed=Abby&backgroundColor=80cbc4&icon=cashCoin,coin,eyeglasses,newspaper"#,
    #menu_items={"About": "Powered by #Langchain & #LlamaIndex, Bringing OpenAI's prowess to your fingertips to generate detailed memos or company insights in seconds. Input company info & get a detailed investment memo.", "Get help": None, "Report a Bug": None}
)

with st.sidebar:
    st.header("👨‍💻 About the Author")
    st.write("""
    **Yaksh Birla** is a financial professional and tech enthusiast. Driven by passion and a love for leveraging AI tools and sharing knowledge, he created this platform to test new AI tools in the context of Web-enabled investment research and financial analysis.

    Connect with me, or contribute!
    """)

    st.divider()
    st.subheader("🔗 Connect with Me", anchor=False)
    st.markdown(
        """
        - [👔 LinkedIn](https://www.linkedin.com/in/yakshb/)
        - [🐙 Github Profile](https://github.com/yakshb)
        - [🌐 Medium](https://medium.com/@yakshb)
        """
    )

    st.divider()
    st.subheader("🏆 Streamlit Hackathon 2023", anchor=False)
    st.write("This application is Yaksh's entry for the Streamlit Hackathon held in September 2023.")

    st.divider()
    st.write("Made with 🦜️🔗 Langchain and 🦙 LlamaIndex")

st.header('Investment Memo Generator ⚡')
st.subheader('Powered by Langchain and LlamaIndex')

with st.expander("What is this app about?"):
    st.info("""
    Welcome to the AI-powered Investment Research Assistant!

    This application is designed to assist investors, analysts, and finance enthusiasts by generating comprehensive investment research for any company, be it public or private. Here's what you can expect:

    - **Deep Analysis**: The AI dives deep into the web to gather the most recent and relevant information about the company in question.
    - **Comprehensive Reports**: The generated research covers various aspects such as the company's value proposition, business model, market opportunities, executive team, financials, and technology.
    - **Simple Interface**: Input the company's name, and the AI will handle the rest, presenting you with a detailed report in a matter of seconds.

    Whether you're considering a new investment, or just curious about a company, this tool aims to provide you with a thorough understanding, all powered by advanced AI algorithms.
    """, icon="ℹ️")

# col1, col2 = st.columns(2)

# with col1:
company_prompt = st.text_input('Enter the Company Name Here')

# with col2:
#     company_links_input = st.text_input('Enter Relevant Links (i.e. Crunchbase Profile)')

user_notes = st.text_area('Add Your Custom Notes Here')

# Prompt templates
research_template = PromptTemplate(
    input_variables = ['company'],
    template='Evaluate the value proposition, product or service, business model, market opportunity, executive team, financials, and technology-related information about the company, {company}. Keep your answers technical and based on facts – do not hallucinate features.'
)

memo_template = PromptTemplate(
    input_variables = ['company'],
    template='As an experienced investor with deep experience in public equity and venture capital, Write me a detailed investment analysis about the company, {company}. This report should, at a minimum, have the following sections: Summary, Product Evaluation, Market Opportunity, Financials or Unit Economics, Executive Team, Risks'
)

notes_template = PromptTemplate(
    input_variables = ['notes'],
    template='In writing an investment memo, leverage the {notes} provided'
)

# Create Web Agent 
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Intermediate Answer",
        func = search.run,
        description = "useful for when you need to answer questions about latest company research, events or news"
    ),
]

# Initiatlize LLM
llm = OpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.6)
agent_chain = initialize_agent(tools, llm, agent='self-ask-with-search', handle_parsing_errors=True, verbose=True)
research_chain = LLMChain(llm=llm, prompt=research_template)
memo_chain = LLMChain(llm=llm, prompt=memo_template)
notes_chain = LLMChain(llm=llm, prompt=notes_template)

if company_prompt:
    latest_info = agent_chain.run(f"Provide a detailed analysis of the company, {company_prompt}")
    # print(latest_info)
    notes = {"notes": latest_info}
    sequential_chain = SimpleSequentialChain(chains=[research_chain, notes_chain, memo_chain], verbose=True)

# Display LLM answers
if st.button(f'Generate Investment Research'):
    with st.spinner(f'Generating Investment Research for {company_prompt}... This could take 1-2 mins'):
        response = sequential_chain.run(company_prompt)
        st.markdown(response)
        # print(response)
        st.download_button('Download Report', response)