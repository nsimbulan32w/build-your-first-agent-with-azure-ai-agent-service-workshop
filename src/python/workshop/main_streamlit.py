import asyncio
import logging
import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    Agent,
    AgentThread,
    FunctionTool,
    AsyncToolSet,
    BingGroundingTool,
    CodeInterpreterTool,
    FileSearchTool,
    ToolSet,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from sales_data_sync import SalesData
from stream_event_handler_sync import StreamEventHandler
from terminal_colors import TerminalColors as tc
from utilities_sync import Utilities
import streamlit as st
from streamlit_chat import message

from stream_event_handler_sync import response_stream
import json

from asyncio import run_coroutine_threadsafe
from threading import Thread

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

AGENT_NAME = "Contoso Sales Agent"
TENTS_DATA_SHEET_FILE = "datasheet/contoso-tents-datasheet.pdf"
FONTS_ZIP = "fonts/fonts.zip"
API_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
PROJECT_CONNECTION_STRING = os.environ["PROJECT_CONNECTION_STRING"]
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MAX_COMPLETION_TOKENS = 10240
MAX_PROMPT_TOKENS = 20480
# The LLM is used to generate the SQL queries.
# Set the temperature and top_p low to get more deterministic results.
TEMPERATURE = 0.1
TOP_P = 0.1
INSTRUCTIONS_FILE = None


toolset = ToolSet()
utilities = Utilities()
sales_data = SalesData(utilities)


project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(ExcludeInteractiveBrowserCredential =False ),
    conn_str=PROJECT_CONNECTION_STRING,
)




# Replace with your project endpoint
#"https://<your-project-host-name>.services.ai.azure.com/api/projects/<your-project-name>"

# Replace with your credential (API key or managed identity)
# Example using API key:
# Example using managed identity:
# credential = DefaultAzureCredential()

# Create the AIProjectClient
# project_client = AIProjectClient(
#     project_endpoint=project_endpoint,
#     credential=credential
# )



# Replace with your project endpoint

#"https://<your-project-host-name>.services.ai.azure.com/api/projects/<your-project-name>"

# Replace with your credential (API key or managed identity)
# Example using API key:
# azure_credential = AzureKeyCredential(api_credential)

client_id = "983a8990-5ce9-491f-a705-3b4159359d23"
def_azure_credential = DefaultAzureCredential(managed_identity_client_id=client_id)

# project_client = AIProjectClient.from_connection_string(
#     credential=def_azure_credential,
#     conn_str=PROJECT_CONNECTION_STRING,
# )

# project_client = AIProjectClient( endpoint=project_endpoint, credential=credential, subscription_id="Azure subscription 1d", resource_group_name="rg-nsimbulane2w-0579_ai", project_name="nsimbulane2w-9293")

functions = FunctionTool(
    {
        sales_data.fetch_sales_data_using_sqlite_query,
    }
)

#INSTRUCTIONS_FILE = "instructions/function_calling.txt"
# INSTRUCTIONS_FILE = "instructions/file_search.txt"
INSTRUCTIONS_FILE = "instructions/code_interpreter.txt"
# INSTRUCTIONS_FILE = "instructions/bing_grounding.txt"
# INSTRUCTIONS_FILE = "instructions/code_interpreter_multilingual.txt"


default_prompt = """
You are a **sales analysis agent** for **Contoso**, a retailer of outdoor, camping, and sports gear.\
You help users by answering **sales-related questions**.
"""

system_prompt = st.sidebar.text_area("System Prompt", default_prompt, height=200)
seed_message = {"role": "system", "content": system_prompt}
# endregion

# region SESSION MANAGEMENT
# Initialise session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [seed_message]
if "model_name" not in st.session_state:
    st.session_state["model_name"] = []
if "cost" not in st.session_state:
    st.session_state["cost"] = []
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = []
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0
# endregion

# region SIDEBAR SETUP

counter_placeholder = st.sidebar.empty()
counter_placeholder.write(
    f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
)
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [seed_message]
    st.session_state["number_tokens"] = []
    st.session_state["model_name"] = []
    st.session_state["cost"] = []
    st.session_state["total_cost"] = 0.0
    st.session_state["total_tokens"] = []
    counter_placeholder.write(
        f"Total cost of this conversation: £{st.session_state['total_cost']:.5f}"
    )


download_conversation_button = st.sidebar.download_button(
    "Download Conversation",
    data=json.dumps(st.session_state["messages"]),
    file_name=f"conversation.json",
    mime="text/json",
)

st.title("Streamlit ChatGPT Demo")

# container for chat history
response_container = st.container()
# container for text box
container = st.container()


# endregion



def add_agent_tools() -> None:
    """Add tools for the agent."""
    font_file_info = None

    # Add the functions tool
    toolset.add(functions)

    # Add the tents data sheet to a new vector data store
    # vector_store = utilities.create_vector_store(
    #     project_client,
    #     files=[TENTS_DATA_SHEET_FILE],
    #     vector_store_name="Contoso Product Information Vector Store",
    # )
    # file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    # toolset.add(file_search_tool)

    # Add the code interpreter tool
    code_interpreter = CodeInterpreterTool()
    toolset.add(code_interpreter)

    # Add the Bing grounding tool
    # bing_connection = project_client.connections.get(connection_name=BING_CONNECTION_NAME)
    # bing_grounding = BingGroundingTool(connection_id=bing_connection.id)
    # toolset.add(bing_grounding)

    # Add multilingual support to the code interpreter
    # font_file_info = utilities.upload_file(project_client, utilities.shared_files_path / FONTS_ZIP)
    # code_interpreter.add_file(file_id=font_file_info.id)

    return font_file_info

def initialize() -> tuple[Agent, AgentThread]:
    """Initialize the agent with the sales data schema and instructions."""

    if not INSTRUCTIONS_FILE:
        return None, None

    font_file_info = add_agent_tools()

    sales_data.connect()
    database_schema_string = sales_data.get_database_info()

    try:
        instructions = utilities.load_instructions(INSTRUCTIONS_FILE)
        # Replace the placeholder with the database schema string
        instructions = instructions.replace(
            "{database_schema_string}", database_schema_string)

        if font_file_info:
            # Replace the placeholder with the font file ID
            instructions = instructions.replace(
                "{font_file_id}", font_file_info.id)

        print("Creating agent...")
        agent = project_client.agents.create_agent(
            model=API_DEPLOYMENT_NAME,
            name=AGENT_NAME,
            instructions=instructions,
            toolset=toolset,
            temperature=TEMPERATURE,
            # headers={"x-ms-enable-preview": "true"},
        )
        #print(f"Created agent, ID: {agent.id}")

        project_client.agents.enable_auto_function_calls(toolset=toolset)
        print("Enabled auto function calls.")

        print("Creating thread...")
        thread = project_client.agents.create_thread()
        #print(f"Created thread, ID: {thread.id}")

        return agent, thread

    except Exception as e:
        logger.error("An error occurred initializing the agent: %s", str(e))
        logger.error("Please ensure you've enabled an instructions file.")


def cleanup(agent: Agent, thread: AgentThread) -> None:
    """Cleanup the resources."""
    existing_files = project_client.agents.list_files()
    for f in existing_files.data:
        project_client.agents.delete_file(f.id)
    project_client.agents.delete_thread(thread.id)
    project_client.agents.delete_agent(agent.id)
    sales_data.close()


def post_message(thread_id: str, content: str, agent: Agent, thread: AgentThread) -> None:
    """Post a message to the Foundry Agent Service."""
    try:
        project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=content,
        )

        stream = project_client.agents.create_stream(
            thread_id=thread.id,
            agent_id=agent.id,
            event_handler=StreamEventHandler(
                functions=functions, project_client=project_client, utilities=utilities),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            instructions=agent.instructions,
        )

        with stream as s:
            s.until_done()

    except Exception as e:
        utilities.log_msg_purple(
            f"An error occurred posting the message: {e!s}")


def generate(agent: Agent, thread: AgentThread,  content: str) -> None:
        
    post_message(agent=agent, thread_id=thread.id, content=content, thread=thread)

    response_string = ''.join([str(s) for s in response_stream])

    st.session_state["past"].append(content)
    st.session_state["generated"].append(response_string)
    st.session_state["model_name"].append(API_DEPLOYMENT_NAME)
    # st.session_state["total_tokens"].append(total_tokens)

    # # from https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
    # cost = total_tokens * 0.001625 / 1000

    # st.session_state["cost"].append(cost)
    # st.session_state["total_cost"] += cost

    st.write(response_string)

        #while True:
            # prompt = input(
            #     f"\n\n{tc.GREEN}Enter your query (type exit or save to finish): {tc.RESET}").strip()
            # if not prompt:
            #     continue


        # cmd = prompt.lower()
        # if cmd in {"exit", "save"}:
        #     break         




    # if st.session_state["generated"]:
    #     with response_container:
    #         for i in range(len(st.session_state["generated"])):
    #             message(
    #                 st.session_state["past"][i],
    #                 is_user=True,
    #                 key=str(i) + "_user",
    #                 avatar_style="shapes",
    #             )
    #             message(
    #                 st.session_state["generated"][i], key=str(i), avatar_style="identicon"
    #             )
    #         counter_placeholder.write(
    #             f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
    #         )     

def main() -> None:
#def main() -> None:

    """
    Example questions: Sales by region, top-selling products, total shipping costs by region, show as a pie chart.
    """
    # with project_client:
    agent, thread = initialize()

    #agent, thread = run_async(initialize())
    #agent, thread = asyncio.run(initialize())


    if not agent or not thread:
        print(f"{tc.BG_BRIGHT_RED}Initialization failed. Ensure you have uncommented the instructions file for the lab.{tc.RESET}")
        print("Exiting...")
        return
    
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            prompt = st.text_area("You:", key="input", height=100)
            submit_button = st.form_submit_button(label="Send")

        if prompt and submit_button:
            # asyncio.run(generate(st.session_state.agent, st.session_state.thread,prompt))
            #generate(agent, thread,prompt)
            post_message(agent=agent, thread_id=thread.id, content=prompt, thread=thread)

            response_string = ''.join([str(s) for s in response_stream])

            st.session_state["past"].append(prompt)
            st.session_state["generated"].append(response_string)
            st.session_state["model_name"].append(API_DEPLOYMENT_NAME)
            # st.session_state["total_tokens"].append(total_tokens)

            # # from https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
            # cost = total_tokens * 0.001625 / 1000

            # st.session_state["cost"].append(cost)
            # st.session_state["total_cost"] += cost

            st.write(response_string)


    
    
    # if not st.session_state.get('agent'):
    #     with project_client:
    #         agent, thread = initialize()
    #         # st.session_state.agent = agent
    #         # st.session_state.thread = thread

    #         if not st.session_state.agent or not st.session_state.thread:
    #             print(f"{tc.BG_BRIGHT_RED}Initialization failed. Ensure you have uncommented the instructions file for the lab.{tc.RESET}")
    #             print("Exiting...")
    #             return
       


if __name__ == '__main__':
    main()








