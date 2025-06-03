# based on https://github.com/Azure-Samples/azureai-samples/blob/main/scenarios/Assistants/multi-agent/multi-agent.ipynb

import os
import time
# from matplotlib import pyplot as plt
# import cv2
import requests
from PIL import Image
from pathlib import Path
import logging
import json

#import asyncio

from dotenv import load_dotenv
from openai import AzureOpenAI

from openai.types.beta import Thread
from openai.types.beta import Assistant

from sales_data_sync import SalesData
from terminal_colors import TerminalColors as tc
from utilities import Utilities
import pandas as pd

import streamlit as st
from streamlit_chat import message
import pandas as pd
import pysqlite3 as sqlite3

# ENV = dotenv.dotenv_values(".env")

# with st.sidebar.expander("Environment Variables"):
#     st.write(ENV)

# region PROMPT SETUP



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


# Create the AOAI client to use for the proxy agent.
assistant_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # Your API key for the assistant api model
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # API version  (i.e. 2024-02-15-preview)
    azure_endpoint=os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    ),  # Your Azure endpoint (i.e. "https://YOURENDPOINT.openai.azure.com/")
)

# Assistant model should be '1106' or higher
assistant_deployment_name = os.getenv(
    "MODEL_DEPLOYMENT_NAME"
)  # The name of your assistant model deployment in Azure OpenAI (i.e. "GPT4Assistant")

# toolset = ToolSet()

# import sqlite3

# conn = sqlite3.connect("data/Chinook.db")
# print("Opened database successfully")  

def fetch_sales_data_using_sqlite_query(query: str) -> str:
    """
    This function is used to answer user questions about Contoso sales data by executing SQLite queries against the database.

    :param sqlite_query: The input should be a well-formed SQLite query to extract information based on the user's question. The query result will be returned as a JSON object.
    :return: Return data in JSON serializable format.
    :rtype: str
    """

    print(
        f"\n{tc.BLUE}Function Call Tools: fetch_sales_data_using_sqlite_query{tc.RESET}\n")
    print(f"{tc.BLUE}Executing query: {query}{tc.RESET}\n")


    try:
        # Perform the query asynchronously
        cursor = conn.execute(query)
        # with conn.execute(sqlite_query) as cursor:
        # with cursor:

        rows = cursor.fetchall()
        columns = [description[0]
                    for description in cursor.description]

        if not rows:  # No need to create DataFrame if there are no rows
            return json.dumps("The query returned no results. Try a different question.")
        data = pd.DataFrame(rows, columns=columns)
        return data.to_json(index=False, orient="split")

    except Exception as e:
        return json.dumps({"SQLite query failed with error": str(e), "query": query})
    
  
# # name of the model deployment for DALL·E 3
# dalle_client = AzureOpenAI(
#     api_key=os.getenv("DALLE3_AZURE_OPENAI_KEY"),
#     api_version=os.getenv("DALLE3_AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("DALLE3_AZURE_OPENAI_ENDPOINT"),
# )
# dalle_deployment_name = os.getenv("DALLE3_DEPLOYMENT_NAME")

# # name of the model deployment for GPT 4 with Vision
# vision_client = AzureOpenAI(
#     api_key=os.getenv("GPT4VISION_AZURE_OPENAI_KEY"),
#     api_version=os.getenv("GPT4VISION_AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("GPT4VISION_AZURE_OPENAI_ENDPOINT"),
# )
# vision_deployment_name = os.getenv("GPT4VISION_DEPLOYMENT_NAME")


# name_dl = "dalle_assistant"
# instructions_dl = """As a premier AI specializing in image generation, you possess the expertise to craft precise visuals based on given prompts. It is essential that you diligently generate the requested image, ensuring its accuracy and alignment with the user's specifications, prior to delivering a response."""
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "generate_image",
#             "description": "Creates and displays an image",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "prompt": {
#                         "type": "string",
#                         "description": "The prompt to be used to create the image",
#                     }
#                 },
#                 "required": ["prompt"],
#             },
#         },
#     }
# ]

# verbose_output = True


# dalle_assistant = assistant_client.beta.assistants.create(
#     name=name_dl, instructions=instructions_dl, model=assistant_deployment_name, tools=tools
# )

# def generate_image(prompt: str) -> str:
#     """
#     Call the Azure OpenAI Dall-e 3 model to generate an image from a text prompt.
#     Executes the call to the Azure OpenAI Dall-e 3 image creator, saves the file into the local directory, and displays the image.
#     """

#     print("Dalle Assistant Message: Creating the image ...")

#     response = dalle_client.images.generate(
#         model=dalle_deployment_name, prompt=prompt, size="1024x1024", quality="standard", n=1
#     )

#     # Retrieve the image URL from the response (assuming response structure)
#     image_url = response.data[0].url

#     # Open the image from the URL and save it to a temporary file.
#     im = Image.open(requests.get(image_url, stream=True).raw)

#     # Define the filename and path where the image should be saved.
#     filename = "temp.jpg"
#     local_path = Path(filename)

#     # Save the image.
#     im.save(local_path)

#     # Get the absolute path of the saved image.
#     full_path = str(local_path.absolute())

#     img = cv2.imread("temp.jpg", cv2.IMREAD_UNCHANGED)

#     # Convert the image from BGR to RGB for displaying with matplotlib,
#     # because OpenCV uses BGR by default and matplotlib expects RGB.
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Display the image with matplotlib.
#     plt.imshow(img_rgb)
#     plt.axis("off")  # Turn off axis labels.
#     plt.show()

#     # Return the full path of the saved image.
#     print("Dalle Assistant Message: " + full_path)
#     return "Image generated successfully and store in the local file system. You can now use this image to analyze it with the vision_assistant"


# name_vs = "vision_assistant"
# instructions_vs = """As a leading AI expert in image analysis, you excel at scrutinizing and offering critiques to refine and improve images. Your task is to thoroughly analyze an image, ensuring that all essential assessments are completed with precision before you provide feedback to the user. You have access to the local file system where the image is stored."""
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "analyze_image",
#             "description": "analyzes and critics an image",
#             "parameters": {"type": "object", "properties": {}, "required": []},
#         },
#     }
# ]

# verbose_output = True


# vision_assistant = assistant_client.beta.assistants.create(
#     name=name_vs, instructions=instructions_vs, model=assistant_deployment_name, tools=tools
# )


# def analyze_image() -> str:
#     """
#     Call the Azure OpenAI GPT4 Vision model to analyze and critic an image and return the result.The resulting output should be a new prompt for dall-e that enhances the image based on the criticism and analysis
#     """
#     print("Vision Assistant Message: " + "Analyzing the image...")

#     import base64
#     from pathlib import Path

#     # Create a Path object for the image file
#     image_path = Path("temp.jpg")

#     # Using a context manager to open the file with Path.open()
#     with image_path.open("rb") as image_file:
#         base64_image = base64.b64encode(image_file.read()).decode("utf-8")

#     content_images = [
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
#         for base64_image in [base64_image]
#     ]
#     response = vision_client.chat.completions.create(
#         model=vision_deployment_name,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Analyze and critic this image and generate a new enhanced prompt for Dall-e with the criticism and analysis.",
#                     },
#                     *content_images,
#                 ],
#             }
#         ],
#         max_tokens=1000,
#     )
#     print("Vision Assistant Message: " + response.choices[0].message.content)
#     return response.choices[0].message.content


    

utilities = Utilities()
sales_data = SalesData(utilities)

name_pa = AGENT_NAME

# toolset = AsyncToolSet()

# functions = AsyncFunctionTool(
#     {
#         sales_data.async_fetch_sales_data_using_sqlite_query,
#     }
# )

#INSTRUCTIONS_FILE = "instructions/function_calling.txt"
#INSTRUCTIONS_FILE = "instructions/file_search.txt"
INSTRUCTIONS_FILE = "instructions/code_interpreter.txt"
# INSTRUCTIONS_FILE = "instructions/bing_grounding.txt"
# INSTRUCTIONS_FILE = "instructions/code_interpreter_multilingual.txt"




# agent_arr = ["dalle_assistant", "vision_assistant"]
# agent_string = ""
# for item in agent_arr:
#     agent_string += f"{item}\n"

# instructions_pa = f"""As a user proxy agent, your primary function is to streamline dialogue between the user and the specialized agents within this group chat. You are tasked with articulating user inquiries with clarity to the relevant agents and maintaining a steady flow of communication to guarantee the user's request is comprehensively addressed. Please withhold your response to the user until the task is completed, unless an issue is flagged by the respective agent or when you can provide a conclusive reply.

# You have access to the local file system where files are stores. For example, you can access the image generated by the Dall-e assistant and send it to the Vision assistant for analysis.

# You have access to the following agents to accomplish the task:
# {agent_string}
# If the agents above are not enough or are out of scope to complete the task, then run send_message with the name of the agent.

# When outputting the agent names, use them as the basis of the agent_name in the send message function, even if the agent doesn't exist yet.

# Run the send_message function for each agent name generated. 

# Do not ask for followup questions, run the send_message function according to your initial input.

# Plan:
# 1. dalle_assistant creates image 
# 2. vision_assistant analyzes images and creates a new prompt for dalle_assistant
# 3. dalle_assistant creates a new image based on the new prompt
# 4. vision_assistant analyzes images and creates a new prompt for dalle_assistant
# 5. dalle_assistant creates a new image based on the new prompt

# Now take a deep breath and accomplish the plan above. Always follow the plan step by step in the exact order and do not ask for followup questions. Do not skip any steps in the plan, do not repeat any steps and always complete the entire plan in order step by step.  
# The dall-e assistant can never run more than one time in a row, review your plan before running the next step.
# """


# database_schema_string = ""
# database_schema_string = sales_data.get_database_info()




DATA_BASE = "database/contoso-sales.db"

def db_connect():
    db_uri = f"file:{utilities.shared_files_path}/{DATA_BASE}?mode=ro"

    try:
        conn = sqlite3.connect(db_uri, uri=True)
        logger.debug("Database connection opened.")
        return conn
    except sqlite3.Error as e:
        logger.exception("An error occurred", exc_info=e)
        conn = None

def get_table_names(conn):
    """Return a list of table names."""
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';") 
    table_names = [table[0] for table in tables if table[0] != "sqlite_sequence"] 
    return table_names   


def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

conn = db_connect()
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)


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

# endregion


st.title("Streamlit ChatGPT Demo")

# container for chat history
response_container = st.container()
# container for text box
container = st.container()





# async def initialize() -> tuple[Assistant, Thread]:
def initialize() -> tuple[Assistant, Thread]:

    """Initialize the agent with the sales data schema and instructions."""

    if not INSTRUCTIONS_FILE:
        return None, None

    font_file_info = add_agent_tools()


    tools = [
            {   "type": "code_interpreter"},
            {
                "type": "function",
                "function": {
                    "name": "fetch_sales_data_using_sqlite_query",
                    "description": "This function is used to answer user questions about Contoso sales data by executing SQLite queries against the database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": f"""
                                        SQL query extracting info to answer the user's question.
                                        SQL should be written using this database schema:
                                        {database_schema_string}
                                        The query should be returned in plain text, not in JSON.
                                        """,
                            }
                        },
                        "required": ["query"],
                    },
                }
            }
        ]

    # tools = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "ask_database",
    #         "description": "Use this function to answer user questions about music. Input should be a fully formed SQL query.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": f"""
    #                             SQL query extracting info to answer the user's question.
    #                             SQL should be written using this database schema:
    #                             {database_schema_string}
    #                             The query should be returned in plain text, not in JSON.
    #                             """,
    #                 }
    #             },
    #             "required": ["query"],
    #         },
    #     }
    # }
    # ]    
    verbose_output = True


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
        agent = assistant_client.beta.assistants.create(
            name=name_pa, instructions=instructions, model=assistant_deployment_name, 
            tools=tools 
            # tools=toolset
        )       

#         messages = [{
#             "role":"user", 
#             "content": "What is the best selling product?"
#         }]

#         response = assistant_client.chat.completions.create(
#             model=assistant_deployment_name, 
#             messages=messages, 
#             tools= tools, 
#             tool_choice="auto"
# )

        print(f"Created agent, ID: {agent.id}")

        # project_client.agents.enable_auto_function_calls(toolset=toolset)
        # print("Enabled auto function calls.")

        print("Creating thread...")
        thread = assistant_client.beta.threads.create()

        print(f"Created thread, ID: {thread.id}")

        return agent, thread
    
    except Exception as e:
        logger.error("An error occurred initializing the agent: %s", str(e))
        logger.error("Please ensure you've enabled an instructions file.")



def add_agent_tools() -> None:
    """Add tools for the agent."""
    font_file_info = None

    # Add the functions tool
    # toolset.add(functions)

    # Add the tents data sheet to a new vector data store
    # vector_store = await utilities.create_vector_store(
    #     project_client,
    #     files=[TENTS_DATA_SHEET_FILE],
    #     vector_store_name="Contoso Product Information Vector Store",
    # )
    # file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    # toolset.add(file_search_tool)

    # Add the code interpreter tool
    # code_interpreter = CodeInterpreterTool()
    # toolset.add(code_interpreter)

    # Add the Bing grounding tool
    # bing_connection = await project_client.connections.get(connection_name=BING_CONNECTION_NAME)
    # bing_grounding = BingGroundingTool(connection_id=bing_connection.id)
    # toolset.add(bing_grounding)

    # Add multilingual support to the code interpreter
    # font_file_info = await utilities.upload_file(project_client, utilities.shared_files_path / FONTS_ZIP)
    # code_interpreter.add_file(file_id=font_file_info.id)

    return font_file_info




# from typing import Dict, Optional

# agents_threads: Dict[str, Dict[str, Optional[str]]] = {
#     "dalle_assistant": {"agent": dalle_assistant, "thread": None},
#     "vision_assistant": {"agent": vision_assistant, "thread": None},
# }


# # Define the send_message function with only the query parameter
# def send_message(query: str, agent_name: str) -> str:
#     # Check if the agent_name is in agents_threads
#     if agent_name not in agent_arr:
#         print(
#             f"Agent '{agent_name}' does not exist. This means that the multi-agent system does not have the necessary agent to execute the task. *** FUTURE CODE: AGENT SWARM***"
#         )
#         # return None
#     # If the program has not exited, proceed with setting the agent recipient
#     recipient_type = agent_name
#     recipient_info = agents_threads[recipient_type]

#     # If the program has not exited, proceed with setting the agent recipient
#     recipient_type = agent_name
#     recipient_info = agents_threads[recipient_type]

#     # Create a new thread if user proxy and agent thread does not exist
#     if not recipient_info["thread"]:
#         thread_object = assistant_client.beta.threads.create()
#         recipient_info["thread"] = thread_object

#     # This function dispatches a message to the proper agent and it's thread
#     return dispatch_message(query, recipient_info["agent"], recipient_info["thread"])
#     # print("Proxy Assistant Message: " + message)

import json


def dispatch_message(message: str, agent: Assistant, thread: Thread) -> str:
    # Loops through all the agents functions to determine which function to use

    available_functions = {}
    # Iterate through each tool in the agent.tools list
    for tool in agent.tools:
        # Check if the tool has a 'function' attribute
        if hasattr(tool, "function"):
            function_name = tool.function.name
            # Attempt to retrieve the function by its name and add it to the available_functions dictionary
            if function_name in globals():
                available_functions[function_name] = globals()[function_name]
        else:
            # Handle the case where the tool does not have a 'function' attribute
            print("This tool does not have a 'function' attribute.")
    # Draft a new message as part of the ongoing conversation.
    message = assistant_client.beta.threads.messages.create(thread_id=thread.id, role="user", content=message)
    # Carry out the tasks delineated in this discussion thread.
    run = assistant_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
    )
    while True:
        # Await the completion of the thread execution.
        while run.status in ["queued", "in_progress"]:
            run = assistant_client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(1)

        # If an action is necessary, initiate the appropriate function to perform the task.
        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            for _tool_call in tool_calls:
                tool_responses = []
                if (
                    run.required_action.type == "submit_tool_outputs"
                    and run.required_action.submit_tool_outputs.tool_calls is not None
                ):
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    for call in tool_calls:
                        if call.type == "function":
                            if call.function.name not in available_functions:
                                raise Exception("Function requested by the model does not exist")

                            # Assign the appropriate function to the agent for invocation.
                            function_to_call = available_functions[call.function.name]
                            tool_response = function_to_call(**json.loads(call.function.arguments))
                            tool_responses.append({"tool_call_id": call.id, "output": tool_response})

            # Present the outcomes produced by the tool.
            run = assistant_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id, run_id=run.id, tool_outputs=tool_responses
            )

        # if the run is completed, return the assistant message else provide error
        elif run.status == "failed":
            raise Exception("Run Failed. ", run.last_error)
        # Craft a reply from the assistant.
        else:
            messages = assistant_client.beta.threads.messages.list(thread_id=thread.id)

            # Transmit the response message back to the facilitating agent.
            return messages.data[0].content[0].text.value
        
# def generate_response(prompt):
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     try:
#         completion = openai.ChatCompletion.create(
#             engine=ENV["AZURE_OPENAI_CHATGPT_DEPLOYMENT"],
#             messages=st.session_state["messages"],
#         )
#         response = completion.choices[0].message.content
#     except openai.error.APIError as e:
#         st.write(response)
#         response = f"The API could not handle this content: {str(e)}"
#     st.session_state["messages"].append({"role": "assistant", "content": response})

#     # print(st.session_state['messages'])
#     total_tokens = completion.usage.total_tokens
#     prompt_tokens = completion.usage.prompt_tokens
#     completion_tokens = completion.usage.completion_tokens
#     return response, total_tokens, prompt_tokens, completion_tokens

# Initiate proxy agent and the main thread. This thread will remain active until the task is completed and will serve as the main communication thread between the other agents.
# async def main() -> None:
def main() -> None:


    agent, thread = initialize()
    if not agent or not thread:
        print(f"{tc.BG_BRIGHT_RED}Initialization failed. Ensure you have uncommented the instructions file for the lab.{tc.RESET}")
        print("Exiting...")
        return

    cmd = None


    # while True:

    #     user_message = input("Enter your query (type exit or save to finish): ")

    #     if not user_message:
    #         continue

    #     cmd = user_message.lower()
    #     if cmd in {"exit", "save"}:
    #         break





    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area("You:", key="input", height=100)
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:

            message = dispatch_message(user_input, agent, thread)    

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(message)
            st.session_state["model_name"].append(assistant_deployment_name)
            # st.session_state["total_tokens"].append(total_tokens)

            # # from https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
            # cost = total_tokens * 0.001625 / 1000

            # st.session_state["cost"].append(cost)
            # st.session_state["total_cost"] += cost

            st.write(message)


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
    #     # print('Here is the output')

        # print(message)




    # response = assistant_client.beta.assistants.delete(agent.id)
    # response = assistant_client.beta.threads.delete(thread.id)

    # assistant_client.close()
    # assistant_client.delete()

# response = assistant_client.beta.assistants.delete(dalle_assistant.id)
# response = assistant_client.beta.assistants.delete(vision_assistant.id)


if __name__ == "__main__":
    # print("Starting async program...")
    # asyncio.run(main())
    print("Starting program...")

    main()

    print("Program finished.")