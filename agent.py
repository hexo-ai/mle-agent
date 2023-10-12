import os
import time
import openai
import asyncio
import nbformat
import nbconvert
import ipynbname
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython import get_ipython
from ipylab import JupyterFrontEnd

def extract_inputs_outputs_from_html(html_data):
    soup = BeautifulSoup(html_data, 'html.parser')
    code_cells = soup.select('div.jp-Cell.jp-CodeCell.jp-Notebook-cell')
    inputs_outputs = []
    for cell in code_cells:
        input_div = cell.select_one('div.jp-InputArea.jp-Cell-inputArea')
        input_code = input_div.select_one('pre').text if input_div and input_div.select_one('pre') else None
        output_div = cell.select_one('div.jp-OutputArea.jp-Cell-outputArea')
        output_code = output_div.select_one('pre').text if output_div and output_div.select_one('pre') else None
        inputs_outputs.append({
            "input": input_code,
            "output": output_code
        })
    return inputs_outputs

def get_current_notebook_name():
    return ipynbname.name()

def extract_html_from_notebook(notebook_name):
    # Load the notebook
    notebook_filename = f'{notebook_name}.ipynb'
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
        
    # Create an HTML exporter with the 'basic' template
    html_exporter = nbconvert.HTMLExporter()
    html_exporter.template_name = 'basic'
    
    # Convert to HTML
    html_data, resources = html_exporter.from_notebook_node(notebook_content)
    return html_data
  
def extract_html_from_current_notebook():
    notebook_name = get_current_notebook_name()
    return extract_html_from_notebook(notebook_name)


def create_jupyter_app():
    app = JupyterFrontEnd()
    async def init():
        await app.ready()
        cmds = app.commands.list_commands()[:5]
        assert len(cmds) > 0, 'Jupyter Frontend is not ready. Please retry'
    _ = asyncio.create_task(init())
    return app

def get_previous_code_and_output_string(extracted_cells):
    included_cells = [{'input':cell['input'].replace("#Include\n",''), 
                    'output':cell['output']}
                    for cell in extracted_cells if cell['input'].startswith("#Include")]

    included_cells_string = []
    for cell in included_cells:
        included_cells_string.extend(['\nInput code:', str(cell['input']),
                                    'Corresponding output:', str(cell['output'])])
    included_cells_string = " ".join(included_cells_string)
    previous_code_and_output_string = "Here are the code cells that have been executed and their corresponding output:" + included_cells_string
    return previous_code_and_output_string


def inject_code_to_current_notebook(code, execute=False):
    """Inject code into the current Jupyter notebook and optionally execute it."""
    # Use IPython's magic command
    shell = get_ipython()
    # Inject the code into a new cell
    shell.set_next_input(code, replace=False)
    # Optionally execute the code
    if execute:
        exec(code)

class LLM:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def ask_gpt(self, prompt, model="gpt-3.5-turbo"):
        # models = "gpt-3.5-turbo", "gpt-4"
        response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a helpful coding assistant. You should always return back only code"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        response_content = response.choices[0].message['content']
        return response_content

class Agent:
    def __init__(self, openai_api_key=None):
        self.app = create_jupyter_app()
        load_dotenv()
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.llm = LLM(self.openai_api_key)
        time.sleep(2)
    
    def call(self, user_instruction, model='gpt-3.5-turbo', execute=False):
        self.app.commands.execute('docmanager:save')
        time.sleep(2)
        html_data = extract_html_from_current_notebook()
        extracted_cells = extract_inputs_outputs_from_html(html_data)
        previous_code_and_output_string = get_previous_code_and_output_string(extracted_cells)
        prompt = user_instruction + ' ' + previous_code_and_output_string
        print("Prompt: ", prompt)
        agent_output = self.llm.ask_gpt(prompt=prompt, model=model)
        inject_code_to_current_notebook(agent_output, execute=execute)
    