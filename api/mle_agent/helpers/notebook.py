import asyncio

import ipynbname
import nbconvert
import nbformat
from bs4 import BeautifulSoup
from ipylab import JupyterFrontEnd
from IPython.core import getipython


def extract_inputs_outputs_from_html(html_data):
    soup = BeautifulSoup(html_data, "html.parser")
    code_cells = soup.select("div.jp-Cell.jp-CodeCell.jp-Notebook-cell")
    inputs_outputs = []
    for cell in code_cells:
        input_div = cell.select_one("div.jp-InputArea.jp-Cell-inputArea")
        if input_div is not None and (tag := input_div.select_one("pre")) is not None:
            input_code = tag.text
        else:
            input_code = None
        output_div = cell.select_one("div.jp-OutputArea.jp-Cell-outputArea")

        if output_div is not None and (tag := output_div.select_one("pre")) is not None:
            output_code = tag.text
        else:
            output_code = None

        inputs_outputs.append({"input": input_code, "output": output_code})
    return inputs_outputs


def get_current_notebook_name():
    return ipynbname.name()


def extract_html_from_notebook(notebook_name):
    # Load the notebook
    notebook_filename = f"{notebook_name}.ipynb"
    with open(notebook_filename, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Create an HTML exporter with the 'basic' template
    html_exporter = nbconvert.HTMLExporter()
    html_exporter.template_name = "basic"

    # Convert to HTML
    html_data, _ = html_exporter.from_notebook_node(notebook_content)
    return html_data


def extract_html_from_current_notebook():
    notebook_name = get_current_notebook_name()
    return extract_html_from_notebook(notebook_name)


def create_jupyter_app():
    app = JupyterFrontEnd()

    async def init():
        await app.ready()
        cmds = app.commands.list_commands()[:5]
        assert len(cmds) > 0, "Jupyter Frontend is not ready. Please retry"

    _ = asyncio.create_task(init())
    return app


def get_previous_code_and_output_string(extracted_cells):
    included_cells = [
        {"input": cell["input"].replace("#Include\n", ""), "output": cell["output"]}
        for cell in extracted_cells
        if cell["input"].startswith("#Include")
    ]

    included_cells_string = []
    for cell in included_cells:
        included_cells_string.extend(
            [
                "\nInput code:",
                str(cell["input"]),
                "Corresponding output:",
                str(cell["output"]),
            ]
        )
    included_cells_string = " ".join(included_cells_string)
    previous_code_and_output_string = (
        "Here are the code cells that have been executed and their corresponding output:"
        + included_cells_string
    )
    return previous_code_and_output_string


def inject_code_to_current_notebook(code, execute=False):
    """Inject code into the current Jupyter notebook and optionally execute it."""
    # Use IPython's magic command
    shell = getipython.get_ipython()
    # Inject the code into a new cell
    if shell is not None:
        shell.set_next_input(code, replace=False)
    # Optionally execute the code
    if execute:
        exec(code)
