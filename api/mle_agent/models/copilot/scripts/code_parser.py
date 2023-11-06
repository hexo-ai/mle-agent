import ast
import os
import textwrap
from typing import List

import networkx as nx
from tree_sitter import Language, Parser

FOLDER_PATH = f"{os.path.dirname(os.path.realpath(__file__))}{os.path.sep}"

Language.build_library(
    # library
    f"{FOLDER_PATH}build/my-languages.so",
    # Languages
    [f"{FOLDER_PATH}build/tree-sitter-python"],
)

PY_LANGUAGE = Language(
    f"{FOLDER_PATH}build/my-languages.so",
    "python",
)
parser = Parser()
parser.set_language(PY_LANGUAGE)


class TreeSitterPythonParser:
    def __init__(self, document):
        self.tree = parser.parse(bytes(document, "utf8"))
        self.document_lines = document.splitlines()

    def create_chunks(self):
        chunks = []
        root_node = self.tree.root_node
        for child in root_node.children:
            if child.type in [
                "import_statement",
                "import_from_statement",
                "comment",
                "expression_statement",
            ]:
                start_line_num = child.start_point[0]
                end_line_num = child.end_point[0]
                code_type = child.type
                joined_code = "\n".join(
                    self.document_lines[start_line_num : end_line_num + 1]
                )
                chunks.append(
                    {
                        "code": f"{joined_code}",
                        "start_line_num": start_line_num,
                        "end_line_num": end_line_num,
                        "type": code_type,
                    }
                )
            elif child.type in ["class_definition", "function_definition"]:
                self._traverse_tree(chunks, child)
        return chunks

    def _traverse_tree(self, chunks, node):
        if node.type in ["class_definition", "function_definition"]:
            start_line_num = node.start_point[0]
            end_line_num = node.end_point[0]
            code_type = node.type
            joined_code = "\n".join(
                self.document_lines[start_line_num : end_line_num + 1]
            )
            chunks.append(
                {
                    "code": f"{joined_code}",
                    "start_line_num": start_line_num,
                    "end_line_num": end_line_num,
                    "type": code_type,
                }
            )
        for child in node.children:
            self._traverse_tree(chunks, child)

    def extract_import_statements(self):
        # types of chunks 'class_definition', 'function_definition', 'import_from_statement',
        # 'import_statement', 'comment', 'expression_statement'
        root_node = self.tree.root_node
        import_statements = []
        for child in root_node.children:
            start_line_num = child.start_point[0]
            end_line_num = child.end_point[0]
            code_type = child.type
            if code_type in ["import_statement", "import_from_statement"]:
                joined_code = "\n".join(
                    self.document_lines[start_line_num : end_line_num + 1]
                )
                import_statements.append(joined_code)
        import_statements = "\n".join(import_statements)
        return import_statements

    def extract_main_code(self):
        main_code = []
        root_node = self.tree.root_node
        for child in root_node.children:
            start_line_num = child.start_point[0]
            end_line_num = child.end_point[0]
            code_type = child.type
            if code_type not in ["function_definition", "class_definition"]:
                joined_code = "\n".join(
                    self.document_lines[start_line_num : end_line_num + 1]
                )
                main_code.append(joined_code)
        main_code = "\n".join(main_code)
        return main_code


class AstParser:
    def __init__(self, document):
        self.document = document
        self.graph = self.construct_graph_from_document()
        self.document_lines = document.splitlines()

    def construct_graph_from_document(self):
        """
        Reads a document, splits it into lines and constructs a nx Graph.
        The nodes of the graph are classes, functions, and import statements.

        Example Usage
        from code_parser import construct_graph_from_document_ast
        graph = construct_graph_from_document_ast(document)
        for node in graph.nodes(data=True):
            print(node)
        """
        tree = ast.parse(self.document)
        graph = nx.Graph()

        # Utility function to get the end line number of a node
        def get_end_lineno(node):
            max_lineno = node.lineno
            for child_node in ast.walk(node):
                if hasattr(child_node, "lineno"):
                    max_lineno = max(max_lineno, child_node.lineno)
            return max_lineno

        # Adding nodes to the graph
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
            ):
                start_lineno = node.lineno
                end_lineno = get_end_lineno(node)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    graph.add_node(
                        node.name,
                        type=node.__class__.__name__,
                        start=start_lineno,
                        end=end_lineno,
                    )
                elif isinstance(node, ast.Import):
                    for n in node.names:
                        graph.add_node(
                            n.name, type="Import", start=start_lineno, end=end_lineno
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    for n in node.names:
                        graph.add_node(
                            f"{module}.{n.name}",
                            type="ImportFrom",
                            start=start_lineno,
                            end=end_lineno,
                        )
        return graph

    def create_chunks(self):
        chunks = []
        for _, node in self.graph.nodes(data=True):
            start_line_num = node["start"] - 1
            end_line_num = node["end"] - 1
            code_type = node["type"]
            joined_code = "\n".join(
                self.document_lines[start_line_num : end_line_num + 1]
            )
            chunks.append(
                {
                    "code": f"{joined_code}",
                    "start_line_num": start_line_num,
                    "end_line_num": end_line_num,
                    "type": code_type,
                }
            )
        node_start_line_nums = [
            node["start"] - 1 for _, node in self.graph.nodes(data=True)
        ]
        zipped_pairs = zip(node_start_line_nums, chunks)
        sorted_pairs = sorted(zipped_pairs)
        sorted_node_start_line_nums, sorted_chunks = zip(*sorted_pairs)
        return sorted_chunks

    def extract_import_statements(self):
        """
        Extracts all import statements from the document.
        """
        tree = ast.parse(self.document)
        # tree = fault_tolerant_ast_parse(document)
        import_statements = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                lineno = node.lineno - 1
                import_statements.append(self.document_lines[lineno])

        import_statements = "\n".join(import_statements)
        return import_statements

    def extract_main_code(self):
        """
        Extracts the code that doesn't belong to classes, functions, or import statements, based on the graph.
        """
        lines_to_skip = set()
        for _, data in self.graph.nodes(data=True):
            start = data.get("start", None) - 1
            end = data.get("end", None)
            if start is not None and end is not None:
                lines_to_skip.update(range(start, end + 1))

        remaining_code = [
            line
            for idx, line in enumerate(self.document_lines)
            if idx not in lines_to_skip
        ]

        return "\n".join(remaining_code)

    def extract_names_from_function(self, function_content):
        """Extract all names from a given function content using the `ast` module."""
        tree = ast.parse(function_content)
        names = set()

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                names.add(node.id)
                self.generic_visit(node)

        NameVisitor().visit(tree)
        return names

    def extract_relevant_imports(self, code: str, imports: List):
        """
        Extracts relevant imports for a given code snippet code,
        considering the aliases in the import statements using the `ast` module.
        """
        code = textwrap.dedent(code)
        imports = [imp.lstrip() for imp in imports.splitlines()]
        names_in_function = self.extract_names_from_function(code)

        relevant_imports_with_alias = []
        for imp in imports:
            tree = ast.parse(imp)

            if isinstance(tree.body[0], ast.Import):
                for name in tree.body[0].names:
                    if name.name.split(".")[0] in names_in_function or (
                        name.asname and name.asname in names_in_function
                    ):
                        if name.asname:
                            relevant_imports_with_alias.append(
                                f"import {name.name} as {name.asname}"
                            )
                        else:
                            relevant_imports_with_alias.append(f"import {name.name}")

            elif isinstance(tree.body[0], ast.ImportFrom):
                module_name = tree.body[0].module
                for alias in tree.body[0].names:
                    if (
                        module_name in names_in_function
                        or alias.name in names_in_function
                    ):
                        if alias.asname:
                            relevant_imports_with_alias.append(
                                f"from {module_name} import {alias.name} as {alias.asname}"
                            )
                        else:
                            relevant_imports_with_alias.append(
                                f"from {module_name} import {alias.name}"
                            )

        return relevant_imports_with_alias

    def extract_function_content(self, document, function_name):
        """
        Extracts the content of a specified function from the document.
        """
        tree = ast.parse(document)

        # Utility function to get the end line number of a node
        def get_end_lineno(node):
            max_lineno = node.lineno
            for child_node in ast.walk(node):
                if hasattr(child_node, "lineno"):
                    max_lineno = max(max_lineno, child_node.lineno)
            return max_lineno

        document_lines = document.splitlines()
        function_content = ""

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_lineno = node.lineno - 1
                end_lineno = get_end_lineno(node)
                function_content = "\n".join(
                    document_lines[start_lineno : end_lineno + 1]
                )

        return function_content
