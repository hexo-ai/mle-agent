import os

from tree_sitter import Language, Parser

FOLDER_PATH = f"{os.path.dirname(os.path.realpath(__file__))}{os.path.sep}"

Language.build_library(
    # library
    f"{FOLDER_PATH}langs/my-languages.so",
    # Languages
    [f"{FOLDER_PATH}langs/tree-sitter-python"],
)

PY_LANGUAGE = Language(
    f"{FOLDER_PATH}langs/my-languages.so",
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
