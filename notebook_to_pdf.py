from nbconvert import HTMLExporter
import nbformat
import codecs

# This code is originally from Tobias Lauber <3 but he allowed us to use it.


def convert_notebook_to_html(notebook_filename):
    """
    Converts a Jupyter Notebook file (.ipynb) to an HTML file.

    Parameters:
    notebook_filename (str): The name of the notebook file to convert.

    Returns:
    str: The name of the generated HTML file.
    """
    try:
        # Load the notebook
        with open(notebook_filename, "r", encoding="utf-8") as f:
            notebook_node = nbformat.read(f, as_version=4)

        # Convert to HTML
        html_exporter = HTMLExporter()
        (body, _) = html_exporter.from_notebook_node(notebook_node)

        # Generate the HTML file name
        html_filename = notebook_filename.replace(".ipynb", ".html")

        # Save the result to an HTML file
        with codecs.open(html_filename, "w", encoding="utf-8") as f:
            f.write(body)

        print(
            f"Notebook '{notebook_filename}' has been converted to '{html_filename}'."
        )
        return html_filename

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# files to convert
convert_notebook_to_html("Nikodem_Chantal_AML_HS24.ipynb")
