import xmltodict


def load(path: str):
    """Load an XML file into a dictionary.

    Parameters
    ----------
    path: str
        Path to the XML file.
    """
    with open(path, "rb") as f:
        doc = xmltodict.parse(f)
    return doc
