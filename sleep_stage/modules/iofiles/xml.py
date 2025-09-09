import xmltodict

def load(path):

    with open(path, "rb") as f:
        doc = xmltodict.parse(f)
    
    return doc
