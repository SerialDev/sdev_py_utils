# import os
# import codecs
# from whoosh import index, sorting
# from whoosh.fields import Schema, NUMERIC, ID, TEXT
# from whoosh.analysis import SimpleAnalyzer
# from whoosh.qparser import QueryParser
# 
# def open_index(indexdir, incremental=False):
#     """
#     * -------------{Function}---------------
#     * Opens the index, if dir or index do not exist they are created . . . 
#     * -------------{returns}----------------
#     * whoos.Index . . . 
#     * -------------{params}-----------------
#     * : indexdir::str -- name of the index directory
#     * : incremental::bool -- whether to preserve the index
#     """
#     if not os.path.exists(indexdir):
#         os.makedirs(indexdir)
#     if incremental and index.exists_in(indexdir):
#         return index.open_dir(indexdir)
# 
#     schema = Schema(number=NUMERIC(stored = True),
#                     filename= ID(stored = True),
#                     line=TEXT(analyzer = SimpleAnalyzer(), stored = True))
#     return index.create_in(indexdir, schema)
# 
# def update_index(files, ix = 'indexdir', incremental = False,
#                  batch='False', tmpdir=None, on_next_file = None):
#     """
#     * -------------{Function}---------------
#     * Updates the given index if an index object is not passed it is  
#     * loaded from (or created in) a directory named 'indexdir' in
#     * the current working directory  . . . 
#     """
#     if isinstance(ix, str):
#         ix = open_index(ix, incremental)
# 
#     # Index the file contents
#     if batch:
#         writer = ix.writer(dir= tmpdir)
#     for fileno, filename in enumerate(files, 1):
#         if on_next_file:
#             on_next_file(fileno, filename)
#         if not os.path.exists(filename):
#             continue
#         if not batch:
#             writer = ix.writer(dir = tmpdir)
#         if incremental:
#             writer.delete_by_term('filename', filename)
#         for number, line in enumerate(codecs.open(filename, 'r', 'latin-1'), 1):
#             writer.add_document(filename = filename,
#                                 number = number,
#                                 line = line.rstrip('\n'))
#         if not batch:
#             writer.commit()
# 
# def search(term, ix='indexdir', limit=None):
#     # Load index
#     if isinstance(ix, str):
#         ix = index.open_dir(ix)
#     # Parse search terms
#     s = ix.searcher()
#     parser = QueryParser('line', schema = ix.schema)
#     q = parser.parse(term)
# 
#     # Searcj  and sort the results
#     mf  = sorting.MultiFacet()
#     mf.add_field('filename')
#     mf.add_field('number')
#     return s.search(q, limit = limit, sortedby = mf)
# 
# def search_page(term, ix = 'indexdir', page = None, pagelen = 20):
#     # Load the index
#     if isinstance(ix, str):
#         ix = index.open_dir(ix)
# 
#     # Parse the search terms
#     s = ix.searcher()
#     parser = QueryParser('line', schema = ix.schema)
#     q = parser.parse(term)
# 
#     # Search and sort the results
#     mf = sorting.MultiFacet()
#     mf.add_field('filename')
#     mf.add_field('number')
#     return s.search_page(q, page, pagelen = pagelen, sortedby = mf)
#         


import os
import json
import logging

from dateutil.parser import parse as string_to_date
from distutils.util import strtobool as string_to_bool

from whoosh.analysis import (CharsetFilter, LowercaseFilter,
                             StopFilter, RegexTokenizer)
from whoosh.fields import (Schema, ID, KEYWORD, TEXT,
                           DATETIME, NUMERIC, BOOLEAN)
from whoosh.support.charset import accent_map
from whoosh.qparser import MultifieldParser
from whoosh.query import FuzzyTerm, Or
from .stopwords import stoplists
#from whoosh import index as whoosh_index

class CustomFuzzyTerm(FuzzyTerm):
    """
    Custom FuzzyTerm query parser to set a custom maxdist
    """

    def __init__(self, fieldname, text, boost=1.0, maxdist=1):
        FuzzyTerm.__init__(self, fieldname, text, 1.0, 2)


logger = logging.getLogger('indexer' + __name__)

##==========================={Index-Schema}=====================================

chfilter = CharsetFilter(accent_map)
stoplist = stoplists["en"].union(stoplists['ru'])
analyzer = RegexTokenizer() | LowercaseFilter() | \
    StopFilter(stoplist = stoplist) | chfilter

# Define the schema
keywordType = KEYWORD(lowercase=True, scorable=True)

def add_fields(schema):
    """
    * -------------{Function}---------------
    * Add dynamic fields so each document can index its fields in
    * the same Whoosh index
    * -------------{returns}----------------
    * Whoosh Schema . . . 
    * -------------{params}-----------------
    * : whoosh.fields.Schema
    """
    schema.add('*_string', TEXT(analyzer=analyzer), glob=True)
    schema.add('*_date', DATETIME, glob=True)
    schema.add('*_number', NUMERIC, glob=True)
    schema.add('*_boolean', BOOLEAN, glob=True)
    return schema

def load_index(directory, indexer):
    """
    * -------------{Function}---------------
    * Load Whoosh index . . . 
    * -------------{returns}----------------
    * whoosh.index object . . . 
    * -------------{params}-----------------
    * : directory -> path where the whoosh file is
    """
    index = indexer.open_dir(os.path.join(directory, "indexes"))
    return index

def get_schema():
    schema = Schema(content=TEXT(analyzer = analyzer),
                     doc_type=TEXT,
                     doc_id=ID(stored=True, unique=True),
                     tags=keywordType)
    schema = add_fields(schema)
    return schema

def create_index(directory, indexer, schema = None, ):
    """
    * -------------{Function}---------------
    * Creates the index folder and Whoosh index files if they do not exists
    * loads existing index 
    * -------------{params}-----------------
    * : directory -> directory to create whoosh file
    * : schema -> whoosh index schema
    """
    if schema == None:
        schema = get_schema()
    
    if not os.path.exists(os.path.join(directory, "indexes")):
        os.mkdir(os.path.join(directory, "indexes"))
        index = indexer.create_in(os.path.join(directory, "indexes"), schema)
    else:
        logger.warning("Index already exists in : {directory} \n loading index instead ".format(directory = directory))

        index = load_index(directory, indexer)
    return index

def create_doctypes_schema(directory):
    """
    * -------------{Function}---------------
    * Creates the doctypes default schema and folder if one does not exist . . . 
    * -------------{params}-----------------
    * : directory -> path for index
    """
    if not os.path.exists(os.path.join(directory, "doctypes")):
        os.mkdir(os.path.join(directory, "doctypes"))
    if not os.path.exists(os.path.join(directory, "doctypes","doctypes_schema.json")):
        with open(os.path.join(directory, "doctypes","doctypes_schema.json"), 'w') as defaultFile:
            defaultFile.write("{}")

def load_doctypes_schema(directory):
    """
    * -------------{Function}---------------
    * Loads the doctypes schema if its valid, recreates it if one does not exist
    * Doctypes schema is a dictionary of doctypes with their fields created
    * and updated when a document is indexed.
    * In order to tell Whoosh which fields to search by default, since there is 
    * no way to say "search in all fields"
    * -------------{returns}----------------
    * doctypeSchema . . . 
    * -------------{params}-----------------
    * : directory -> path for index
    """
    with open(os.path.join(directory, "doctypes","doctypes_schema.json"), 'r+') as rawJSON:
        try:
            doctypes_schema = json.load(rawJSON)
        except ValueError:
            rawJSON.write("{}")
            doctypes_schema = {}
    return doctypes_schema

def update_doctypes_schema(directory, doctypes_schema, schema_to_update):
    """
    * -------------{Function}---------------
    * Updates and persists doctypes schema in its file . . . 
    * -------------{params}-----------------
    * : directory -> path for index
    * : doctypes_schema -> current schema
    * : schema_to_update -> update to current schema
    """
    doctypes_schema.update(schema_to_update)
    with open(os.path.join(directory, "doctypes","doctypes_schema.json"), 'w') as f:
        f.write(json.dumps(doctypes_schema))

def clear_index(directory, indexer, schema):
    """
    * -------------{Function}---------------
    * Clears the index folder and doctypes index files if they  exists
    * creates a new index and doctype dir even if dir has already exists 
    * -------------{params}-----------------
    * : directory -> directory to create whoosh file
    * : schema -> whoosh index schema
    """
    if os.path.exists(os.path.join(directory, "indexes")):
        indexer.create_in(os.path.join(directory, "indexes"), schema)

    if os.path.exists(os.path.join(directory, "doctypes")):
        with open(os.path.join(directory, "doctypes", "doctypes_schema.json"), 'w') as f:
            f.write("{}")

##==========================={Indexer}==========================================

def get_field_type(field, fields_type):
    """
    * -------------{Function}---------------
    * Determines the field type based on the field name(convention) . . . 
    * -------------{returns}----------------
    * field_type . . . 
    """
    supported_types = ['string', 'numeric', 'date', 'boolean']

    # Checks if the field type has been passed
    field_type = fields_type.get(field, None)
    if field_type is not None and field_type in supported_types:
        return field_type
    else:
        return "default"

def get_typed_field_name(field, field_type):
    """
    * -------------{Function}---------------
    * returns the field name with its whoosh type appended . . . 
    * -------------{returns}----------------
    * type name . . . 
    """
    typed_field_name = "{field}_{field_type}".format(field = field, field_type = field_type)
    return typed_field_name.lower()

def get_formatted_data(data, field_type):
    """
    * -------------{Function}---------------
    * Converts data from string to the relevant type . . . 
    """
    if field_type == 'string':
        return data.decode("utf-8")
    elif field_type == 'date':
        return string_to_date(data)
    elif field_type == 'boolean':
        return string_to_bool(data)


def indexed_doc(directory, indexer,  doc_type, doc, fields, fields_type):
    """
    * -------------{Function}---------------
    * Add a doc to index, tag and given fields are also stored . . . 
    """
    index_schema = load_index(directory, indexer)
    doctypes_schema = load_doctypes_schema(directory)

    # The document that will be indexed
    indexed_doc = {}

    # Support for old way of indexing
    contents = []

    # caching the result to avoid calling it from the loop
    fields_in_doc = doc.keys()

    # the fields that the indexer will store as schema of the doctype
    fields_in_schema = []

    # normalize doct_type
    doc_type = doc_type.lower()

    # Extracts and formats every doc_field to be indexed
    for field in fields:
        # Process the field only if it exists and if it's not a special one
        if field in fields_in_doc and field not in ['id', 'doc_type', 'tags']:
            data = doc[field]

            # Field type is needed to convert the data into the proper type
            # from string
            field_type = get_field_type(field, fields_type)

            if field_type == "default":
                logger.warning("Field {field} is going to be indexed with default setting ".format(field = field))

                # Only strings are supported in BC mode
                if isinstance(data, str):
                    contents.append(data)
                else:
                    logger.warning("Data type not supported for field {field} ({data})".format(field = field, data = data) )
            else:
                typed_field_name = get_typed_field_name(field, field_type)
                fields_in_schema.append(typed_field_name)
                indexed_doc[typed_field_name] = get_formatted_data(data, field_type)
        # Non breaking handle of error cases
        elif field not in fields_in_doc:
            logger.warning("cannot find field {field} in document".format(field = field))
        else:
            logger.warning("Field {field} is automatically indexed".format(field = field))
    # Adds the doctype as a tag
    tags = doc["tags"].append(doc_type)
    tags = u" ".join(doc["tags"][0::1])

    # adds special fields
    indexed_doc["doc_id"] = doc["id"]
    indexed_doc["tags"] = tags
    indexed_doc["doc_type"] = doc_type
    indexed_doc["content"] = u"  ".join(contents)

    logger.info("About to index {indexed}".format(indexed = indexed_doc.keys()))
    writer = index_schema.writer()
    writer.update_document(**indexed_doc)
    writer.commit()

    logger.info("Update schema for doc_type {doc_type} with {fields_in_schema}".format(doc_type = doc_type, fields_in_schema = fields_in_schema))

    schema_to_update = {doc_type: fields_in_schema}
    update_doctypes_schema(directory, doctypes_schema, schema_to_update)
    
def search_doc(directory, word, doc_types, num_page=1, num_by_page=10, show_num_results=True):
    """
    * -------------{Function}---------------
    * Returns a list of docs that contains a given set of words that matches a g
    * -------------{returns}----------------
    * {set} query results . . . 
    * -------------{params}-----------------
    * : directory -> path of the index
    * : word -> words to query
    * : doc_types -> type of doc to search
    * : num_page -> number of pages to search
    * : show_num_results -> number of results to return
    """
    index_schema = load_index(directory)
    doctypes_schema = load_doctypes_schema(directory)


    # Retrieves the fields to search from the doctypes schema
    fields_to_search = []
    for doc_type in doc_types:
        doc_type = doc_type.lower()
        try:
            schema = doctypes_schema[doc_type]
            fields_to_search = fields_to_search + schema
        except:
            logger.warning("Schema not found for {doc_type}".format(doc_type = doc_type))

    # By default we search "content" (for BC) and "tags"
    fields = ['content', 'tags'] + fields_to_search
    logger.info("search will be performed on fields {fields}".format(fields = fields))

    # Creates the query parser
    # MultifieldParser allows search on multiple fields
    # We use custom FuzzyTerm class to set the Leveshtein distance to 2
    parser = MultifieldParser(fields, schema=doctypes_schema, termclass=CustomFuzzyTerm)
    query = parser.parse(word)

    # Creates a filter on the doctype field
    doctype_filter_matcher = []
    for doc_type in doc_types:
        term = FuzzyTerm("doc_type", doc_type.lower(), 1.0, 2)
        doctype_filter_matcher.append(term)

    doc_type_filter = Or(doctype_filter_matcher)

    # Processes the search(request the index, whoosh magic)
    with index_schema.searcher() as searcher:
        results = searcher.search_page(query, num_page,
                                       pagelen= num_by_page, filter=doc_type_filter )
        results_id = [result["doc_id"] for result in results]
        logger.info("Results: {results_id}".format(results_id = results_id))

        # Ensures BC if the number of results is not requested
        if show_num_results:
            return {'ids': results_id, 'num_results':len(results)}
        else:
            return {'ids': results_id}

def remove_doc(directory, id):
    """
    * -------------{Function}---------------
    * Remove matching doc from the index . . . 
    """
    index_schema = load_index(directory)
    writer = index_schema.writer()
    writer.delete_by_term("doc_id", id)
    writer.commit()

def remove_all(directory, indexer):
    schema = get_schema()
    clear_index(directory, indexer , schema )
