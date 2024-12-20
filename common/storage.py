
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres.base import PGVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from sqlalchemy import make_url
import os


pg_uri = os.getenv("PG_URI")
mongo_uri = os.getenv("MONGO_URI")
username=os.getenv("NEO4J_USER")
password=os.getenv("NEO4J_PASS") 
url=os.getenv("NEO4J_URI")
database=os.getenv("NEO4J_DB")

def get_neo4j_storage_context(config, database=database):
    '''
        connecto to neo4j database.
    '''
    config.logger.debug(f'{username}/{password}/{database}')
    graph_store = Neo4jGraphStore(username=username, 
                                  password=password,
                                  url=url,
                                  database=database,)
    return StorageContext.from_defaults(graph_store=graph_store)


def get_mongo_storage_context(config):
    '''
        mongodb
    '''
    config.logger.debug(f"{mongo_uri}")
    return StorageContext.from_defaults(docstore=MongoDocumentStore.from_uri(uri=mongo_uri), 
                                        index_store=MongoIndexStore.from_uri(uri=mongo_uri),)
                                    
def get_pg_storage_context(db_schema, dims):
    '''
        postgres vector db.
    '''
    url = make_url(pg_uri)
    pg_doc_store = PostgresDocumentStore.from_uri(uri=pg_uri, 
                                                  table_name='doc_store', 
                                                  schema_name=db_schema)

    pg_idx_store = PostgresIndexStore.from_uri(uri=pg_uri, 
                                               table_name='idx_store', 
                                               schema_name=db_schema)

    pg_vec_store = PGVectorStore.from_params(database=url.database, 
                                             host=url.host, port=url.port, 
                                             password=url.password, user=url.username, 
                                             schema_name=db_schema, table_name=f"vec_store_{dims}", 
                                             embed_dim=dims,  # openai embedding dimension 
                                             hnsw_kwargs = {
                                                "hnsw_m": 16,
                                                "hnsw_ef_construction": 64,
                                                "hnsw_ef_search": 40,
                                                "hnsw_dist_method": "vector_cosine_ops",
                                             },
        )
    return StorageContext.from_defaults(index_store=pg_idx_store, 
                                                    docstore=pg_doc_store, 
                                                    vector_store=pg_vec_store)
