

from llama_index.core.node_parser import(SentenceWindowNodeParser, 
                                         SemanticDoubleMergingSplitterNodeParser, 
                                         SimpleFileNodeParser, 
                                         SemanticSplitterNodeParser,
                                         NodeParser, 
                                         LlamaParseJsonNodeParser,
                                         HTMLNodeParser, 
                                         JSONNodeParser, 
                                         SimpleNodeParser, 
                                         MarkdownNodeParser,
                                         LangchainNodeParser, 
                                         HierarchicalNodeParser, 
                                         MarkdownElementNodeParser,
                                         UnstructuredElementNodeParser)
from llama_index.core.node_parser import (TokenTextSplitter, 
                                          MetadataAwareTextSplitter,
                                          SentenceSplitter, 
                                          CodeSplitter)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor



def gen_nodes_by_pipeline(docs, embedding):
    sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20, include_prev_next_rel=True)
    title_extractor = TitleExtractor()
    keyword_extractor = KeywordExtractor()
    transformations = [sentence_splitter, title_extractor, keyword_extractor, embedding]
    pipeline = IngestionPipeline(transformations = transformations,)
    return pipeline.run(documents=docs, num_workers=8,)

def sentence_splitter(docs):
    splitter = SentenceSplitter(chunk_size=512, 
                                chunk_overlap=20, 
                                separator=" ", 
                                include_prev_next_rel=True)
    return splitter.get_nodes_from_documents(documents=docs)

def simple_file_node_parser(docs):
    parser = SimpleFileNodeParser()
    return parser.get_nodes_from_documents(documents=docs)

def sentence_window_node_parser(docs):
    parser = SentenceWindowNodeParser.from_defaults(window_size=2, window_metadata_key="text_window", original_text_metadata_key='original_sentence')
    return parser.get_nodes_from_documents(documents=docs)

def unstructured_element_node_parser(docs, embedding):
    import nest_asyncio
    nest_asyncio.apply()
    parser = UnstructuredElementNodeParser()
    return parser.get_nodes_from_documents(documents=docs)

def sematic_splitter_node_parse(docs, embedding):
    parser = SemanticSplitterNodeParser.from_defaults(embed_model=embedding)
    return parser.build_semantic_nodes_from_documents(docs)