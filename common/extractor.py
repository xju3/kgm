
from llama_index.extractors.entity import EntityExtractor
from llama_index.extractors.marvin import MarvinMetadataExtractor
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
import nest_asyncio
import nltk

def title_extractor(nodes):
    nest_asyncio.apply()
    extractor = TitleExtractor()
    return extractor.extract(nodes=nodes)


def keyword_extractor(nodes):
    nest_asyncio.apply()
    extractor = KeywordExtractor()
    return extractor.extract(nodes=nodes)

def entity_extractor(nodes):
    nest_asyncio.apply()
    nltk.download('punkt_tab')
    extractor = EntityExtractor(device="cpu", label_entities=True)
    return extractor.extract(nodes=nodes)