from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Optional
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from itertools import repeat
from supabase import create_client, Client, ClientOptions
from langchain.embeddings import HuggingFaceEmbeddings
import os
import numpy as np
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

def _texts_to_documents(
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[Any, Any]]] = None,
    ) -> List[Document]:
        """Return list of Documents from list of texts and metadatas."""
        if metadatas is None:
            metadatas = repeat({})

        docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        return docs

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")


supabase_clinent: Client = create_client(url, key,
                                options=ClientOptions(
                                            postgrest_client_timeout=10,
                                            storage_client_timeout=10
                                        ))

embedding_model_name = 'maidalun1020/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'cpu'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}
k = 4
embed_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

###################################################################################################
# TODO: 需要一个连接外部的接口
def insertToSupabase(texts: List[str]):
    """
    Insert data to table named documents
    """

    embeddings = embed_model.embed_documents(texts)
    ids = [str(uuid.uuid4()) for _ in texts]
    docs = _texts_to_documents(texts, metadatas=None)
    SupabaseVectorStore._add_vectors(client = supabase_clinent, 
                                        table_name = 'documents', 
                                        vectors = embeddings, 
                                        documents = docs, 
                                        ids = ids, 
                                        chunk_size = 500)

def deleteFromSupabase(ids: Optional[List[str]] = None, **kwargs: Any) -> None:
    """Delete by vector IDs.

    Args:
        ids: List of ids to delete.
    """

    if ids is None:
        raise ValueError("No ids provided to delete.")

    rows: List[Dict[str, Any]] = [
        {
            "id": id,
        }
        for id in ids
    ]

    # TODO: Check if this can be done in bulk
    for row in rows:
        supabase_clinent.from_('documents').delete().eq("id", row["id"]).execute()
###################################################################################################

supabase_vector_store: Any  = SupabaseVectorStore(
                                        client=supabase_clinent,
                                        embedding=embed_model,
                                        table_name="documents",
                                        # query_name="match_documents",
                                    ) 
    
class SupabaseRetrieval(BaseTool):
    """Tool that queries Supabase"""

    name: str = "retrieve_from_knowledge_base"
    description: str = (
        "This tool queries the knowledge base and get similar results. \
        When user asks some information and indicates you to refer to the knowledge base, use this tool for reference."
    )


    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        docs = supabase_vector_store.similarity_search(query)
        print('docs length:',len(docs))
        results = ""
        for i in range(0, np.min([len(docs), k])):
            results = docs[i].page_content + "\n"
        return results
        

SupabaseRetrieval()