# To ingest the document embeddings in vectorstore use the following command

```bash
python ingest.py --file data/documents/NTP_Report.pdf
```

# To start the app

```bash
python main.py
```

loader.py -> PDF loading + chunking + metadata
embedder.py -> embedding helper layer
vectorstore.py -> Chroma setup + storing chunks + retriever
ingest.py -> one-time ingestion script
semantic.py -> retrieval function

To delete and remove reranker model

%USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-reranker-base

```cmd
rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-reranker-base
```

Delete all Hugging Face cached models

%USERPROFILE%\.cache\huggingface

```cmd
rmdir /s /q %USERPROFILE%\.cache\huggingface
```