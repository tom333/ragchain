import requests

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant

import json 
from pprint import pprint
import os
from dotenv import load_dotenv



load_dotenv()
QDRANT_HOST = os.environ["QDRANT_HOST"]

#####################
# Téléchargement des infos
#####################
URL = "https://data.gouv.nc/api/explore/v2.1/catalog/datasets"
res = requests.get(URL).json()

apis = []

number_pages = res['total_count']
offset = 0

while offset <= number_pages:
    response = requests.get(URL, params={'limit': 100, 'offset': offset}).json()
    apis.extend(response['results'])
    offset += 100

print(len(apis))
# print(apis[0])

    
with open("./data/datasets.json", "w", encoding="utf8") as outfile: 
    json.dump(apis, outfile, ensure_ascii=False)
    
#####################
# Mise à plat des données
#####################
with open("./data/datasets.json") as f:
    datas = json.load(f)

for ds in datas:
    if ds["visibility"] == "domain":
        print(ds["dataset_id"])
        with open(f"./data/txt/{ds['dataset_id']}.txt", "w", encoding="utf8") as txt:
            txt.write(f"""
titre: {ds["metas"]["default"]["title"]}
description: {ds["metas"]["default"]["description"]}
thèmes: {ds["metas"]["default"]["theme"]}
keyword: {ds["metas"]["default"]["keyword"]}
publisher: {ds["metas"]["default"]["publisher"]}
            """)


loader = DirectoryLoader('./data/txt/', glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
docs = loader.load()


vectordb = Qdrant.from_documents(
    docs,
    FastEmbedEmbeddings(),
    url=QDRANT_HOST,
    force_recreate=True,
    collection_name="data.gouv.nc",
)

