from elasticsearch import Elasticsearch
import pandas as pd
import hashlib
import glob as g
import os
from time import gmtime, strftime

def logMessage(msg):
    print(msg)
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

# Generated a hash of a value.  You can pass the entire pandas row to this.
def hash_string(value):
    return (hashlib.sha1(str(value).encode('utf-8')).hexdigest())

# index to elastic search
def esIndexRecord(df, esIndex):
    es = Elasticsearch()  # init ElasticSearch

    # Create index if not present.
    # This relies on ES templates.
    if es.indices.exists(index=esIndex) == False:
        logMessage("Creating index")
        print('Index Name --> ', esIndex)
        es.indices.create(index=esIndex)

    # Insert data into ES.
    logMessage("Ingesting data into Elasticsearch")
    df_iter = df.iterrows()
    recCnt = 0
    for id, document in df_iter:
        recid = hash_string(document.to_string(header=False, index=False))
        exists = es.exists(index=esIndex, id=recid, _source=False)
        if exists == False:  # If exists, do not update.
            recCnt = recCnt + 1
            res = es.index(index=esIndex, id=recid, body=document.to_dict())
    print ("Ingested ", recCnt, " records into Elasticsearch")

def getData (filePath):
    glDf = pd.read_parquet(filePath)
    return glDf

def main(on):
    baseIndexName = 'wsb_post_20210620'
    print("Processing file, ", on)
    df = getData(on)

    # Establish elastic search instance.
    indexName = baseIndexName
    esIndexRecord(df, indexName)

if __name__ == '__main__':
    logMessage("Executing batch process.")

    filePath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210620\\wsb_*.gzip'
    files = g.glob(filePath, recursive=True)

    #Used only for loading the API data.
    #for f in files:
    #    main(f)

    # Multi processing into elastic.
    from multiprocessing import Pool
    with Pool(len(files)) as p:
        print(p.map(main, files))

    logMessage("Batch process done.")
    os._exit(0)