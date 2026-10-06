import sys

from ask import get_collection, retrieve

question = " ".join(sys.argv[1:])
docs, metas = retrieve(get_collection(), question, k=8)
for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
    print(f"\n[{i}] {meta['source']} (chunk {meta['chunk']})")
    print(doc[:300] + "...")