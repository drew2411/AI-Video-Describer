# import chromadb

# client = chromadb.PersistentClient(path="./chroma_db_mad_joint")
# print("Collections available:", [c.name for c in client.list_collections()])

# coll = client.get_collection("mad_joint_clip_L14")

# print("Number of items:", coll.count())

# import chromadb

# client = chromadb.PersistentClient(path="./chroma_db_mad_joint")
# coll = client.get_collection("mad_joint_clip_L14")

# all_types = {}
# batch_size = 1000
# for i in range(0, coll.count(), batch_size):
#     res = coll.get(
#         include=["metadatas"],
#         limit=batch_size,
#         offset=i
#     )
#     for m in res["metadatas"]:
#         t = m.get("type", "unknown")
#         all_types[t] = all_types.get(t, 0) + 1

# print(all_types)

import chromadb
client = chromadb.PersistentClient(path="./chroma_db_mad_joint")
coll = client.get_collection("mad_joint_clip_L14")

res = coll.query(
    query_embeddings=[[0.0]*768],  # dummy vector same dim as CLIP-L14
    where={"type": {"$eq": "text_caption"}},
    n_results=10
)
print(res)
print("Text caption entries found:", len(res.get("ids", [[]])[0]))

