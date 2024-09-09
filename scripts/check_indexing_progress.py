import argparse
from wimbd.es import es_init

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--index_name", type=str, required=True)
args = parser.parse_args()

# Connect to Elasticsearch
es = es_init(args.config)

# Replace with your index name
index_name = args.index_name

# 1. Get index stats
index_stats = es.indices.stats(index=index_name)
docs_count = index_stats['indices'][index_name]['total']['docs']['count']
indexing_operations = index_stats['indices'][index_name]['total']['indexing']['index_current']

print(f"Document count: {docs_count}")
print(f"Ongoing indexing operations: {indexing_operations}")

# 2. Get cluster health
cluster_health = es.cluster.health(index=index_name)
index_health = cluster_health['status']
print(f"Index health: {index_health}")

# 3. Get pending cluster tasks
pending_tasks = es.cluster.pending_tasks()
if pending_tasks['tasks']:
    print("There are pending tasks related to the cluster:")
    for task in pending_tasks['tasks']:
        print(task)
    else:
        print("No pending cluster tasks.")

# Check if indexing is complete
if indexing_operations == 0:
    print("Indexing is complete.")
else:
    print("Indexing is still in progress.")
