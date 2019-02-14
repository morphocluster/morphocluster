from morphocluster.processing import Tree
import pandas as pd
import os

# Input
tree_fn = "/data1/mschroeder/NoveltyDetection/Morphocluster/export/2018-12-02-23-43-29--51--rk-2-128-64-32-16-8-4-h-average_c.zip"

# Output
label_fn = "/data1/mschroeder/NoveltyDetection/Morphocluster/export/2018-12-02-23-43-29--51--rk-2-128-64-32-16-8-4-h-average_c_flat_labels.csv"

tree = Tree.from_saved(tree_fn)

node_idx = pd.Index(tree.nodes["node_id"])

def get_nodes(node_ids):
    return [
        tree.nodes.loc[node_idx.get_loc(node_id)]
        for node_id in node_ids]


def clean_path_name(path):
    result = ""

    for name in path:
        if not result:
            result = name
            continue

        components = name.split("/")

        # Filter out components that are already in the result
        components = [c for c in components if c not in result]

        result = "/".join([result] + components)

    return result


tree.objects["label"] = None

root_id = tree.get_root_id()

for path, node_ids in tree.walk():
    path_nodes = get_nodes(path[1:])
    path_name = clean_path_name(str(node["name"]) for node in path_nodes)

    print(path_name)

    nodes = get_nodes(node_ids)

    # Do not descend into unnamed nodes
    node_ids[:] = [n["node_id"] for n in nodes if not pd.isnull(n["name"])]

    for node in nodes:
        if node["node_id"] == root_id:
            continue

        if pd.isnull(node["name"]):
            # This node has no name, append all objects below to the parent node
            subtree = sum(
                (node_ids for _, node_ids in tree.walk([node["node_id"]])), [])

            object_selector = tree.objects["node_id"].isin(subtree)
            tree.objects.loc[object_selector, "label"] = path_name
        else:
            node_name = str(node["name"])

            canonical_name = clean_path_name((path_name, node_name))

            print(" {}".format(canonical_name))

            # This node has a name, append objects to this node
            object_selector = tree.objects["node_id"] == node["node_id"]
            tree.objects.loc[object_selector, "label"] = canonical_name

result_mask = ~tree.objects["label"].isna()
print("{:,d} labeled objects.".format(result_mask.sum()))
tree.objects[result_mask].to_csv(label_fn, columns=["object_id", "label"], index=False)
