# sample subgraphs per node, multiple or single

# node features of the subgraphs are initialized by 
# $[W_x * X_v || W_p * log(P) || W_r * 1[v = root(S_i)]]$

# two message passing par layer, one inbetween subgraph and accross other subgraphs

# use gin for both message passing layers

# after L layers of message passing, we obtain represtentation for each node

# for graph classification, two layer of pooling is used, one for subgraph level and one for graph level

# for node classification, representations for each node is pooled accross subgraphs