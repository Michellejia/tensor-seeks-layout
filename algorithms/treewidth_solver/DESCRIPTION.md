# Short Description

The artifact implements an exact algorithm for layout selection over the input dump format used in this project.

The pipeline is:

1. Parse the input dump and build the interaction graph.
2. Export the graph in PACE `.gr` format.
3. Run an external exact treewidth solver to obtain a tree decomposition.
4. Check whether the decomposition is already nice, and convert it to a nice tree decomposition if needed.
5. Run the exact dynamic program over the nice tree decomposition.
6. Write the optimal operator choices back into the input dump as a `# layout_selection` section.

The artifact intentionally excludes benchmark results, the full benchmark set, and vendored third-party solver code. It includes only two representative sample inputs for smoke testing and documents how to obtain the external treewidth solver from its public repository.
