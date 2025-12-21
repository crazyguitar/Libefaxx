# EFA

This example demonstrates how to programmatically query AWS Elastic Fabric Adapter (EFA)
device information using libfabric. It provides functionality equivalent to the
`fi_info -p efa` command-line tool, but exposes EFA capabilities through a
programmatic interface that can be integrated into distributed training
systems and infrastructure tools.

By retrieving EFA provider attributes, endpoints, and device capabilities,
this example supports inspection and validation of high-performance networking
configurations commonly used in large-scale LLM training and GPU cluster
deployments on AWS. The output is useful for verifying EFA availability,
diagnosing configuration issues, and understanding network characteristics
relevant to latency-sensitive and bandwidth-intensive workloads such as
data-parallel and model-parallel training.

The example in main.cu prints EFA device information in a manner similar to `fi_getinfo`.
To learn more about how EFA information is queried and how these interfaces can
be extended or utilized in practice, please refer to the source code in
[src/include/rdma/fabric/efa.h](https://github.com/crazyguitar/Libefaxx/blob/main/src/include/rdma/fabric/efa.h)
