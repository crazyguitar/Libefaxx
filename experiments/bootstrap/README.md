# Bootstrap

Unlike traditional TCP/IP, where a client connects using an IP address or domain
name server, **EFA provides low-level, kernel-bypass RDMA networking**, operating
below the TCP/IP layer to enable high-performance communication. Establishing a
connection requires a handshake to exchange hardware endpoints and populate each
other’s address vectors (AVs). This pattern is common in many GPU communication
libraries.

For example, [PyTorch NCCL process group](https://pytorch.org/docs/stable/distributed.html)
requires a master address to create a TCPStore server, allowing slave processes
to connect and exchange NCCL IDs for communication. Similarly, [NVSHMEM](https://developer.nvidia.com/nvshmem)
uses an MPI-based bootstrap where processes exchange peer IDs via MPI before calling
NVSHMEM APIs.

In this example, we follow a similar approach, exchanging **EFA endpoints via MPI**
to initialize connections, analogous to the MPI bootstrap used in NVSHMEM.
