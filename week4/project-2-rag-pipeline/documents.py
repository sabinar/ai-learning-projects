# Your knowledge base — documents the RAG system will answer questions about
# Using distributed systems content since that's your domain

DOCUMENTS = [
    {
        "id": "cap-theorem",
        "title": "CAP Theorem",
        "content": """
        The CAP theorem, formulated by Eric Brewer, states that a distributed system 
        can only guarantee two of three properties simultaneously: Consistency, 
        Availability, and Partition Tolerance. Consistency means every read receives 
        the most recent write. Availability means every request receives a response. 
        Partition tolerance means the system continues operating despite network 
        failures between nodes. In practice, partition tolerance is mandatory for 
        distributed systems, so the real choice is between consistency and availability.
        Systems like HBase and Zookeeper choose consistency over availability (CP).
        Systems like Cassandra and CouchDB choose availability over consistency (AP).
        """
    },
    {
        "id": "kubernetes",
        "title": "Kubernetes",
        "content": """
        Kubernetes is an open-source container orchestration platform originally 
        developed by Google. It automates deployment, scaling, and management of 
        containerized applications. Key concepts include Pods (smallest deployable 
        units), Deployments (manage replica sets), Services (network endpoints), 
        and Namespaces (virtual clusters). Kubernetes uses a declarative model — 
        you describe the desired state and Kubernetes works to maintain it. 
        The control plane consists of the API server, etcd (distributed key-value 
        store), scheduler, and controller manager. Worker nodes run kubelet, 
        kube-proxy, and the container runtime. Kubernetes supports horizontal 
        pod autoscaling based on CPU, memory, or custom metrics.
        """
    },
    {
        "id": "kafka",
        "title": "Apache Kafka",
        "content": """
        Apache Kafka is a distributed event streaming platform designed for 
        high-throughput, fault-tolerant messaging. Core concepts include Topics 
        (categories of messages), Partitions (ordered, immutable sequences of 
        records), Producers (write to topics), and Consumers (read from topics). 
        Kafka uses a pull-based model where consumers control their read position 
        via offsets. Consumer groups allow parallel processing — each partition 
        is consumed by exactly one consumer in a group. Kafka retains messages 
        for a configurable period, enabling replay. ZooKeeper (or KRaft in newer 
        versions) manages cluster metadata. Kafka Connect integrates with external 
        systems. Kafka Streams enables real-time stream processing.
        """
    },
    {
        "id": "service-mesh",
        "title": "Service Mesh",
        "content": """
        A service mesh is a dedicated infrastructure layer for managing 
        service-to-service communication in microservices architectures. 
        It handles traffic management, observability, and security without 
        requiring application code changes. Istio is the most popular service 
        mesh, using Envoy as a sidecar proxy injected into each pod. Features 
        include load balancing, circuit breaking, retries, timeouts, mutual TLS, 
        and distributed tracing. The control plane (istiod) manages configuration 
        and certificate distribution. The data plane consists of Envoy proxies 
        that intercept all network traffic. Service meshes add latency overhead 
        (typically 1-5ms) but provide powerful observability through metrics, 
        logs, and traces.
        """
    },
    {
        "id": "raft-consensus",
        "title": "Raft Consensus Algorithm",
        "content": """
        Raft is a consensus algorithm designed to be more understandable than 
        Paxos while providing equivalent guarantees. It ensures distributed systems 
        agree on a sequence of values despite failures. Raft uses leader election — 
        one node acts as leader, handling all client requests and replicating logs 
        to followers. Leaders send heartbeats to prevent election timeouts. If a 
        follower doesn't receive heartbeats, it starts a new election. Log entries 
        are committed once a majority of nodes acknowledge them. Raft guarantees 
        that committed entries are never lost. Systems using Raft include etcd 
        (used by Kubernetes), CockroachDB, and TiKV. Split-brain scenarios are 
        prevented by requiring majority quorum for any decision.
        """
    },
    {
        "id": "docker",
        "title": "Docker Containers",
        "content": """
        Docker is a platform for building, shipping, and running applications 
        in containers. Containers package application code with all dependencies 
        into a portable unit that runs consistently across environments. Unlike 
        virtual machines, containers share the host OS kernel, making them 
        lightweight and fast. Key concepts include Images (read-only templates), 
        Containers (running instances of images), Dockerfile (instructions to 
        build an image), and Docker Hub (public registry). Docker uses union 
        filesystems for efficient layer caching — unchanged layers are reused 
        across builds. Multi-stage builds reduce final image size by separating 
        build and runtime environments. Docker Compose orchestrates multi-container 
        applications locally using a YAML configuration file.
        """
    },
    {
        "id": "mlops",
        "title": "MLOps",
        "content": """
        MLOps (Machine Learning Operations) applies DevOps principles to machine 
        learning systems. Key practices include experiment tracking (MLflow, W&B), 
        model versioning, automated training pipelines, and continuous monitoring. 
        The ML lifecycle covers data collection, feature engineering, model training, 
        evaluation, deployment, and monitoring. Data drift occurs when input 
        distribution changes after deployment. Concept drift occurs when the 
        relationship between inputs and outputs changes. Model serving patterns 
        include REST APIs, batch inference, and streaming inference. Feature stores 
        (Feast, Tecton) provide consistent features between training and serving. 
        CI/CD for ML automates testing and deployment of models and pipelines.
        """
    },
    {
        "id": "rag",
        "title": "Retrieval Augmented Generation",
        "content": """
        Retrieval Augmented Generation (RAG) combines information retrieval with 
        language model generation. Instead of relying solely on parametric knowledge 
        (weights), RAG retrieves relevant documents at query time and provides them 
        as context. The pipeline has four steps: document ingestion (load and chunk 
        documents), embedding (convert chunks to vectors), retrieval (find similar 
        chunks for a query), and generation (LLM answers using retrieved context). 
        Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings 
        for efficient similarity search. Chunking strategy affects retrieval quality — 
        chunks too large lose precision, too small lose context. RAG reduces 
        hallucinations by grounding answers in retrieved facts. Hybrid search 
        combines dense retrieval (embeddings) with sparse retrieval (BM25) for 
        better results.
        """
    },
    {
        "id": "badminton",
        "title": "Badminton techniques",
        "content": """
        Badminton techniques focus on mastering core skills like proper grip (forehand/backhand), swift footwork, and essential strokes—high clear, smash, drop, and net shots—to control the rally. Effective play combines these with tactical positioning, such as utilizing the attack clear for speed or angling shots to exploit weak areas in an opponent's defense. 
        Key Badminton Techniques & Tips
        5 Basic Skills: Master the grip, footwork, stance, and strokes for a solid foundation.
        The Smash: Use the entire body, shoulder, and forearm for power to hit the shuttlecock down hard.
        High Clear: Hit the shuttle high and deep to the opponent’s baseline to gain time.
        Drop Shot: A controlled, gentle hit from the rear court aimed just over the net.
        Net Play: Focus on net shots and kills at the front court, requiring delicate touch.
        Defending the Smash: Use a "Ready Position"—racket up and knees bent—to react to fast shots.
        Footwork: Position yourself early for shots by moving swiftly rather than just relying on arm strength
        """
    }
]