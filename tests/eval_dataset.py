"""Golden Q&A set grounded in DATA/true_data, used by the RAGAS regression suite."""

EVAL_CASES = [
    {
        "question": "What restart policies can a Kubernetes Job's Pod have, and what happens under each when the container fails?",
        "reference": (
            "A Job's Pod can use OnFailure, where the kubelet restarts the container "
            "in place within the same Pod if it crashes, or Never, where the entire "
            "Pod is marked failed and the Job controller creates a new Pod to retry."
        ),
    },
    {
        "question": "How does a Kubernetes Job differ from a Deployment?",
        "reference": (
            "A Job runs a finite workload to completion and its Pods exit successfully "
            "once done, whereas a Deployment manages Pods that run indefinitely as a "
            "long-running service."
        ),
    },
    {
        "question": "Which kubectl command shows the completion status of a Kubernetes Job, and what does the COMPLETIONS column mean?",
        "reference": (
            "`kubectl get jobs` shows job status; the COMPLETIONS column reports the "
            "number of completed pods out of the desired number of completions, e.g. 1/1."
        ),
    },
    {
        "question": "What are the three statuses a Kubernetes Job can have?",
        "reference": (
            "A Job can be Active (still running), Completed (finished successfully), "
            "or Failed (failed after reaching its backoff limit)."
        ),
    },
    {
        "question": "In the Kubernetes fine parallel processing work-queue pattern, what service holds the queue of work items, and how does each worker pod consume it?",
        "reference": (
            "A Redis instance holds the work queue; each worker pod leases one item "
            "from the queue, processes it, marks it complete, and repeats until the "
            "queue is empty."
        ),
    },
    {
        "question": "According to Kubernetes networking fundamentals, how can Pods communicate with each other?",
        "reference": (
            "All Pods can communicate with all other Pods without NAT, and all nodes "
            "can communicate with all Pods (and vice versa) without NAT."
        ),
    },
    {
        "question": "What is a Kubernetes Node, and what is a Namespace?",
        "reference": (
            "A Node is a single physical or virtual host capable of running pods, "
            "managed by the master and running at minimum kubelet and kube-proxy. "
            "A Namespace is a logical cluster or environment used as the primary "
            "method of dividing a cluster or scoping access."
        ),
    },
    {
        "question": "What are the two main types of Kubernetes Pod autoscalers, and what does each one scale?",
        "reference": (
            "The Horizontal Pod Autoscaler (HPA) scales the number of pod replicas "
            "based on metrics like CPU utilization, while the Vertical Pod Autoscaler "
            "(VPA) adjusts the CPU/memory resource requests and limits of existing pods."
        ),
    },
]
