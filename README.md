# FiftyOne Server 

## Diagram

```mermaid
graph TD
    %% Separate the FiftyOne server container elements more clearly
    subgraph FiftyOne_Server_Container
        A1[FiftyOne App]
        A2[MongoDB Database]
        A3[API Layer]
        A1 --> A2
        A1 --> A3
        A2 --> |MongoDB Data| A3
    end
    
    %% Other containers are placed with clearer labeling
    subgraph Other_Containers
        B1[VL Model Container]
        B2[3D Model Training Container]
        B3[Inference Container]
    end

    %% Shared network block connecting FiftyOne Server and Other Containers
    C[Shared Docker Network]
    C --> FiftyOne_Server_Container
    C --> Other_Containers

    %% Create clearer separation for each container's API access
    FiftyOne_Server_Container 
    B1 --> |Send Data| A3
    B2 --> |Send Labels & Annotations| A3
    B3 --> |Send Inference Results| A3

    %% Add space to reduce line crossings
    B1 --> A3
    B2 --> A3
    B3 --> A3

    %% Styling for clarity
    style FiftyOne_Server_Container fill:#1E1E1E,stroke:#FFFFFF,stroke-width:2px;
    style Other_Containers fill:#1E1E1E,stroke:#FFFFFF,stroke-width:2px;
    style C fill:#1E1E1E,stroke:#FFFFFF,stroke-width:2px;
    linkStyle default stroke:#FFFFFF,stroke-width:1px;


```

## Setup

To run build the image

```
docker build -t fiftyone-server .
```

Create a network so that the containers can communicate with fiftyone-server

```
docker network create fiftyone_network
```

To run the container

```
docker run -d --name fiftyone -v /srv/datastore/fiftyone/:/fiftyone -v /srv/datastore/datasets_ML/:/srv/datastore/datasets_ML/ -p 5151:5151 -p 27017:27017 -it --network fiftyone_network fiftyone-server
```

To connect other containers to it we simply do the following (example for a container called transfusion):

```
docker run --name transfusion -v `pwd`:/workspace -v /srv/datastore:/srv/datastore --shm-size=16g -it --gpus all --network fiftyone_network -e FIFTYONE_DATABASE_URI=mongodb://fiftyone:27017 lnmargar4285/transfusion
```

