# FiftyOne Server 

## Diagram

```mermaid
graph TD
    %% FiftyOne server setup
    subgraph FiftyOne_Server_Container
        A1[FiftyOne App]
        A2[MongoDB Database]
        A1 --> A2
    end
    
    %% Other containers
    subgraph Other_Containers
        B1[Model Container]
        B2[Training Container]
        B3[Inference Container]
    end

    %% Shared Docker network
    C[Shared Docker Network: fiftyone_network]
    C --> Other_Containers

    %% Communication between containers via shared network
    B1 --> |Send Data & Annotations via Docker Network| C
    B2 --> |Send Labels & Annotations via Docker Network| C
    B3 --> |Send Inference Results via Docker Network| C

    %% Interaction with FiftyOne app and MongoDB through the network
    C --> |API Calls and Data Transfer| A1
    C --> |Data Stored in MongoDB| A2



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

To connect other containers to it we simply do the following (example for a container called bevfusion):

```
docker run --name bevfusion_container -v `pwd`:/workspace -v /srv/datastore:/srv/datastore --shm-size=16g -it --gpus all --network fiftyone_network -e FIFTYONE_DATABASE_URI=mongodb://fiftyone_server:27017 bevfusion:last
```

