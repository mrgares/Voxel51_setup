FROM voxel51/fiftyone:latest

# Install dependencies for adding the MongoDB repository key and ensure lsb-release is available
RUN apt-get update && apt-get install -y wget gnupg lsb-release curl

# Add MongoDB 4.4 repository and GPG key (correctly store the key in /usr/share/keyrings)
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-4.4.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list

# Manually install libssl1.1
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
RUN dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb

# Install MongoDB 4.4
RUN apt-get update && apt-get install -y mongodb-org

# Create MongoDB data directory
RUN mkdir -p /data/db

# Expose MongoDB port
EXPOSE 27017

# Expose FiftyOne App port
EXPOSE 5151

# Start MongoDB and FiftyOne
CMD mongod --logpath /var/log/mongodb.log --bind_ip_all && fiftyone app launch --address 0.0.0.0 --port 5151
