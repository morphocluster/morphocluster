FROM postgres:12-alpine


RUN mkdir -p /docker-entrypoint-initdb.d
COPY ./initdb-morphocluster.sh /docker-entrypoint-initdb.d/morphocluster.sh
RUN chmod +r-x /docker-entrypoint-initdb.d/morphocluster.sh