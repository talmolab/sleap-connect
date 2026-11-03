FROM ghcr.io/talmolab/sleap-nn-cuda:latest

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN uv pip install sleap-rtc

RUN mkdir -p /app/shared_data && chmod 777 /app/shared_data

EXPOSE 8080 8001 5000 3478/udp 3478/tcp 9001/tcp 9000/tcp

ENTRYPOINT ["sleap-rtc"]
CMD ["worker"]
