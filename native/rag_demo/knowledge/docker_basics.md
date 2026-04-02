# Docker 基础知识

## 核心概念
镜像(Image)是只读模板，容器(Container)是镜像的运行实例。
Dockerfile 定义如何构建镜像，docker-compose.yml 定义多容器编排。

## 常用命令
docker build -t myapp . 构建镜像。
docker run -d -p 8080:80 myapp 后台运行容器并映射端口。
docker ps 查看运行中的容器，docker logs 查看容器日志。
docker exec -it container_id bash 进入容器内部。

## 数据持久化
容器销毁后数据会丢失，需要用 Volume 持久化。
docker run -v /host/path:/container/path 挂载目录。
也可以用 docker volume create 创建命名卷。

## 网络
容器之间通过 docker network 通信。
同一个 network 下的容器可以用容器名互相访问。
docker-compose 会自动创建网络，服务之间用服务名通信。
