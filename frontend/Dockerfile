# syntax=docker/dockerfile:1

# pull the official base image
FROM node:lts-alpine3.16 as build-step
# set working direction
WORKDIR /frontend
# add `/app/node_modules/.bin` to $PATH
ENV PATH /frontend/node_modules/.bin:$PATH
# install application dependencies

COPY package.json yarn.lock ./

COPY ./src ./src

COPY ./public ./public

RUN yarn install

RUN yarn build

FROM nginx:stable-alpine
COPY --from=build-step /frontend/build /usr/share/nginx/html
COPY deployment/proxy_params /etc/nginx/proxy_params
COPY deployment/nginx.default.conf /etc/nginx/conf.d/default.conf

