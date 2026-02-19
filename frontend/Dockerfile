# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Run stage (static hosting)
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx default serves on port 80
EXPOSE 80
