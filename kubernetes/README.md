# Preperation before usage

## Prep:

1. Install kubectl: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/

2. Add kubernetes credentials: ```cp config04 ~/.kube/config```

3. Run ```kubectl get pods``` to check connection

## Manual deployment
1. Change directory ```cd deployment```

2. Deploy latest ami_server and ami_frontend build ```kubectl apply -f .```

3. To apply only one do ```kubectl apply -f frontend_deployment.yml``` or ```kubectl apply -f server_deployment.yml```
