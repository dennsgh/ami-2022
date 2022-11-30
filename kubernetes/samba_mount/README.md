# SAMBA Storage Mount

Use this example as a template for your own pods. The mounted volume allows you to store
large amounts of data in a persistent storage.

## Usage
1. Replace the values in the `smbcreds-groupXX.yml` with your own access credentials.
   The values have to be base64 encoded before you add them to the file.
    
   Use the command `echo "sting to encode" | base64`  and copy the output.

2. Change all the references to group01 to your own group. The Search and Replace feature 
   of most modern IDEs works well for this.

3. Execute ```kubectl apply -f .``` from the `ami_k8s/examples/samba_mount` directory.

4. Check that your mounted volumes works by executing `df -h` inside your pod.
   To access you pod use `kubectl exec -it -n groupXX pod/<your pod name> -- bash` .
   Find your pod name with the command `kubectl get -n groupXX pods` .


