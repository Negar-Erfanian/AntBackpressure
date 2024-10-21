# Backpressure-based Ant pheromone routing for mobile ad hoc networks

## Papers

### 1. Survey papers
1. Zhang, Hang, Xi Wang, Parisa Memarmoshrefi, and Dieter Hogrefe. "A survey of ant colony optimization based routing protocols for mobile ad hoc networks." IEEE access 5 (2017): 24139-24161.
2. Beegum, T. Rehannara, Mohd Yamani Idna Idris, Mohamad Nizam Bin Ayub, and Hisham A. Shehadeh. "Optimized routing of UAVs using Bio-Inspired Algorithm in FANET: A Systematic Review." IEEE Access (2023).


## To do list

- [x] `bpANT_test_mixed_opts.py` read csvfile and skip redundant test cases 
- [x] `bpANT_test_mixed_opts.py` test if policy file already exists, and skip virtual routing if so.
- [ ] `backpressureAnt.py` `AntHocNet.py` inherit from backpressure object rather than repeat the code

## Setting Up the Docker Container on the Server

1. **Pull the Docker Image**  
   Use the following SSH command to pull the required Docker image:  
   ```bash
   docker pull tensorflow/tensorflow:2.9.1-gpu-jupyter
   ```

2. **Launch the Container**  
   Start the Docker container with the command:  
   ```bash
   docker run --user $(id -u):$(id -g) -it --rm -v ~/antBP:/tf/antBP -w /tf/antBP -p 8123:8888 tensorflow/tensorflow:2.9.1-gpu-jupyter
   ```

3. **Access Jupyter Interface**  
   Open the Jupyter interface in your browser, then launch a terminal from Jupyter and run the following commands to set up the environment:  
   ```bash
   cd antBP
   pip3 install -r requirements.txt
   ```

4. **Test the Setup**  
   From the Jupyter terminal, test the setup by running:  
   ```bash
   python3 backpressureAnt.py 49
   ```  
   If any errors occur, install the missing packages as indicated in the error message.

5. **Commit the Container**  
   Once the test passes, commit the container to save the installed packages. First, get the container ID using:  
   ```bash
   docker container ls
   ```  
   Assuming the container ID is `c3f279d17e0a`, commit the changes with:  
   ```bash
   docker commit c3f279d17e0a tensorflow/tensorflow:2.9.1-gpu-jupyter
   ```

Now, the newly installed Python packages will be available the next time you launch the container.
## Instructions for Running Code to Replicate Results

To begin, generate the data as outlined in the paper by executing the following command:  
`bash bash/data_gen.sh`  
Make sure to set `seed = 100` and `size = 10`. For the graph type (`gtype`), use `'poisson'` to better reflect communication networks.

Once the data is generated, follow these steps to replicate the CSV files required for generating the figures:

- For the results of Figures 2.a and 2.b, run:  
  `bash bash/test_fig3.sh`  
  `bash bash/test_fig3_antideal.sh`

- For the results of Figure 2.c, run:  
  `bash bash/test_fig4.sh`  
  `bash bash/test_fig4_antideal.sh`

- For the results of Figure 3, run:  
  `bash bash/test_fig6.sh`  
  `bash bash/test_fig6_antideal.sh`

You can adjust `max_jobs` in all bash files according to the number of available CPU cores on your server.

Finally, to generate the figures from the CSV files, execute the corresponding cell in `src/monitor.ipynb`.

