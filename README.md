# Ant Backpressure Routing for Wireless Multi-hop Networks with Mixed Traffic Patterns

## Abstract 
A mixture of streaming and short-lived traffic presents a common yet challenging scenario for Backpressure routing in wireless multi-hop networks. Although state-of-the-art shortest-path biased backpressure (SP-BP) can significantly improve the latency of backpressure routing while retaining throughput optimality, it still suffers from the last-packet problem due to its inherent per-commodity queue structure and link capacity assignment. To address this challenge, we propose Ant Backpressure (Ant-BP), a fully distributed routing scheme that incorporates the multi-path routing capability of SP-BP into ant colony optimization (ACO) routing, which allows packets of different commodities to share link capacity in a first-in-first-out (FIFO) manner. Numerical evaluations show that Ant-BP can improve the latency and delivery ratio over SP-BP and ACO routing schemes, while achieving the same throughput of SP-BP under low-to-medium traffic loads.

```latex
@INPROCEEDINGS{erfaniantaghvayi2024ant,
  title={Ant Backpressure Routing for Wireless Multi-hop Networks with Mixed Traffic Patterns},
  author={Erfaniantaghvayi, Negar and Zhao, Zhongyuan and Chan, Kevin and Verma, Gunjan and Swami, Ananthram and Segarra, Santiago},
  booktitle={IEEE Military Communications Conference (MILCOM)}, 
  note={arXiv preprint arXiv:2408.12702},
  year={2024},
  pages={},
  doi={},
}
```

Preprint <https://arxiv.org/abs/2408.12702>

## Papers

### 1. Survey papers
1. Zhang, Hang, Xi Wang, Parisa Memarmoshrefi, and Dieter Hogrefe. "A survey of ant colony optimization based routing protocols for mobile ad hoc networks." IEEE access 5 (2017): 24139-24161.
2. Beegum, T. Rehannara, Mohd Yamani Idna Idris, Mohamad Nizam Bin Ayub, and Hisham A. Shehadeh. "Optimized routing of UAVs using Bio-Inspired Algorithm in FANET: A Systematic Review." IEEE Access (2023).


## Setting Up the Docker Container on the Server

1. **Pull the Docker Image**  
   Use the following SSH command to pull the required Docker image:  
   ```bash
   docker pull tensorflow/tensorflow:2.9.1-gpu-jupyter
   ```

2. **Clone the project into the directory AntBackpressure**  
   Use the following SSH command to pull the required Docker image:  
   ```bash
   git clone git@github.com:Negar-Erfanian/AntBackpressure.git ~/AntBackpressure
   ```

3. **Launch the Container**  
   Start the Docker container with the command:  
   ```bash
   docker run --user $(id -u):$(id -g) -it --rm -v ~/AntBackpressure:/tf/AntBackpressure -w /tf/AntBackpressure -p 8123:8888 tensorflow/tensorflow:2.9.1-gpu-jupyter
   ```

4. **Access Jupyter Interface**  
   Open the Jupyter interface in your browser, then launch a terminal from Jupyter and run the following commands to set up the environment:  
   ```bash
   cd AntBackpressure
   pip3 install -r requirements.txt
   ```

5. **Test the Setup**  
   From the Jupyter terminal, test the setup by running:  
   ```bash
   python3 backpressureAnt.py 49
   ```  
   If any errors occur, install the missing packages as indicated in the error message.

6. **Commit the Container**  
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

