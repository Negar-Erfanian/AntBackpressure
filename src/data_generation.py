import argparse
from backpressureAnt import *

# input arguments calling from command line
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="../ba_graph_100", type=str, help="test data directory.")
parser.add_argument("--gtype", default="ba", type=str, help="graph type.")
parser.add_argument("--size", default=100, type=int, help="size of dataset")
parser.add_argument("--seed", default=500, type=int, help="initial seed")
args = parser.parse_args()

data_path = args.datapath
gtype = args.gtype
size = args.size
seed0 = args.seed
# Create fig folder if not exist
if not os.path.isdir(data_path):
    os.mkdir(data_path)

graph_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

for num_nodes in graph_sizes:
    for id in range(size):
        seed = id + seed0
        # seed = id + 200
        m = 2
        # num_nodes = np.random.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
        # num_nodes = np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
        bp_env = Backpressure(num_nodes, 100, seed=seed, m=m, pos='new', gtype=gtype)
        flows_perc = np.random.randint(15, 30)
        num_flows = round(flows_perc/100 * num_nodes)
        nodes = bp_env.graph_c.nodes()
        num_arr = np.random.permutation(nodes)
        arrival_rates = np.random.uniform(0.2, 1.0, (num_flows,))
        # link_rates = np.random.randint(12, (num_flows,))
        link_rates = np.random.uniform(10, 14, size=(bp_env.num_links,))
        pos_c = np.zeros((num_nodes, 2))
        for i in range(num_nodes):
            pos_c[i, :] = bp_env.pos_c[i]

        flows = []
        for fidx in range(num_flows):
            src = num_arr[2*fidx]
            dst = num_arr[2*fidx+1]
            bp_env.add_flow(src, dst, rate=arrival_rates[fidx])
            flow = {'src': src, 'dst': dst, 'rate': arrival_rates[fidx]}
            flows.append(flow)

        # bp_env.flows_init()
        filename = "bp_case_seed{}_m{}_n{}_f{}.mat".format(seed, m, num_nodes, num_flows)
        filepath = os.path.join(data_path, filename)
        sio.savemat(filepath,
                    {"network": {"num_nodes": num_nodes, "seed": seed, "m": m},
                     "link_rate": link_rates,
                     "flows": flows,
                     "pos_c": pos_c
                    })






