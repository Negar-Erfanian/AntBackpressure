import argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio
import copy
from backpressureAnt import *
import warnings
warnings.filterwarnings('ignore')


# input flags
flags = tf.compat.v1.flags

flags.DEFINE_string('datapath', '../data_ba_100', 'input data path.')  
flags.DEFINE_string('out', '../out', 'output data path.')  
flags.DEFINE_string('schemes', '1,3,4,6', 'Routing scheme.')
flags.DEFINE_string('root', '..', 'Root dir of project.')
flags.DEFINE_float('radius', 0.0, 'Interference radius.')
flags.DEFINE_string('gtype', 'ba', 'graph type or dataset dir')
flags.DEFINE_string('sizes', '100', 'List of network sizes V')
flags.DEFINE_float('lb', 1, 'Burst multiplier.')
flags.DEFINE_float('ls', 1, 'Streaming multiplier.')
flags.DEFINE_float('pburst', 0.0, 'Probability of having a flow being bursty.')
flags.DEFINE_integer('T', 1000, 'Number of time slots.')
flags.DEFINE_integer('TV', 500, 'Time steps for virtual routing')
flags.DEFINE_bool('debug', False, 'Only set to True while debugging locally')
# flags.DEFINE_bool('approx', False, 'Warmstarted under streaming traffic.')
flags.DEFINE_bool('robust_test', False, 'Doing robust test.')
# flags.DEFINE_bool('bias', True, 'Link fail and mobility.')
flags.DEFINE_string('function', 'proportional', 'pheromone routing function')
flags.DEFINE_float('exploration_rate', 0.0, 'Exploration rate for virtual ants')
flags.DEFINE_float('decay', 0.998, 'decay rate of the pheromones')
flags.DEFINE_float('unit', 0.01, 'unit being added to the pheromones')
flags.DEFINE_bool('not_going_back', False, 'Ants not going back on a link')
flags.DEFINE_bool('ph_diff', True, 'Differential pheromone on each directin of a link')
flags.DEFINE_float('init', 0.01, 'Initial value of the pheromones')
FLAGS = flags.FLAGS


'''
Input argument for test
L_stream, L_burst
|V|, dataset
P_burst (for physical traffic)
TV (virtual routing total time steps) C
T (physical routing total time steps)
Policy universe:
    - Ant-BP:
    - Ant-BP-mirror:
    - Ant-baseline:
    - Ant-coldstart:
    - Ant-ideal (AntHocNet): not included here
    - SP-BP: 
Features of each policy:
    - warmstart policy: None, SP-BP, Ant
    - virtual routing: mirror, approx
    - freeze: True (default), False
'''

# We test 6 policies, where 5 is tested in another file
lgds = {
    0: 'Ant-Baseline-ub',
    1: 'Ant-Baseline',
    2: 'Ant-coldstart',
    3: 'Ant-BP',
    4: 'Ant-BP-mirror',
    5: 'Ant-ideal',
    6: 'SP-BP',
}

debug = FLAGS.debug
std = 0.01
T = FLAGS.T
TV = FLAGS.TV
cf_radius = FLAGS.radius
datapath = FLAGS.datapath
ph_diff = FLAGS.ph_diff
print(f'datapath is {datapath}')
val_mat_names = sorted(os.listdir(datapath))
use_gnn = False


def get_list_of_integers(flag_str):
    strs = flag_str.split(',')
    ints = [int(i) for i in strs]
    text = '-'.join(strs)
    return ints, text, len(ints)


schemes, schemes_txt, scheme_len = get_list_of_integers(FLAGS.schemes)
sizes, sizes_txt, sizes_len = get_list_of_integers(FLAGS.sizes)

link_rate_max = 42  # 42
link_rate_min = 10  # 10
link_rate_avg = (link_rate_max + link_rate_min) / 2
arrival_max = 1.0  # 1.0
arrival_min = 0.2  # 0.2
arrival_avg = (arrival_min + arrival_max) / 2
burst_cutoff = 30
load_bursty = FLAGS.lb
load_streaming = FLAGS.ls
pburst = FLAGS.pburst  # 0.1, 0.2, 0.3, 0.4, 0.5 (0.2 and 0.5 are done already)
robust_test = FLAGS.robust_test

if pburst == 0.:
    bursty_info = 'streaming_s{}'.format(load_streaming)
else:
    bursty_info = 'mixed_pb{}_s{}_b{}'.format(pburst, load_streaming, load_bursty)

# Create fig folder if not exist
modeldir = os.path.join(FLAGS.root, "model")
if not os.path.isdir(modeldir):
    os.mkdir(modeldir)

log_dir = os.path.join(FLAGS.root, "log")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

pkl_dir = os.path.join(FLAGS.root, "pkl")
if not os.path.isdir(pkl_dir):
    os.mkdir(pkl_dir)

# Get a list of available GPUs
if use_gnn:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Set the number of GPUs to use
    num_gpus = len(gpus)
    # Set up a MirroredStrategy to use all available GPUs
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:%d" % i for i in range(num_gpus)])
    else:
        strategy = tf.distribute.get_strategy()  # default strategy

    agent = GDPGAgent(FLAGS, 5000)
    actor_model = os.path.join(modeldir, 'model_ChebConv_{}_a{}_actor'.format(FLAGS.training_set, 5, 5))
    try:
        # Define and compile your model within the strategy scope
        with strategy.scope():
            agent.load(actor_model)
    except:
        print("unable to load {}".format(actor_model))
else:
    print("skip loading GNN")

output_dir = FLAGS.out
# csvfiles = {}
# for opt in opts:
#     if opt == 0:
#         output_csv = os.path.join(output_dir,
#                                   f'test_{datapath.split("/")[-1]}_T_{T}_ir_{cf_radius:.1f}_opts_{schemes_txt}_link{link_rate_avg}_function_{FLAGS.function}_exploration_{FLAGS.exploration_rate}_decay_{FLAGS.decay}_unit_{FLAGS.unit}_init_{FLAGS.init}_warmopt_{warmopt}_mixed_{bursty_info}_streamingrate_{ar_multiplier_streaming}_approx_{approx}_robusttest_{robust_test}_bias_{bias}_multigraph_{multigraph}_opts.csv')
#     else:
#         output_csv = os.path.join(output_dir,
#                                   f'test_{datapath.split("/")[-1]}_T_{T}_ir_{cf_radius:.1f}_opts_{schemes_txt}_link{link_rate_avg}_mixed_{bursty_info}_streamingrate_{ar_multiplier_streaming}_approx_{approx}_robusttest_{robust_test}_bias_{bias}_multigraph_{multigraph}_opts.csv')
#     csvfiles[opt] = output_csv
output_csv = os.path.join(output_dir,
                          f'test_{datapath.split("/")[-1]}_ir_{cf_radius:.1f}_opts_{schemes_txt}_link_{link_rate_avg}_pb_{pburst}_ls_{load_streaming}_lb_{load_bursty}_robust_{robust_test}_T_{T}_TV_{TV}.csv')
if os.path.isfile(output_csv):
    df_res = pd.read_csv(output_csv, index_col=False)
else:
    df_res = pd.DataFrame(
        columns=[
            'filename', 'seed', 'num_nodes', 'm', 'T', 'cf_radius', 'cf_degree', 'f_case', 'num_flows',
            'flow_rate', 'cutoff', 'src', 'dst',
            'function', 'exploration_rate', 'decay_rate', 'unit', 'not_going_back', 'ph_diff',
            'opt', 'Algo', 'physical', 'z',
            'src_delay_raw', 'src_jitter_raw', 'delivery_raw', 'active_links', 'cnt_out_raw', 'cnt_in_raw',
            'runtime',
        ]
    )


def get_directional_links(adj, link_list, rates):
    assert len(link_list) == rates.size
    rows, cols = adj.nonzero()
    edges = adj.indices
    rates_direct = np.zeros((edges.shape[0], 1))
    for i in range(edges.shape[0]):
        src, dst = rows[i], cols[i]
        if (src, dst) in link_list:
            rates_direct[i, 0] = rates[link_list.index((src, dst))]
        elif (dst, src) in link_list:
            rates_direct[i, 0] = rates[link_list.index((dst, src))]
        else:
            pass
    return rates_direct


# Define tensorboard
# agent.log_init()
# graph_configs = 0
# tt = 100
# graphs_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
for id in range(len(val_mat_names)):
    filepath = os.path.join(datapath, val_mat_names[id])
    mat_contents = sio.loadmat(filepath)
    net_cfg = mat_contents['network'][0, 0]
    # link_rates = mat_contents["link_rate"][0]
    flows_cfg = mat_contents["flows"][0]
    pos_c = mat_contents["pos_c"]

    seed = int(net_cfg['seed'].flatten()[0])
    NUM_NODES = int(net_cfg['num_nodes'].flatten()[0])
    m = net_cfg['m'].flatten()[0]

    if NUM_NODES not in sizes:
        continue

    if debug:
        print(f' we have {NUM_NODES} nodes')

    for f_case in range(10):
        # Configuration
        if FLAGS.gtype.lower() == 'poisson':
            bp_env = BackpressureAnt(NUM_NODES, T, seed, m, pos_c, cf_radius=cf_radius, gtype=filepath)
        else:
            bp_env = BackpressureAnt(NUM_NODES, T, seed, m, pos_c, cf_radius=cf_radius, gtype=FLAGS.gtype)
        if not bp_env.connected:
            print("Unconnected {}".format(val_mat_names[id]))
            continue

        # bp_env.queues_init()

        # Generate random flows
        np.random.seed(seed * 10 + f_case)
        # flows_perc = np.random.randint(15, 30)
        flows_perc = np.random.randint(30, 50)
        num_flows = round(flows_perc / 100 * bp_env.num_nodes)
        nodes = bp_env.graph_c.nodes()
        num_arr = np.random.permutation(nodes)

        srcs = []
        dsts = []
        cutoffs = []
        flows = []

        fll = 0
        for fidx in range(num_flows):
            src = num_arr[2 * fidx]
            dst = num_arr[2 * fidx + 1]
            cutoff = -1
            ar_multiplier = load_streaming
            if np.random.uniform(0, 1) < pburst:
                fll += 1
                cutoff = burst_cutoff
                ar_multiplier = load_bursty
            flow = {'src': src, 'dst': dst, 'ar': ar_multiplier, 'cut': cutoff}
            flows.append(flow)
            srcs.append(src)
            dsts.append(dst)
            cutoffs.append(cutoff)
        print(f'out of {num_flows} flows {fll} of them are bursty')

        arrival_rates = np.random.uniform(arrival_min, arrival_max, (num_flows,))
        link_rates = np.random.uniform(link_rate_min, link_rate_max, size=(bp_env.num_links,))

        # print(filepath)

        # shortest_paths = np.zeros((NUM_NODES, NUM_NODES))
        # bias_matrix = np.zeros_like(bp_env.queue_matrix)
        # cali_const = np.nanmean(link_shares)
        # cali_const = 0.1
        cali_const = 1.0 / link_rate_avg

        for scheme in schemes:
            opts = [None, None]
            bias = True  # only for virtual routing
            if scheme == 0:
                algo = lgds[scheme]
                opts = [0, 0]
                warm_opt = 'ant'
                mirror = False
                freeze = True
                bias = False
                TV = 1000
            elif scheme == 1:
                algo = lgds[scheme]
                opts = [0, 0]
                warm_opt = 'ant'
                mirror = False
                freeze = True
                TV = 1000
            elif scheme == 2:
                algo = lgds[scheme]
                opts = [None, 0]
                warm_opt = None
                mirror = True
                freeze = False
                TV = 1000
            elif scheme == 3:
                algo = lgds[scheme]
                opts = [49, 0]
                warm_opt = 'SP-BP'
                mirror = False
                freeze = True
                TV = 1000
            elif scheme == 4:
                algo = lgds[scheme]
                opts = [49, 0]
                warm_opt = 'SP-BP'
                mirror = True
                freeze = True
                TV = 1000
            elif scheme == 6:
                algo = lgds[scheme]
                opts = [None, 49]
                warm_opt = None
                mirror = True
                freeze = True
                TV = 1000
            else:
                raise ValueError("Unsupported scheme: {}".format(scheme))

            # Loop over virtual and physical routing
            for vi in range(len(opts)):
                opt = opts[vi]
                if opt is None:
                    continue
                # Ant configuration
                func = FLAGS.function
                exploration_rate = FLAGS.exploration_rate
                decay = FLAGS.decay
                unit = FLAGS.unit
                not_going_back = FLAGS.not_going_back
                ph_diff = FLAGS.ph_diff
                init = FLAGS.init

                if vi == 0:
                    cT = TV
                    # virtual routing
                    np.random.seed(seed * 10 + f_case + 20)
                    flows_vi = copy.deepcopy(flows)
                    if not mirror:
                        for flow in flows_vi:
                            flow['ar'] = load_streaming
                            flow['cut'] = -1
                    if scheme == 6:
                        exploration_rate = 0.0
                        decay = 1.0
                        unit = 0.01
                        init = 0.01
                else:
                    cT = T
                    flows_vi = copy.deepcopy(flows)
                    np.random.seed(seed * 10 + f_case)

                bp_env.t_recordings = [cT - 1]
                # configure flows and realize random instances
                start_time = time.time()
                bp_env.clear_all_flows()
                flow_rates = []
                for fidx in range(num_flows):
                    flow = flows_vi[fidx]
                    bp_env.add_flow(flow['src'], flow['dst'], rate=flow['ar'] * arrival_rates[fidx], cutoff=flow['cut'])
                    flow_rates.append(flow['ar'] * arrival_rates[fidx])

                bp_env.flows_init()
                bp_env.flows_reset()
                bp_env.links_init(link_rates)
                bp_env.queues_init()
                bp_env.pheromone_init(decay=decay, unit=unit, init=init)

                # Handle load_pherom and save_pherom
                if robust_test and scheme in [1, 3, 4, 5]:
                    l1 = [0.5, 1, 2]
                    l2 = [5, 10, 20]
                    if load_bursty in l1:
                        if mirror:
                            b_info = 'mixed_pb{}_s{}_b{}'.format(pburst, 1.0, 1.0)
                        else:
                            b_info = 'streaming_s{}'.format(1.0)
                        # b_info = (burst_cutoff, 1., pburst)
                    elif load_bursty in l2:
                        if mirror:
                            b_info = 'mixed_pb{}_s{}_b{}'.format(pburst, 1.0, 10.0)
                        else:
                            b_info = 'streaming_s{}'.format(1.0)
                        # b_info = (burst_cutoff, 10., pburst)
                    else:
                        raise ValueError("unsupported {} for robustness test".format(load_bursty))
                else:
                    if mirror:
                        b_info = bursty_info
                    else:
                        b_info = 'streaming_s{}'.format(load_streaming)

                # physical routing, configure warm start policy
                if opts[0] is not None:
                    if opts[0] > 0:
                        items = 'sp_bp_{}'.format(opts[0])
                    else:
                        items = '{}_{:.3f}_{:.1f}_{:.2f}_{}_{}'.format(func, decay, exploration_rate, unit, not_going_back, ph_diff)

                    success = bp_env.load_pherom(
                        opts[0],
                        val_mat_names[id], f_case, items,
                        TV - 1,
                        bursty=b_info,
                        test=vi == 0,
                    )
                    if success and vi == 0:
                        print(f"skip virtual routing as policy file already exists: {success}")
                        continue
                    elif not success and robust_test:
                        print(f"skip robustness test since I am unable to load policy: {success}")
                        continue
                    else:
                        print(f'warm start {success}: min-max: {np.min(bp_env.pheromones), np.max(bp_env.pheromones)}')

                if freeze and vi > 0:
                    bp_env.freeze_pherom()

                # skip test if test results of current case already exist
                if not df_res.query(
                        "@val_mat_names[{}] == filename and \
                        @seed == seed and \
                        @m == m and \
                        @cT == T and \
                        @algo == Algo and \
                        @cf_radius == cf_radius and \
                        @opt == opt and \
                        @vi == physical and \
                        @f_case == f_case and \
                        @func == function and \
                        @exploration_rate == exploration_rate and \
                        @decay == decay_rate and \
                        @unit == unit and \
                        @not_going_back == not_going_back and \
                        @ph_diff == ph_diff and \
                        @NUM_NODES == num_nodes".format(id)
                ).empty:
                    print("skip test case: {}, {}, {}, {}, physical {}".format(val_mat_names[id], f_case, algo, cT, vi))
                    continue

                # delay_est = link_rate_avg * np.ones_like(link_rates)
                delay_mtx = np.zeros_like(bp_env.queue_matrix)

                # computing bias
                delay_est = np.divide(np.nanmean(link_rates) ** 2, link_rates)
                for link, delay in zip(bp_env.link_list, delay_est):
                    src, dst = link
                    bp_env.graph_c[src][dst]["delay"] = delay
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                bias_matrix = shortest_paths
                bias_vector = bp_env.bias_diff(bias_matrix)
                link_bias_vec = bias_vector * (link_rate_avg / np.min(delay_est))
                if bias and vi == 0:
                    # to see how Ant-Baseline perform without bias=False
                    link_bias = link_bias_vec
                else:
                    link_bias = None

                # routing simulation
                active_links = np.zeros((T,))
                # start_time = time.time()

                for t in range(T):
                    if vi == 0 and t >= TV:
                        break

                    bp_env.pkt_arrival(t)

                    # Commodity and W computation
                    if opt > 0:
                        W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix_exp, 0.0, link_bias_vec)
                        W_amp[C_opt == -1] = 0.0
                        bp_env.opt_comd_mtx[:, t] = C_opt
                    else:
                        W_amp, W_sign = bp_env.ph_routing(
                            t, func=func, exploration_rate=exploration_rate,
                            not_going_back=not_going_back,
                            link_bias=link_bias,
                            ph_diff=ph_diff,
                        )

                    bp_env.W[:, t] = W_amp
                    bp_env.WSign[:, t] = W_sign

                    active_links[t] = np.count_nonzero(W_amp)

                    # Greedy Maximal Scheduling & Transmission
                    mwis = bp_env.scheduling(bp_env.W[:, t] * bp_env.link_rates[:, t])
                    if opt > 0:
                        bp_env.transmission(t, mwis)
                        # Collect number of packets in networks for each flow, only for BP
                        for fidx in range(bp_env.num_flows):
                            bp_env.flow_pkts_in_network[fidx, t] = np.sum(
                                bp_env.queue_matrix[:, bp_env.flows[fidx].dest_node])
                    else:
                        bp_env.transmission_ph(t, mwis)

                # print("Average packets in network:")
                # print(round(bp_env.flow_pkts_in_network.mean(), 2))
                cnt_in, cnt_out, delay_e2e, delay_e2e_raw, jitter_e2e, undeliver = bp_env.collect_delay(opt, cT)
                src_delay_mean = np.nanmean(delay_e2e)
                src_delay_max = np.nanmax(delay_e2e)
                src_delay_std = np.nanstd(delay_e2e)
                src_jitter_mean = np.nanmean(jitter_e2e)
                src_jitter_max = np.nanmax(jitter_e2e)
                src_jitter_std = np.nanstd(jitter_e2e)
                delivery_raw = np.divide(cnt_out.astype(float), cnt_in.astype(float))
                delivery_mean = np.nanmean(delivery_raw)
                delivery_max = np.nanmax(delivery_raw)
                delivery_std = np.nanstd(delivery_raw)
                runtime = time.time() - start_time

                if vi == 0:
                    print(f'we save pheromones for opt {opt}')
                    bp_env.save_pherom(opt, val_mat_names[id], f_case, items, b_info)

                if debug:
                    print("{}: n {}, f {}, s {}, cf_deg {:.3f}, c {}, ".format(val_mat_names[id], NUM_NODES, num_flows,
                                                                               seed, bp_env.mean_conflict_degree,
                                                                               f_case),
                          "opt {}, runtime {:.2f}, links {:.1f}".format(opt, runtime, np.nanmean(active_links)),
                          "Delay: mean {:.3f}, max {:.3f}, std {:.3f}".format(src_delay_mean, src_delay_max,
                                                                              src_delay_std),
                          "Jitter: mean {:.3f}, max {:.3f}, std {:.3f}".format(src_jitter_mean, src_jitter_max,
                                                                               src_jitter_std),
                          "Delivery: mean {:.3f}, max {:.3f}, std {:.3f}".format(delivery_mean, delivery_max,
                                                                                 delivery_std),
                          # "cali: {:.3f}".format(np.nanmean(cali_const)),
                          # "function: {}".format(func),
                          # "exploration_rate: {:.3f}".format(exploration_rate),
                          # "decay: {:.3f}".format(decay),
                          # "unit: {:.3f}".format(unit),
                          # "not_going_back: {}".format(not_going_back),
                          )

                result = {
                    "filename": val_mat_names[id],
                    "seed": seed,
                    "num_nodes": NUM_NODES,
                    "m": m,
                    "T": cT,
                    "cf_radius": cf_radius,
                    "cf_degree": bp_env.mean_conflict_degree,
                    "opt": opt,
                    "Algo": algo,
                    "f_case": f_case,
                    "physical": vi,
                    "num_flows": bp_env.num_flows,
                    "src_delay_raw": delay_e2e,
                    "src_jitter_raw": jitter_e2e,
                    "delivery_raw": delivery_raw,
                    "cnt_out_raw": cnt_out,
                    "cnt_in_raw": cnt_in,
                    # "undeliver": undeliver,
                    "flow_rate": flow_rates,
                    "cutoff": cutoffs,
                    "src": srcs,
                    "dst": dsts,
                    "runtime": runtime,
                    "active_links": np.nanmean(active_links),
                    "z": np.nanmean(cali_const),
                    "function": func,
                    "exploration_rate": exploration_rate,
                    "decay_rate": decay,
                    "unit": unit,
                    "not_going_back": not_going_back,
                    "ph_diff": ph_diff,
                }
                # df_res = df_res.append(result, ignore_index=True)
                new_row = pd.DataFrame(result)
                df_res = pd.concat([df_res, new_row], ignore_index=True)
                df_res.to_csv(output_csv, index=False)

        # only run 1 case on one graph for debugging
        if debug:
            break
    if debug:
        break

print(f'Done')