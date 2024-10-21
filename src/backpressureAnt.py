
# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import queue
import sys
import os
import time
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scipy.io as sio
import sparse
np.set_printoptions(threshold=np.inf)
# Import utility functions
from util import *



class BackpressureAnt:
    def __init__(self, num_nodes, T, seed=3, m=2, pos=None, cf_radius=0.0, gtype='ba'):
        self.num_nodes = int(num_nodes)
        self.T = int(T)
        self.t_recordings = [self.T-1]
        self.seed = int(seed) # other format such as int64 won't work
        self.m = int(m)
        self.gtype = gtype.lower()
        self.trace = True
        self.cf_radius = cf_radius
        self.case_name = 'AntBP_seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, self.gtype)
        if self.gtype == 'ba':
            graph_c = nx.barabasi_albert_graph(self.num_nodes, self.m, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'grp':
            graph_c = nx.gaussian_random_partition_graph(self.num_nodes, 15, 3, 0.4, 0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'ws':
            graph_c = nx.connected_watts_strogatz_graph(self.num_nodes, k=6, p=0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'er':
            graph_c = nx.fast_gnp_random_graph(self.num_nodes, 15.0/float(self.num_nodes), seed=self.seed)  # Conectivity graph
        elif '.mat' in self.gtype:
            postfix = self.gtype.split('/')[-1]
            postfix = postfix.split('.')[0]
            self.case_name = 'seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, postfix)
            try:
                mat_contents = sio.loadmat(self.gtype)
                adj = mat_contents['adj'].todense()
                pos = mat_contents['pos_c']
                graph_c = nx.from_numpy_array(adj)
            except:
                raise RuntimeError("Error creating object, check {}".format(self.gtype))
        else:
            raise ValueError("unsupported graph model for connectivity graph")
        self.connected = nx.is_connected(graph_c)
        self.graph_c = graph_c
        self.node_positions(pos)
        self.box = self.bbox()
        self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
        self.adj_c = nx.adjacency_matrix(self.graph_c)
        self.num_links = len(self.graph_i.nodes)
        self.num_di_links = 2 * self.num_links
        self.link_list = list(self.graph_i.nodes)
        self.edge_maps = np.zeros((self.num_di_links,), dtype=int)
        self.edge_maps_rev = np.zeros((self.num_di_links,), dtype=int)
        self.link_mapping()
        if cf_radius > 0.5:
            self.add_conflict_relations(cf_radius)
        else:
            self.adj_i = nx.adjacency_matrix(self.graph_i)
        self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
        self.fid_cmd_map = np.zeros((self.num_nodes,), dtype=int)
        self.clear_all_flows()
        self.pheromone_freezed = False
        self.queue_lengths = np.zeros((self.num_nodes, self.num_nodes), dtype=float) # repurpose for ph_routing
        if not self.trace:
            self.delivery = sparse.COO(np.zeros((self.num_nodes, self.num_nodes, self.num_nodes), dtype=float))

    def random_walk(self, ss=0.1, n=10):
        disconnected = True
        while disconnected:
            mask = np.random.choice(np.arange(0, self.num_nodes), size=n, replace=False)
            d_pos = np.random.normal(0, ss, size=(n, 2))
            pos_c_np = self.pos_c_np
            pos_c_np[mask, :] += d_pos
            b_min = np.min(self.box)
            b_max = np.max(self.box)
            pos_c_np = pos_c_np.clip(b_min, b_max)
            d_mtx = distance_matrix(pos_c_np, pos_c_np)
            adj_mtx = np.zeros([self.num_nodes, self.num_nodes], dtype=int)
            adj_mtx[d_mtx <= 1.0] = 1
            np.fill_diagonal(adj_mtx, 0)
            graph_c = nx.from_numpy_array(adj_mtx)
            self.connected = nx.is_connected(graph_c)
            disconnected = not self.connected
        return graph_c, pos_c_np

    # def topology_update(self, graph_c, pos_c_np):
    #     self.connected = nx.is_connected(graph_c)
    #     if not self.connected:
    #         raise RuntimeError("graph is no more connected")
    #     self.graph_c = graph_c
    #     self.node_positions(pos_c_np)
    #     self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
    #     self.adj_c = nx.adjacency_matrix(self.graph_c)
    #     self.num_links = len(self.graph_i.nodes)
    #     self.num_di_links = 2 * self.num_links
    #     link_list_old = self.link_list
    #     pheromons_old = self.pheromones
    #     self.pheromones = np.zeros((self.num_di_links, self.num_nodes), dtype=float)
    #     self.pheromones_vis = np.zeros((self.num_di_links, self.num_nodes, 1), dtype=float)
    #     self.link_list = list(self.graph_i.nodes)
    #     new_links_map = np.zeros((self.num_links,), dtype=int)
    #     for i in range(self.num_links):
    #         e0, e1 = self.link_list[i]
    #         if (e0, e1) in link_list_old:
    #             j = link_list_old.index((e0, e1))
    #             self.pheromones[i]=pheromons_old[j]
    #             self.pheromones[i+self.num_links]=pheromons_old[j+len(link_list_old)]
    #         elif (e1, e0) in link_list_old:
    #             j = link_list_old.index((e1, e0))
    #             self.pheromones[i]=pheromons_old[j]
    #             self.pheromones[i+self.num_links]=pheromons_old[j+len(link_list_old)]
    #         else:
    #             j = -1
    #         new_links_map[i] = j
    #     self.edge_maps = np.zeros((self.num_di_links,), dtype=int)
    #     self.edge_maps_rev = np.zeros((self.num_di_links,), dtype=int)
    #     self.link_mapping()
    #     if self.cf_radius > 0.5:
    #         self.add_conflict_relations(self.cf_radius)
    #     else:
    #         self.adj_i = nx.adjacency_matrix(self.graph_i)
    #     self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
    #     self.W = np.zeros((self.num_links, self.T))
    #     self.WSign = np.ones((self.num_links, self.T))
    #     self.opt_comd_mtx = -np.ones((self.num_links, self.T), dtype=int)
    #     self.link_comd_cnts = np.zeros((self.num_links, self.num_nodes))
    #     self.di_link_comd_cnts = np.zeros((self.num_di_links, self.num_nodes))
    #     self.pkt_vis = np.zeros((self.num_di_links, self.num_nodes, self.T))
    #     #print(f'pheroms are {np.array_equal(self.pheromones, pheromons_old)}')
    #     return new_links_map

    class Flow:
        def __init__(self, source_node, arrival_rate, dest_node):
            self.source_node = source_node
            self.arrival_rate = arrival_rate
            self.dest_node = dest_node
            self.cut_off = -1

    def node_positions(self, pos):
        if pos is None:
            pos_file = os.path.join("..", "pos", "graph_c_pos_{}.p".format(self.case_name))
            if not os.path.isfile(pos_file):
                pos_c = nx.spring_layout(self.graph_c)
                with open(pos_file, 'wb') as fp:
                    pickle.dump(pos_c, fp, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(pos_file, 'rb') as fp:
                    pos_c = pickle.load(fp)
        elif isinstance(pos, str) and pos == 'new':
            pos_c = nx.spring_layout(self.graph_c)
        elif isinstance(pos, np.ndarray):
            pos_c = dict(zip(list(range(self.num_nodes)), pos))
        else:
            raise ValueError("unsupported pos format in backpressure object initialization")
        self.pos_c = pos_c

    def bbox(self):
        pos_c = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            pos_c[i, :] = self.pos_c[i]
        self.pos_c_np = pos_c
        return [np.amin(pos_c[:,0])-0.05, np.amax(pos_c[:,1])+0.05, np.amin(pos_c[:,1])-0.12, np.amax(pos_c[:,1])+0.05]

    def add_conflict_relations(self, cf_radius):
        """
        Adding conflict relationship between links whose nodes are within cf_radius * median_link_distance
        :param cf_radius: multiple of median link distance
        :return: None (modify self.adj_i, and self.graph_i inplace)
        """
        pos_c_vec = np.zeros((self.num_nodes, 2))
        for key, item in self.pos_c.items():
            pos_c_vec[key, :] = item
        dist_mtx = distance_matrix(pos_c_vec, pos_c_vec)
        rows, cols = np.nonzero(self.adj_c)
        link_dist = dist_mtx[rows, cols]
        median_dist = np.nanmedian(link_dist)
        intf_dist = cf_radius * median_dist
        for link in self.link_list:
            src, dst = link
            intf_nbs_s, = np.where(dist_mtx[src, :] < intf_dist)
            intf_nbs_d, = np.where(dist_mtx[dst, :] < intf_dist)
            intf_nbs = np.union1d(intf_nbs_s, intf_nbs_d)
            for v in intf_nbs:
                _, nb2hop = np.nonzero(self.adj_c[v])
                for u in nb2hop:
                    if {v, u} == {src, dst}:
                        continue
                    elif (v, u) in self.link_list:
                        self.graph_i.add_edge((v, u), (src, dst))
                    elif (u, v) in self.link_list:
                        self.graph_i.add_edge((u, v), (src, dst))
                    else:
                        pass
                        # raise RuntimeError("Something wrong with adding conflicting edge")
        self.adj_i = nx.adjacency_matrix(self.graph_i)

    def link_mapping(self):
        # Mapping between links in connectivity graph and nodes in conflict graph
        j = 0
        for e0, e1 in self.graph_c.edges:
            try:
                i = self.link_list.index((e0, e1))
            except:
                i = self.link_list.index((e1, e0))
            # Link direction A
            self.edge_maps[j] = i
            self.edge_maps_rev[i] = j
            # Link direction B
            self.edge_maps[j + self.num_links] = i + self.num_links
            self.edge_maps_rev[i + self.num_links] = j + self.num_links
            j += 1

    def add_flow(self, src, dst, rate=2, cutoff=-1):
        fi = self.Flow(src, rate, dst)
        if 0 < cutoff < self.T:
            fi.cut_off = int(cutoff)
        else:
            fi.cut_off = self.T
        self.flows.append(fi)
        self.src_nodes.append(src)
        self.dst_nodes.append(dst)
        self.num_flows = len(self.flows)

    def clear_all_flows(self):
        self.flows = []
        self.num_flows = 0
        self.src_nodes = []
        self.dst_nodes = []
        self.fid_cmd_map.fill(np.nan)

    def flows_init(self):
        self.flows_sink_departures = np.zeros((self.num_flows, self.T), dtype=int)
        self.flows_arrivals = np.zeros((self.num_flows, self.T), dtype=int)
        self.flow_pkts_in_network = np.zeros((self.num_flows, self.T), dtype=int)
        np.random.seed(self.seed)
        for fidx in range(self.num_flows):
            arrival_rate = self.flows[fidx].arrival_rate
            T = int(self.flows[fidx].cut_off)
            self.flows_arrivals[fidx, 0:T] = np.random.poisson(arrival_rate, size=(T,))
            self.fid_cmd_map[self.flows[fidx].dest_node] = fidx

    def flows_reset(self):
        self.flows_sink_departures = np.zeros((self.num_flows, self.T), dtype=int)
        self.flow_pkts_in_network = np.zeros((self.num_flows, self.T), dtype=int)

    def freeze_pherom(self):
        self.pheromone_freezed = True

    def unfreeze_pherom(self):
        self.pheromone_freezed = False

    def links_init(self, rates, std=2):
        if hasattr(rates, '__len__'):
            assert len(rates) == self.num_links
            stds = std * np.ones_like(rates)
        else:
            stds = std
        link_rates = np.zeros((self.num_links, self.T))
        for t in range(self.T):
            link_rates[:, t] = np.clip(np.random.normal(rates, stds), 0, rates + 3 * std)
        self.link_rates = np.round(link_rates)

    # def link_failure(self, link_bias):
    #     deleted_link = np.random.choice(self.link_list)
    #     deleted_link_index = self.link_list.index(deleted_link)
    #     self.link_list.remove(deleted_link)
    #     self.pheromones = np.delete(self.pheromones, deleted_link_index+self.num_links, axis=0)
    #     self.pheromones = np.delete(self.pheromones, deleted_link_index, axis=0)
    #     self.pheromones_vis = np.delete(self.pheromones_vis, deleted_link_index+self.num_links, axis=0)
    #     self.pheromones_vis = np.delete(self.pheromones_vis, deleted_link_index, axis=0)
    #     link_bias = np.delete(link_bias, deleted_link_index, axis=0)
    #     return link_bias

    def queues_init(self):
        # Initialize system state
        self.queue_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.W = np.zeros((self.num_links, self.T))
        self.WSign = np.ones((self.num_links, self.T))
        self.opt_comd_mtx = -np.ones((self.num_links, self.T), dtype=int)
        self.link_comd_cnts = np.zeros((self.num_links, self.num_flows))
        self.di_link_comd_cnts = np.zeros((self.num_di_links, self.num_flows))
        self.pkt_vis = np.zeros((self.num_di_links, self.num_flows, self.T))
        self.backlog = {}
        self.backlog_ph = {}
        for i in range(self.num_nodes):
            backlog_i = {}
            backlog_i_ph = {}
            for j in range(self.num_nodes + 1):
                # Each queue holds pkt from node i to node j
                # queue backlog_ph[i][i] holds undetermined pkts
                # queue backlog_ph[i][self.num_nodes] holds pkt reach its destination node i
                qi = queue.Queue()
                backlog_i[j] = qi
                qi_ph = queue.Queue()
                backlog_i_ph[j] = qi_ph
            self.backlog[i] = backlog_i
            self.backlog_ph[i] = backlog_i_ph
        self.queue_lengths = np.zeros((self.num_nodes, self.num_nodes))
        self.HOL_t0 = np.zeros((self.num_nodes, self.num_nodes))
        self.HOL_delay = np.zeros((self.num_nodes, self.num_nodes))
        self.SJT_delay = np.zeros((self.num_nodes, self.num_nodes))

    def pheromone_init(self, decay=0.97, unit=0.01, init=0.5):
        self.phmns_decay = decay
        self.phmns_unit = unit
        # The shape of pheromones is total number of directional links x total number of destinations
        link_rate_avg = np.nanmean(self.link_rates)
        if decay < 1.0:
            max_val = link_rate_avg * unit / (1.0 - decay)
        else:
            max_val = 1.0
        self.pheromones = init * max_val * np.ones((self.num_di_links, self.num_flows), dtype=float)
        self.pheromones_vis = np.zeros((self.num_di_links, self.num_flows, 1), dtype=float)
        self.queue_matrix_exp = np.zeros_like(self.queue_matrix)
        self.phmns_exp = 1 + (1 - decay)

    def bias_diff(self, bias_matrix):
        link_bias = np.zeros((self.num_links, self.num_nodes), dtype=float)
        for lidx in range(self.num_links):
            src, dst = self.link_list[lidx]
            bdiff = bias_matrix[src, :] - bias_matrix[dst, :]
            link_bias[lidx, :] = bdiff
        return link_bias

    def pkt_arrival(self, t):
        for fidx in range(self.num_flows):
            flow = self.flows[fidx]
            src = flow.source_node
            dst = flow.dest_node
            self.queue_matrix[src, dst] += self.flows_arrivals[fidx, t]
            self.queue_lengths[src, src] += self.flows_arrivals[fidx, t]
            self.queue_matrix_exp[src, dst] += self.flows_arrivals[fidx, t]
            for i in range(self.flows_arrivals[fidx, t]):
                self.backlog[src][dst].put((t, t))
                # pheromone routing, queue to itself means undetermined
                self.backlog_ph[src][src].put((t, t, dst, None))

    def update_HOL_matrix(self, t):
        '''should be run after packet arrivals'''
        if self.trace:
            for src in range(self.num_nodes):
                for cmd in self.dst_nodes:
                    if self.backlog[src][cmd].empty() or (src == cmd):
                        self.HOL_delay[src][cmd] = 0
                    else:
                        pkt = self.backlog[src][cmd].queue[0]
                        t0, t1 = pkt
                        self.HOL_t0[src][cmd] = t0
                        self.HOL_delay[src][cmd] = t - t1

    def update_SJT_matrix(self, t):
        '''should be run after packet arrivals'''
        if self.trace:
            for src in range(self.num_nodes):
                for cmd in self.dst_nodes:
                    self.SJT_delay[src][cmd] = 0
                    if self.backlog[src][cmd].empty() or (src == cmd):
                        pass
                    else:
                        for pkt in self.backlog[src][cmd].queue:
                            t0, t1 = pkt
                            self.SJT_delay[src][cmd] += t - t1

    def commodity_selection(self, queue_mtx, mbp=0.0, link_phmn=None):
        W_amp = np.zeros((self.num_links,), dtype=float)
        W_sign = np.ones((self.num_links,), dtype=float)
        comds = -np.ones((self.num_links,), dtype=int)
        j = 0
        for link in self.link_list:
            wts_link = queue_mtx[link[0], self.dst_nodes] - queue_mtx[link[1], self.dst_nodes]
            directions = np.sign(wts_link)
            # find out the source nodes
            ql_src_vec = np.where(directions > 0.0,
                                  self.queue_matrix[link[0], self.dst_nodes],
                                  self.queue_matrix[link[1], self.dst_nodes])
            # create a mask that source nodes has more than 1 packet to transmit
            ql_mask = np.where(ql_src_vec > 0.1, np.ones_like(self.dst_nodes), np.zeros_like(self.dst_nodes))
            if link_phmn is None:
                wts_link = np.multiply(wts_link, ql_mask)
            else:
                wts_link = np.multiply(wts_link + link_phmn[j, self.dst_nodes], ql_mask)
            cmd = np.argmax(abs(wts_link))
            W_sign[j] = np.sign(wts_link[cmd])
            W_amp[j] = np.amax([abs(wts_link[cmd]) - mbp, 0])
            comds[j] = self.dst_nodes[cmd] if np.amax(abs(wts_link)) > 0.0 else -1
            # if W_sign[j] == 1:
            #     ql_src = self.queue_matrix[link[0], self.dst_nodes[cmd]]
            # else:
            #     ql_src = self.queue_matrix[link[1], self.dst_nodes[cmd]]
            # comds[j] = self.dst_nodes[cmd] if (np.amax(abs(wts_link)) > 0 and ql_src > 0) else -1
            j += 1
        return W_amp, W_sign, comds

    def ph_routing(self, t, func='proportional', exploration_rate=0.0, not_going_back=False, link_bias=None, ph_diff=False):
        W_amp = np.zeros((self.num_links,), dtype=float)
        W_sign = np.ones((self.num_links,), dtype=float)
        for v in range(self.num_nodes):
            n_undecided = self.backlog_ph[v][v].qsize()
            if n_undecided == 0:
                continue
            _, nb_set = np.nonzero(self.adj_c[v])
            rlinks = -np.ones_like(nb_set)
            dlinks = -np.ones_like(nb_set)
            for j in range(len(nb_set)):
                u = nb_set[j]
                if (v, u) in self.link_list:
                    i = self.link_list.index((v, u))
                    rlinks[j] = i + self.num_links
                    dlinks[j] = i
                elif (u, v) in self.link_list:
                    i = self.link_list.index((u, v))
                    rlinks[j] = i
                    dlinks[j] = i + self.num_links
                else:
                    pass
                #dlinks[j] = i
            for j in range(n_undecided):
                pkt = self.backlog_ph[v][v].get_nowait()
                if pkt is None:
                    raise RuntimeError("Ph_routing Backlog error node: {}".format(v))
                t0, t1, cmd, last = pkt
                cmd_fid = self.fid_cmd_map[cmd]
                # Drop packets trapped in the network for too long
                # if t1 - t0 > 100:
                #     continue
                if last is not None and not_going_back:
                    # do something here to nb_set to avoid pkt being sent back to its last node
                    remov = np.where(nb_set == last)[0]
                    dlinks = np.delete(dlinks, remov)
                    rlinks = np.delete(rlinks, remov)
                    nb_set = np.delete(nb_set, remov)
                # select outbound node
                if nb_set.size < 1.0:
                    continue

                if ph_diff:
                    phs = self.pheromones[dlinks, cmd_fid] - self.pheromones[rlinks, cmd_fid]
                else:
                    phs = self.pheromones[dlinks, cmd_fid]

                if link_bias is not None:
                    dilink_bias = np.vstack((link_bias, -link_bias))
                    phs += dilink_bias[dlinks, cmd]

                rand_value = np.random.rand()
                if rand_value < exploration_rate:
                    probs = np.ones_like(nb_set)/float(nb_set.size)
                else:
                    if func == 'proportional':
                        phs[phs < 0] = 0
                        if np.all(phs == 0):
                            probs = np.ones_like(nb_set) / float(nb_set.size)
                        else:
                            probs = phs / np.sum(phs)
                    elif func == 'softmax':
                        probs = softmax(phs, alpha=1)
                    elif func == 'powerlaw':
                        probs = power_law_probabilities(phs, beta=2)
                    elif func == 'rankbased':
                        probs = rank_based_probabilities(phs)
                    elif func == 'elu':
                        probs = elu(phs)
                        if np.all(probs == 0):
                            probs = np.ones_like(nb_set) / float(nb_set.size)
                        else:
                            probs = probs / np.sum(probs)
                has_nan_values = np.isnan(probs).any()
                if has_nan_values:
                    continue
                u = np.random.choice(nb_set, p=probs)
                self.backlog_ph[v][u].put((t0, t, cmd, v))
                self.queue_lengths[v, u] += 1
                self.queue_lengths[v, v] -= 1

        # routing is finished up to here, but we also update the following values to be compatible with BP routing
        for j in range(self.num_links):
            v, u = self.link_list[j]
            wts_link = max(self.queue_lengths[v, u], self.queue_lengths[u, v])
            W_sign[j] = np.sign(self.queue_lengths[v, u] - self.queue_lengths[u, v] + 0.01)
            # create a mask that source nodes has more than 1 packet to transmit
            ql_mask = 1 if wts_link > 0.1 else 0
            wts_link = np.multiply(wts_link, ql_mask)
            W_amp[j] = float(wts_link)

        return W_amp, W_sign

    def scheduling(self, weights, all_same=False):
        keep_index = np.argwhere(weights > 0.0)
        if all_same:
            weights[keep_index] = np.max(weights)
        wts_postive = weights[keep_index]
        # graph_small = self.graph_i
        adj = self.adj_i[keep_index.flatten(), :]
        adj = adj[:, keep_index.flatten()]
        mwis, total_wt = local_greedy_search(adj, wts_postive)
        solu = list(mwis)
        solu = keep_index[solu].flatten().tolist()
        return solu

    def transmission_ph(self, t, mwis):
        """
        Matrix formed transmission with ph_routing, It takes 0.597 seconds to run 100 time slots on graph (15) seed 3
        :param t: time
        :param mwis: list of scheduled links
        :return:
        """
        dsts = -np.ones((len(mwis),), dtype=int)
        srcs = -np.ones_like(dsts)
        schs = -np.ones_like(dsts)
        schs_di = -np.ones((len(mwis),), dtype=int)
        for idx in range(len(mwis)):
            link = mwis[idx]
            shift = 0
            if self.WSign[link, t] < 0:
                dsts[idx] = self.link_list[link][0]
                srcs[idx] = self.link_list[link][1]
                shift = self.num_links
            elif self.WSign[link, t] > 0:
                srcs[idx] = self.link_list[link][0]
                dsts[idx] = self.link_list[link][1]
            else:
                continue
            schs[idx] = link
            schs_di[idx] = link + shift
        schs_di = schs_di[schs_di != -1]
        schs = schs[schs != -1]
        dsts = dsts[dsts != -1]
        srcs = srcs[srcs != -1]
        num_pkts = np.minimum(self.queue_lengths[srcs, dsts], self.link_rates[schs, t])
        if not self.pheromone_freezed:
            self.pheromones = self.pheromones * self.phmns_decay
        for idx in range(schs.size):
            link = schs[idx]
            dlink = schs_di[idx]
            src = srcs[idx]
            dst = dsts[idx]
            num = num_pkts[idx]
            # print(f'we have link {self.link_list[link]} with source {src} and destination {dst}')
            # print(f'the queue length is {self.queue_lengths[src,dst]} and we want to send {num} packets')
            self.queue_lengths[src, dst] -= num
            for i in range(int(num)):
                pkt = self.backlog_ph[src][dst].get_nowait()
                if pkt is None:
                    raise RuntimeError("Ph_transmission: Backlog error node: {}".format(src))
                t0, t1, cmd, last = pkt
                cmd_fid = self.fid_cmd_map[cmd]
                if not self.pheromone_freezed:
                    self.pheromones[dlink, cmd_fid] += self.phmns_unit
                if dst == cmd:
                    #print(f'commodity is the same as destination')
                    fidx = self.dst_nodes.index(cmd)
                    self.flows_sink_departures[fidx, t] += 1
                    self.backlog_ph[dst][self.num_nodes].put((t0, t, cmd, src))
                else:
                    self.backlog_ph[dst][dst].put((t0, t, cmd, src))
                    self.queue_lengths[dst, dst] += 1
                self.link_comd_cnts[link, cmd_fid] += 1
                self.di_link_comd_cnts[dlink, cmd_fid] += 1
                self.pkt_vis[dlink, cmd_fid, t] += 1
        if t in self.t_recordings:
            self.pheromones_vis = np.concatenate((self.pheromones_vis, self.pheromones[..., np.newaxis]), axis=-1)

    def transmission(self, t, mwis):
        """
        Matrix formed transmission, It takes 0.597 seconds to run 100 time slots on graph (15) seed 3
        :param t: time
        :param mwis: list of scheduled links
        :return:
        """

        dsts = -np.ones((len(mwis),), dtype=int)
        srcs = -np.ones_like(dsts)
        schs = -np.ones_like(dsts)
        schs_di = -np.ones((len(mwis),), dtype=int)
        for idx in range(len(mwis)):
            link = mwis[idx]
            shift = 0
            if self.WSign[link, t] < 0:
                dsts[idx] = self.link_list[link][0]
                srcs[idx] = self.link_list[link][1]
                shift = self.num_links
            elif self.WSign[link, t] > 0:
                srcs[idx] = self.link_list[link][0]
                dsts[idx] = self.link_list[link][1]
            else:
                continue
            schs[idx] = link
            schs_di[idx] = link + shift
        schs_di = schs_di[schs_di != -1]
        schs = schs[schs != -1]
        dsts = dsts[dsts != -1]
        srcs = srcs[srcs != -1]
        opt_comds = self.opt_comd_mtx[schs, t]
        num_pkts = np.minimum(self.queue_matrix[srcs, opt_comds], self.link_rates[schs, t])
        opt_comds_fids = self.fid_cmd_map[opt_comds]
        if self.trace:
            for idx in range(len(mwis)):
                src = srcs[idx]
                dst = dsts[idx]
                num = num_pkts[idx]
                cmd = opt_comds[idx]
                if cmd == -1:
                    continue
                elif dst == cmd:
                    fidx = self.dst_nodes.index(cmd)
                    self.flows_sink_departures[fidx, t] = num
                for i in range(int(num)):
                    pkt = self.backlog[src][cmd].get_nowait()
                    if pkt is None:
                        raise RuntimeError("Backlog error node: {}, commodity: {}".format(src, cmd))
                    t0, t1 = pkt
                    self.backlog[dst][cmd].put((t0, t))
        self.queue_matrix_exp = self.queue_matrix_exp * self.phmns_exp
        queue_exp_per_pkt = np.nan_to_num(
            np.divide(self.queue_matrix_exp[srcs, opt_comds], self.queue_matrix[srcs, opt_comds]), nan=0.0)
        self.queue_matrix_exp[dsts, opt_comds] += num_pkts
        self.queue_matrix_exp[srcs, opt_comds] -= queue_exp_per_pkt * num_pkts
        self.queue_matrix_exp[self.queue_matrix_exp < 0.5] = 0.0
        self.queue_matrix[dsts, opt_comds] += num_pkts
        self.queue_matrix[srcs, opt_comds] -= num_pkts

        if not self.trace:
            coords = np.vstack([srcs, dsts, opt_comds])
            coo_pkts = sparse.COO(coords=coords, data=num_pkts, shape=(self.num_nodes, self.num_nodes, self.num_nodes))
            self.delivery += coo_pkts

        self.link_comd_cnts[schs, opt_comds_fids] += num_pkts
        self.di_link_comd_cnts[schs_di, opt_comds_fids] += num_pkts

        self.pheromones = self.pheromones * self.phmns_decay
        self.pheromones[schs, opt_comds_fids] += self.phmns_unit * np.multiply(num_pkts, self.WSign[mwis, t])
        # there are shadow commodities in SP-bias (0 packets to transmit)
        sink_true = np.logical_and(opt_comds == dsts, num_pkts > 0.1)
        sink_dsts = dsts[sink_true]
        if t in self.t_recordings:
            self.pheromones_vis = np.concatenate((self.pheromones_vis, self.di_link_comd_cnts[..., np.newaxis]), axis=-1)
        if len(sink_dsts) > 0:
            self.queue_matrix[sink_dsts, sink_dsts] = 0
            self.queue_matrix_exp[sink_dsts, sink_dsts] = 0
            if not self.trace:
                fidxs = np.zeros_like(sink_dsts)
                for sidx in range(len(sink_dsts)):
                    fidx, = np.where(self.dst_nodes == sink_dsts[sidx])
                    if len(fidx) > 0:
                        fidxs[sidx] = fidx[0]
                self.flows_sink_departures[fidxs, t] = num_pkts[sink_true]

    def update_bias_mean(self, bias_matrix):
        # step 1: find out neighbors, construct an out adj matrix
        out_adj = np.zeros((self.num_nodes, self.num_nodes))
        bias_matrix_new = np.copy(bias_matrix)
        for cmd in self.dst_nodes:
            for idx_link in range(self.num_links):
                e0, e1 = self.link_list[idx_link]
                val = self.pheromones[idx_link, self.fid_cmd_map[cmd]]
                if val > 0:
                    out_adj[e0, e1] = abs(val)
                elif val < 0:
                    out_adj[e1, e0] = abs(val)
                else:
                    pass
            out_adj = out_adj / np.linalg.norm(out_adj, ord=1, axis=1, keepdims=True)
            # step 2: update bias
            tmp = np.dot(out_adj, bias_matrix[cmd]+1)
            bias_matrix_new[~np.isnan(tmp), cmd] = tmp[~np.isnan(tmp)]
            bias_matrix_new[cmd, cmd] = 0
        return bias_matrix_new

    def update_bias(self, bias_matrix, delay_mtx):
        # step 1: find out neighbors, construct an out adj matrix
        bias_matrix_new = np.copy(bias_matrix)
        for v in range(self.num_nodes):
            _, nb_set = np.nonzero(self.adj_c[v])
            sp_v = (bias_matrix[nb_set, :] + delay_mtx[nb_set, v:v + 1]).min(axis=0)
            bias_matrix_new[v, :] = np.minimum(sp_v, bias_matrix[v, :])
        return bias_matrix_new

    def collect_delay(self, opt, T=1000):
        """
        Modified for Ph_routing
        :return:
        """
        flows_in = self.flows_arrivals[:, 0:T].sum(axis=1)
        flows_out = np.zeros((self.num_flows,), dtype=int)
        flows_delay = np.zeros((self.num_flows,))
        flows_jitter = np.zeros((self.num_flows,))
        flows_delay_est = np.zeros((self.num_flows,))
        flows_delay_raw = []
        flows_undeliver = []
        for fidx in range(self.num_flows):
            flow = self.flows[fidx]
            src = flow.source_node
            dst = flow.dest_node
            if opt > 0:
                flows_out[fidx] = len(self.backlog[dst][dst].queue)
            else:
                flows_out[fidx] = len(self.backlog_ph[dst][self.num_nodes].queue)
                # print(f'flow {fidx}: {flows_in[fidx]} pkt in and {flows_out[fidx]} pkt out')
            delay_per_pkt = np.zeros((flows_out[fidx],))
            for i in range(flows_out[fidx]):
                if opt > 0:
                    t0, t1 = self.backlog[dst][dst].queue[i]
                else:
                    t0, t1, cmd, last = self.backlog_ph[dst][self.num_nodes].queue[i]
                delay_per_pkt[i] = float(t1 - t0)
            flows_delay[fidx] = np.nanmean(delay_per_pkt)
            flows_jitter[fidx] = np.nanvar(delay_per_pkt)
            flows_delay_raw.append(delay_per_pkt)
            if opt > 0:
                delay_undelivered = []
                for i in range(self.num_nodes):
                    if i == flow.dest_node:
                        continue
                    for idx in range(len(self.backlog[i][dst].queue)):
                        t0, t1 = self.backlog[i][dst].queue[idx]
                        delay_undelivered.append(self.T - t0)
                delay_undelivered = np.array(delay_undelivered)
                delay_all_pkts = np.concatenate((delay_per_pkt, delay_undelivered), axis=0)
                flows_delay_est[fidx] = np.nanmean(delay_all_pkts)
                flows_undeliver.append(delay_undelivered)
            else:
                flows_undeliver.append([])
        if opt == 0:
            for v in range(self.num_nodes):
                _, nb_set = np.nonzero(self.adj_c[v])
                for u in nb_set:
                    qsize = self.backlog_ph[v][u].qsize()
                    for idx in range(qsize):
                        t0, t1, cmd, last = self.backlog_ph[v][u].queue[idx]
                        cmd_fid = self.fid_cmd_map[cmd]
                        flows_undeliver[cmd_fid].append(self.T - t0)
            # for fidx in range(self.num_flows):
            #     flows_delay_est[fidx] = np.nanmean(flows_undeliver[fidx])

        return flows_in, flows_out, flows_delay, flows_delay_raw, flows_jitter, flows_undeliver

    def plot_routes(self, delays, opt, with_labels=True, fdisp=-1):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        ces = ['g', 'm']
        for fidx in range(len(self.flows)):
            fig, ax = plt.subplots(1, 1)
            if 0 <= fdisp < len(self.flows):
                if fdisp != fidx:
                    continue
            for i in range(2):
                f_cnts = self.di_link_comd_cnts[self.edge_maps[i * self.num_links:(1 + i) * self.num_links], fidx]
                weights = 1 + 10 * f_cnts / (np.amax(f_cnts) + 0.000001)
                # weights = 1 + np.float_power(f_cnts, 0.33)
                vis_network(
                    self.graph_c,
                    self.src_nodes[fidx:fidx + 1],
                    self.dst_nodes[fidx:fidx + 1],
                    self.pos_c,
                    weights,
                    delay_f[:, fidx],
                    with_labels,
                    ax=ax,
                    colors=[ces[i], 'r', 'b'],
                    alpha=0.5
                )

            fig_name = "flow_routes_visual_{}_f{}_s{}_d{}_cf{:.0f}_opt{}.png".format(
                self.case_name, fidx,
                self.flows[fidx].source_node,
                self.flows[fidx].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            ax = plt.gca()
            ax.set_xlim(bbox[0:2])
            ax.set_ylim(bbox[2:4])
            # plt.tight_layout(pad=-0.1)
            plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close()
            # print("Flow {} plot saved to {}".format(fidx, fig_name))

    def plot_pheromones(self, delays, opt, with_labels=True):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        ces = ['g', 'm']
        for fidx in range(len(self.flows)):
            fig, ax = plt.subplots(1,1)
            for i in range(2):
                j = 1 - i
                f_cnts_p = np.abs(
                    self.pheromones[self.edge_maps[i*self.num_links:(1+i)*self.num_links], fidx]
                )
                f_cnts_n = np.abs(
                    self.pheromones[self.edge_maps[j*self.num_links:(1+j)*self.num_links], fidx]
                )
                f_cnts = np.clip(f_cnts_p - f_cnts_n, 0, None)
                weights = f_cnts  # 10 * f_cnts / (np.amax(f_cnts) + 0.000001) #
                vis_network(
                    self.graph_c,
                    self.src_nodes[fidx:fidx+1],
                    self.dst_nodes[fidx:fidx+1],
                    self.pos_c,
                    weights,
                    delay_f[:, fidx],
                    with_labels,
                    ax=ax,
                    colors=[ces[i], 'r', 'b'],
                    alpha=0.5
                )
            fig_name = "flow_pheromone_visual_{}_f{}_s{}_d{}_cf{:.1f}_opt{}.png".format(
                self.case_name, fidx,
                self.flows[fidx].source_node,
                self.flows[fidx].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            ax = plt.gca()
            ax.set_xlim(bbox[0:2])
            ax.set_ylim(bbox[2:4])
            # plt.tight_layout(pad=-0.1)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close()
            # print("Flow {} plot saved to {}".format(fidx, fig_name))

    def plot_delay(self, delay_n2c, opt):
        for fidx in range(self.num_flows):
            node_colors = ['y' for node in range(self.num_nodes)]
            node_sizes = 10*delay_n2c[:, fidx]
            node_colors[self.src_nodes[fidx]] = 'g'
            node_colors[self.dst_nodes[fidx]] = 'b'
            node_sizes[self.dst_nodes[fidx]] = 400
            ax = nx.draw(
                self.graph_c,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                pos=self.pos_c)
            fig_name = "flow_delay_visual_{}_f{}_s{}_d{}_cf{:.1f}_opt{}.png".format(
                self.case_name, fidx,
                self.flows[fidx].source_node,
                self.flows[fidx].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            plt.savefig(fig_name, dpi=300)
            plt.close()
            # print("Flow {} plot saved to {}".format(fidx, fig_name))

    def plot_metrics(self, opt):
        arrivals = np.sum(self.flows_arrivals, axis=0)
        pkts_in_network = np.sum(self.flow_pkts_in_network, axis=0)
        departures = np.sum(self.flows_sink_departures, axis=0)

        plt.plot(arrivals)
        plt.plot(departures)
        plt.plot(pkts_in_network)

        plt.suptitle('Departures, Arrivals, and Current amount pkts in network')
        plt.xlabel('T')
        plt.ylabel('the number of packages')
        plt.legend(['Exogenous arrivals', 'Sink departures', 'Pkts in network'], loc='upper right')
        fig_name = "flow_packets_arrivals_per_timeslot_{}_cf{:.1f}_opt_{}.png".format(self.case_name, self.cf_radius, opt)
        fig_name = os.path.join("..", "fig", fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close()
        # print("Metrics plot saved to {}".format(fig_name))
        return arrivals, pkts_in_network, departures

    def animate_pheromones(self, num_frames, delays, opt, interval, with_labels=True, save_path=True):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        ces = ['g', 'm']
        links1 = [(u, v) for u, v in self.link_list]
        links2 = [(v, u) for u, v in self.link_list]
        print(f'green direction is {links1}')
        print(f'red direction is {links2}')

        def update(frame, f_show, ax):
            links = links1 + links2
            for f in f_show:
                ax[0, f].clear()
                edge_labels = {}
                for i in range(2):
                    j = 1 - i
                    f_cnts_p = np.abs(
                        self.pheromones_vis[self.edge_maps[i * self.num_links:(1 + i) * self.num_links], f, frame]
                    )
                    f_cnts_n = np.abs(
                        self.pheromones_vis[self.edge_maps[j * self.num_links:(1 + j) * self.num_links], f, frame]
                    )
                    f_cnts = np.clip(f_cnts_p - f_cnts_n, 0, None)
                    weights = f_cnts  # 10 * f_cnts / (np.amax(f_cnts) + 0.000001) #
                    les = self.edge_maps[i * self.num_links:(1 + i) * self.num_links]
                    link_list = [links[ix] for ix in les]
                    for idx, w in enumerate(f_cnts):
                        if i == 0:
                            edge_labels[link_list[idx]] = [round(w, 1)]
                        else:
                            l = link_list[idx][1], link_list[idx][0]
                            edge_labels[l].append(round(w, 1))
                    vis_network(
                        self.graph_c,
                        self.src_nodes[f_show[f]:f_show[f] + 1],
                        self.dst_nodes[f_show[f]:f_show[f] + 1],
                        self.pos_c,
                        weights,
                        delay_f[:, f_show[f]],
                        with_labels,
                        ax=ax[0, f],
                        colors=[ces[i], 'r', 'b'],
                        alpha=0.5
                    )
                # vis_edges(self.graph_c, pos=self.pos_c, edge_labels=edge_labels, ax=ax[0, f], font_size= 7)

                ax[0, f].set_xlim(bbox[0:2])
                ax[0, f].set_ylim(bbox[2:4])
                ax[0, f].set_title(f'flow {f}')

                ax[1, f].clear()
                edge_labels = {}
                # edges = [(idx) for idx in self.graph_c.edges]
                for i in range(2):
                    weights = self.pkt_vis[self.edge_maps[i * self.num_links:(1 + i) * self.num_links], f, frame]
                    les = self.edge_maps[i * self.num_links:(1 + i) * self.num_links]
                    link_list = [links[ix] for ix in les]
                    for idx, w in enumerate(weights):
                        if i == 0:
                            edge_labels[link_list[idx]] = int(w)
                        else:
                            l = link_list[idx][1], link_list[idx][0]
                            if edge_labels[l] == 0:
                                edge_labels[l] = int(w)
                    vis_network(
                        self.graph_c,
                        self.src_nodes[f_show[f]:f_show[f] + 1],
                        self.dst_nodes[f_show[f]:f_show[f] + 1],
                        self.pos_c,
                        weights,
                        delay_f[:, f_show[f]],
                        with_labels,
                        ax=ax[1, f],
                        colors=[ces[i], 'r', 'b'],
                        alpha=0.5
                    )
                    vis_edges(self.graph_c, pos=self.pos_c, edge_labels=edge_labels, ax=ax[1, f])

                ax[1, f].set_xlim(bbox[0:2])
                ax[1, f].set_ylim(bbox[2:4])
                ax[1, f].set_title(f'Number of packets transmitted at time {frame+1}')

        def save_ani():
            ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(f_show, ax), interval=interval, repeat=False)
            anim_name = "packets_arrivals_per_timeslot_{}_cf{:.1f}.png".format(self.case_name, self.cf_radius)
            anim_name = os.path.join("..", "fig", anim_name)
            ani.save(anim_name + ".gif", writer="pillow", fps=1, dpi=300)

        def show_ani():
            ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(f_show, ax), interval=interval, repeat=False)
            plt.tight_layout()
            plt.show()

        f_show = range(self.num_flows)
        fig, ax = plt.subplots(2, self.num_flows, figsize=(18, 10))

        if save_path:
            save_ani()
        else:
            show_ani()

    def save_pherom(self, opt, id, f_case, items, bursty):

        for idx, t in enumerate(self.t_recordings):
            pherom_name_tim = f'Pheromone_{id}_opt_{opt}_fcase_{f_case}_{t}_{items}_{bursty}.pkl'
            print(f'while saving pherom_name_tim is {pherom_name_tim} and max and min are {np.max(self.pheromones_vis[..., idx+1]), np.min(self.pheromones_vis[..., idx+1])}')
            pherom_name_tim = os.path.join("..", "pkl", pherom_name_tim)
            with open(pherom_name_tim, 'wb') as file:
                pickle.dump(self.pheromones_vis[..., idx+1], file)

    def load_pherom(self, opt, id, f_case, items, t_recording=0, bursty=None, test=False):
        pherom_name_tim = f'Pheromone_{id}_opt_{opt}_fcase_{f_case}_{t_recording}_{items}_{bursty}.pkl'
        pherom_name_tim = os.path.join("..", "pkl", pherom_name_tim)
        if pickle_file_exists(pherom_name_tim):
            if test:
                return pherom_name_tim
            with open(pherom_name_tim, 'rb') as file:
                self.pheromones = pickle.load(file)
            self.pheromones_vis = np.zeros((self.num_di_links, self.num_flows, 1), dtype=float)
            print(f'Load {pherom_name_tim}, max-min: {np.max(self.pheromones), np.min(self.pheromones)}')
            return True
        else:
            return False



def main(args):
    # Configuration
    opt = int(args[0])
    seed = 102
    mbp = 0.0
    NUM_NODES = 100
    # LAMBDA = 1 # we are not use it
    T = 1000
    link_rate_max = 42  # 42
    link_rate_min = 10  # 10
    link_rate_avg = (link_rate_max + link_rate_min) / 2

    cf_radius = 0.0 # relative conflict radius based on physical distance model
    exploration_rate = 0.0 # 0.3
    func = 'proportional'
    if opt==0:
        warmstart = True
        decay = 0.998
        unit = 0.01
        init = 0.01
        burst_prob = 0.5
    else:
        warmstart = False
        decay = 1.0
        unit = 0.01
        init = 0.01
        burst_prob = 0.0 # set to zero for generating routing policy
    opt_warmstart = 49
    all_same = False
    not_going_back = False
    ph_diff = True

    # Create fig folder if not exist
    if not os.path.isdir(os.path.join("..", "fig")):
        os.mkdir(os.path.join("..", "fig"))
    # Create pos folder if not exist
    if not os.path.isdir(os.path.join("..", "pos")):
        os.mkdir(os.path.join("..", "pos"))
    # Create pos folder if not exist
    if not os.path.isdir(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))

    if not os.path.isdir(os.path.join("..", "pkl")):
        os.mkdir(os.path.join("..", "pkl"))
    if not os.path.isdir(os.path.join("..", "log")):
        os.mkdir(os.path.join("..", "log"))

    start_time = time.time()
    np.random.seed(seed)
    # bp_env = Backpressure(NUM_NODES, T, seed)
    # matfile = '../data/data_poisson_10/poisson_graph_seed514_m8_n100_f21.mat'
    # bp_env = BackpressureAnt(NUM_NODES, T, seed, cf_radius=cf_radius, gtype=matfile)
    bp_env = BackpressureAnt(NUM_NODES, T, seed, cf_radius=cf_radius, gtype='ba')
    case_name = bp_env.case_name

    #f0 = Flow(3, 2, 6)
    #f1 = Flow(1, 5, 5)
    # bp_env.clear_all_flows()
    # bp_env.add_flow(3, 6, rate=3)
    # bp_env.add_flow(1, 7, rate=2)
    # bp_env.add_flow(2, 11, rate=2)
    flows_perc = np.random.randint(30, 50)
    num_flows = round(flows_perc / 100 * bp_env.num_nodes)
    nodes = bp_env.graph_c.nodes()
    num_arr = np.random.permutation(nodes)
    arrival_rates = np.random.uniform(0.2, 1.0, (num_flows,))
    link_rates = np.random.uniform(link_rate_min, link_rate_max, size=(bp_env.num_links,))
    bp_env.links_init(link_rates)

    bp_env.clear_all_flows()
    srcs = []
    dsts = []
    flow_rates = []
    cutoffs = []
    flows = []
    fll = 0
    sflows = []
    bflows = []
    for fidx in range(num_flows):
        src = num_arr[2 * fidx]
        dst = num_arr[2 * fidx + 1]
        cutoff = -1
        ar_multiplier = 1.0
        if np.random.uniform(0,1) < burst_prob:
            cutoff = 30
            fll += 1
            bflows.append(fidx)
        else:
            sflows.append(fidx)
        bp_env.add_flow(src, dst, rate=ar_multiplier * arrival_rates[fidx], cutoff=cutoff)
        flow = {'src': src, 'dst': dst, 'rate': ar_multiplier * arrival_rates[fidx], 'cut': cutoff}
        flows.append(flow)
        srcs.append(src)
        dsts.append(dst)
        flow_rates.append(ar_multiplier * arrival_rates[fidx])
        cutoffs.append(cutoff)
    print(f'out of {num_flows} flows {fll} of them are bursty')
    bp_env.flows_init()
    # Now the following two calls must be placed after flows_init()
    bp_env.queues_init()
    bp_env.pheromone_init(decay=decay, unit=unit, init=init)

    # logfile = os.path.join("..", "log", "Output_{}_opt_{}.txt".format(bp_env.case_name, opt))
    # with open(logfile, "a") as f:
    #     print("Edges:", file=f)
    #     print(bp_env.graph_i.nodes(), file=f)
    #     print("Link Rates:", file=f)
    #     print(bp_env.link_rates, file=f)

    print("Init graph('{}') in {:.3f} seconds".format(bp_env.case_name, time.time() - start_time),
          ": conflict radius {}, degree {:.2f}".format(bp_env.cf_radius, bp_env.mean_conflict_degree))
    start_time = time.time()

    # items = (func, decay, exploration_rate, unit, not_going_back)
    if warmstart:
        bp_env.load_pherom(opt_warmstart, case_name, 0, 'test', T-1, 'stream')
        bp_env.freeze_pherom()

    # Bias computation
    if opt in [0, 49]:
        delay_est = np.divide(link_rate_avg ** 2, link_rates)
        for link, delay in zip(bp_env.link_list, delay_est):
            src, dst = link
            bp_env.graph_c[src][dst]["delay"] = delay
        shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
        bias_matrix = shortest_paths
        bias_vector = bp_env.bias_diff(bias_matrix)
        link_bias_vec = bias_vector * (link_rate_avg / np.min(delay_est))
    else:
        raise ValueError("unsupported opt {}".format(opt))

    for t in range(bp_env.T):
        bp_env.pkt_arrival(t)

        # Commodity and W computation
        if opt > 0:
            W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix, mbp, link_bias_vec)
            W_amp[C_opt == -1] = 0.0
            bp_env.opt_comd_mtx[:, t] = C_opt
        else:
            W_amp, W_sign = bp_env.ph_routing(
                t,
                func=func,
                exploration_rate=exploration_rate,
                not_going_back=not_going_back,
                ph_diff=ph_diff,
            )
        bp_env.W[:, t] = W_amp
        bp_env.WSign[:, t] = W_sign

        # Greedy Maximal Scheduling & Transmission
        mwis = bp_env.scheduling(bp_env.W[:, t] * bp_env.link_rates[:, t])

        if opt > 0:
            bp_env.transmission(t, mwis)
            for fidx in range(bp_env.num_flows):
                bp_env.flow_pkts_in_network[fidx, t] = np.sum(bp_env.queue_matrix[:, bp_env.flows[fidx].dest_node])
        else:
            bp_env.transmission_ph(t, mwis)

    print("Main loop {} time slots in {:.3f} seconds".format(T, time.time() - start_time))

    start_time = time.time()
    # delay_n2c, diff_vec = bp_env.estimate_delay()
    # print("Estimating delay in {:.3f} seconds".format(time.time() - start_time))
    # bp_env.plot_routes(delay_n2c, opt)
    cnt_in, cnt_out, delay_e2e, delay_e2e_raw, delay_est, undeliver = bp_env.collect_delay(opt, bp_env.T)
    bp_env.plot_metrics(opt)
    if opt > 0 and burst_prob == 0.0:
        bp_env.save_pherom(opt, case_name, 0, 'test', 'stream')

    if opt > 0:
        logfile = os.path.join("..", "log",f"debug_num_nodes_{NUM_NODES}_time_{T}_opt_{opt}.txt")
    else:
        logfile = os.path.join("..", "log",f"debug_decay_{decay}_explorationrate_{exploration_rate}_function_{func}_num_nodes_{NUM_NODES}_time_{T}.txt")
    with open(logfile, "a") as f:
        print("\nph_diff is {}".format(ph_diff), file=f)
        if warmstart:
            print("warmstart with {}".format(opt_warmstart), file=f)
        print("Estimating delay in {:.3f} seconds".format(time.time() - start_time), file=f)
        print("seed: {}, number of nodes: {}, T: {}".format(seed, NUM_NODES, T), file=f)
        print(f'out of {num_flows} flows {fll} of them are bursty', file=f)
        print("Pkt in: {}\nPkt out: {}\ndelivery_rate: {}\nLatency: {}".format(cnt_in, cnt_out, cnt_out/cnt_in, delay_e2e), file=f)
        print("mean delivery_rate: s{}, b{}\nmean Latency: s{}, b{}".format(
            np.nanmean(cnt_out[sflows] / cnt_in[sflows]),
            np.nanmean(cnt_out[bflows] / cnt_in[bflows]),
            np.nanmean(delay_e2e[sflows]), np.nanmean(delay_e2e[bflows])), file=f)
        print("function: {} decay: {} unit: {} init: {} exploration_rate: {}".format(func, decay, unit, init, exploration_rate), file=f)
    # delay_est_vec = np.ones((bp_env.num_nodes, bp_env.num_flows))
    # delay_est_vec = np.multiply(np.reshape(delay_est, (1, bp_env.num_flows)), delay_est_vec)
    # bp_env.plot_routes(delay_est_vec, opt)
    # bp_env.plot_pheromones(delay_est_vec, opt)
    # bp_env.animate_pheromones(bp_env.T, delay_est_vec, opt, interval=20, with_labels=True, save_path=False)

    print("Done")


if __name__ == "__main__":
    # print(f'sys.argv[1:] is {sys.argv}')
    main(sys.argv[1:])
    # main(args = [49])