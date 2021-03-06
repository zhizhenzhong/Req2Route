import copy
import yaml
import numpy as np
from scipy.sparse.csgraph import dijkstra

class Graph:
    def __init__(self, topo_file):
        self.topo = None
        try:
            with open(topo_file, 'r', encoding='utf8') as fr:
                self.topo = yaml.load(fr)
        except:
            print('load topo error! please use valid yaml file.')
        assert 'nodes' in self.topo, 'topo file must contains nodes'
        assert 'domain' in self.topo, 'topo file must contains domain'
        assert 'capacity' in self.topo, 'topo file must contains capacity'
        assert 'links' in self.topo, 'topo file must contains links'
        assert 'linkstate' in self.topo, 'topo file must contains linkstate'
        assert 'indexnode' in self.topo, 'topo file must contains indexnode'
        assert 'interdomain_dis' in self.topo, 'topo file must contains interdomain dis'
        self.nodes = self.topo['nodes']
        self.domain = self.topo['domain']
        self.indexnode = self.topo['indexnode']
        self.capacity = np.array(self.topo['capacity'], dtype=int)
        self.links = np.array(self.topo['links'], dtype=int)
        self.linkstate = np.array(self.topo['linkstate'], dtype=int)
        self.interdomain_dis = self.topo['interdomain_dis']
        self.index2node = {index : node for node, index in self.indexnode.items()}
        self.node2domain = {node : domain_id for domain_id, domain_nodes in self.domain.items() for node in domain_nodes}
        self.initial_capacity = copy.deepcopy(self.capacity)

    def get_index_route(self, input_route): #get path index from path name
        return [self.indexnode[x] for x in input_route]

    def get_route_name(self, input_route): #get path name after decode (there is - signal after decode)
        return [self.index2node[int(x)] for x in input_route.split('-')]

    def reset_capacity(self):
        self.capacity = copy.deepcopy(self.initial_capacity)
    
    def set_capacity(self, caps, hidden_caps):  
        """update net capacity using given capacity list,
        e.g., [10, 9, 10, 9, ..., 10, 10]"""
        for index, linkstate_data in enumerate(self.linkstate):
            if linkstate_data[2] == 1:
                self.capacity[self.indexnode[linkstate_data[0]]][self.indexnode[linkstate_data[1]]] = hidden_caps[index]
            if linkstate_data[2] == 10:
                self.capacity[self.indexnode[linkstate_data[0]]][self.indexnode[linkstate_data[1]]] = caps[index - 152]

    def cap_mat2list(self, graph_matrix):  
        """use self.capacity to get current capacity list (used for Existing input)"""
        cap = np.array([0 for i in range(48)])
        for index, linkstate_data in enumerate(self.linkstate):
            if linkstate_data[2] == 10:
                cap[index-152] = graph_matrix[self.indexnode[linkstate_data[0]]][self.indexnode[linkstate_data[1]]]
        return cap

    def is_buildable(self, path, verbose=False, use_place = False):
        if verbose: print('\n[is_buildable] start (index): {}'.format(path))
        path_name = [self.index2node[int(x)] for x in path]
        if verbose: print('start:', path_name)
        weights = (self.capacity > 0) * self.links
        if len(path) <= 1:
            return False
        success = True
        links = []
        real_path = []
        for src, dst in zip(path[:-1], path[1:]):
            src, dst = int(src), int(dst)
            if src == 61 or dst == 61:
                success = False
            else:
                if self.node2domain[self.index2node[src]] != self.node2domain[self.index2node[dst]]:
                    if self.capacity[src, dst] <= 0:
                        success = False
                        print(' ! no capacity: [{}] -> [{}]'.format(self.index2node[src], self.index2node[dst]))
                        if use_place: print('! Wrong in existing task loading!', path_name)
                        break
                    else:
                        links.append((src, dst))
                        if verbose: print(' - cross link: [{}] -> [{}]'.format(self.index2node[src], self.index2node[dst]))
                # src and dst are from the same domain
                else:
                    if src == dst:
                        success = False
                    if verbose: print(' + calling dijkstra: {} -> {}'.format(self.index2node[src], self.index2node[dst]))
                    #modify graph (weights) for intra domain dijkstra!
                    domain_index = self.node2domain[self.index2node[src]]
                    weights = (self.capacity > 0) * self.links
                    for nn in range(61):
                        if self.node2domain[self.index2node[nn]] != domain_index:
                            for tt in range(61):
                                weights[nn][tt] = 0
                                weights[tt][nn] = 0

                    dist, shortest_path = self.dijkstra(weights, src, dst)
                    shortest_path_name = [self.index2node[x] for x in shortest_path]
                    
                    if dist == -1:
                        success = False
                        print(' ! no dijkstra path: {} -> {} {} {}'.format(self.index2node[src], self.index2node[dst], dist, intra_path))
                    else:
                        for src_intra, dst_intra in zip(shortest_path[:-1], shortest_path[1:]):
                            links.append((src_intra, dst_intra))
                        if verbose: print(' - dijkstra path: {}'.format(shortest_path_name))
        if verbose:
            print(' = success:', success)
            print(' = links:', links)
            print('[is_buildable] end')
        if success:
            real_path = [links[0][0]] + [y for x, y in links]

        return success, real_path

    def build_path(self, path):
        if len(path) > 1:
            for src, dst in zip(path[:-1], path[1:]):
                self.capacity[src, dst] -= 1

    def break_path(self, path):
        if len(path) > 1:
            for src, dst in zip(path[:-1], path[1:]):
                self.capacity[src, dst] += 1

    def dijkstra(self, weights, src, dst):
        assert src < self.nodes and dst < self.nodes
        dist_matrix, predecessors = dijkstra(weights, directed=False, return_predecessors=True)
        path = []
        i, j = src, dst
        while i != j and j >= 0:
            path.append(j)
            j = predecessors.item(i, j)
        path.reverse()
        # process result
        path.insert(0, src)
        dist = dist_matrix[src, dst]
        if dist_matrix[src, dst] == np.inf:
            dist = -1
        else:
            dist = int(dist)
        return dist, path

    def seq_before_zero(self, seq):
        zero_at = -1
        for i, node in enumerate(seq):
            if node == 0:
                zero_at = i
                break
        seq_none_zero = seq[:zero_at] if zero_at > -1 else seq
        return seq_none_zero


if __name__ == '__main__':
    graph = Graph('topo9.yaml')
 
    print('\nprocess finished~~~')
    
