import numpy as np
import os
import random


class tNode(object):
    def __init__(self, idx=-1, word=None):
        self.left = None
        self.right = None
        self.word = word
        self.size = 0
        self.height = 1
        self.parent = None
        self.label = None
        self.children = []
        self.idx = idx
        self.span = None

    def add_parent(self, parent):
        self.parent = parent

    def add_child(self, node):
        assert len(self.children) < 2
        self.children.append(node)

    def add_children(self, children):
        self.children.extend(children)

    def get_left(self):
        left = None
        if self.children:
            left = self.children[0]
        return left

    def get_right(self):
        right = None
        if len(self.children) == 2:
            right = self.children[1]
        return right

    @staticmethod
    def get_height(root):
        if root.children:
            root.height = max(root.get_left().height, root.get_right().height) + 1
        else:
            root.height = 1
        print(root.idx, root.height, 'asa')

    @staticmethod
    def get_size(root):
        if root.children:
            root.size = root.get_left().size + root.get_right().size + 1
        else:
            root.size = 1

    @staticmethod
    def get_spans(root):
        if root.children:
            root.span = root.get_left().span + root.get_right().span
        else:
            root.span = [root.word]

    @staticmethod
    def get_numleaves(self):
        if self.children:
            self.num_leaves = self.get_left().num_leaves + self.get_right().num_leaves
        else:
            self.num_leaves = 1

    @staticmethod
    def postOrder(root, func=None, args=None):

        if root is None:
            return
        tNode.postOrder(root.get_left(), func, args)
        tNode.postOrder(root.get_right(), func, args)

        if args is not None:
            func(root, args)
        else:
            func(root)

    @staticmethod
    def encodetokens(root, func):
        if root is None:
            return
        if root.word is None:
            return
        else:
            root.word = func(root.word)

    @staticmethod
    def relabel(root, fine_grained):
        if root is None:
            return
        if root.label is not None:
            if fine_grained:
                root.label += 2
            else:
                if root.label < 0:
                    root.label = 0
                elif root.label == 0:
                    root.label = 1
                else:
                    root.label = 2


def processTree(root, funclist=None, argslist=None):
    if funclist is None:
        root.postOrder(root, root.get_height)
        root.postOrder(root, root.get_num_leaves)
        root.postOrder(root, root.get_size)
    else:
        # print funclist,argslist
        for func, args in zip(funclist, argslist):
            root.postOrder(root, func, args)

    return root


def test_tNode():
    nodes = {}
    for i in range(7):
        nodes[i] = tNode(i)
        if i < 4: nodes[i].word = i + 10
    nodes[0].parent = nodes[1].parent = nodes[4]
    nodes[2].parent = nodes[3].parent = nodes[5]
    nodes[5].parent = nodes[6].parent = nodes[6]
    nodes[6].add_child(nodes[4])
    nodes[6].add_child(nodes[5])
    nodes[4].add_children([nodes[0], nodes[1]])
    nodes[5].add_children([nodes[2], nodes[3]])
    root = nodes[6]
    postOrder = root.postOrder
    postOrder(root, tNode.get_height, None)
    postOrder(root, tNode.get_numleaves, None)
    postOrder(root, root.get_spans, None)
    print(root.height, root.num_leaves)
    for n in nodes.itervalues(): print(n.span)


class Vocab(object):
    def __init__(self, path):
        self.words = []
        self.word2idx = {}
        self.idx2word = {}

        self.load(path)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                w = line.strip()
                assert w not in self.words
                self.words.append(w)
                self.word2idx[w] = len(self.words) - 1  # 0 based index
                self.idx2word[self.word2idx[w]] = w

    def __len__(self):
        return len(self.words)

    def encode(self, word):
        # if word not in self.words:
        # word = self.unk_word
        return self.word2idx[word]

    def decode(self, idx):
        assert idx >= len(self.words)
        return self.idx2word[idx]

    def size(self):
        return len(self.words)


def load_sentiment_treebank(data_dir, fine_grained):
    voc = Vocab(os.path.join(data_dir, 'vocab-cased.txt'))

    split_paths = {}
    for split in ['train', 'test', 'dev']:
        split_paths[split] = os.path.join(data_dir, split)

    fnlist = [tNode.encodetokens, tNode.relabel]
    arglist = [voc.encode, fine_grained]
    # fnlist,arglist=[tNode.relabel],[fine_grained]

    data = {}
    for split, path in split_paths.items():
        sentencepath = os.path.join(path, 'sents.txt')
        treepath = os.path.join(path, 'parents.txt')
        labelpath = os.path.join(path, 'labels.txt')
        trees = parse_trees(sentencepath, treepath, labelpath)
        if not fine_grained:
            trees = [tree for tree in trees if tree.label != 0]
        trees = [(processTree(tree, fnlist, arglist), tree.label) for tree in trees]
        data[split] = trees

    return data, voc


def parse_trees(sentencepath, treepath, labelpath):
    trees = []
    with open(treepath, 'r') as ft, open(labelpath) as fl, open(
            sentencepath, 'r') as f:
        while True:
            parentidxs = ft.readline()
            labels = fl.readline()
            sentence = f.readline()
            if not parentidxs or not labels or not sentence:
                break
            parentidxs = [int(p) for p in parentidxs.strip().split()]
            labels = [int(l) if l != '#' else None for l in labels.strip().split()]

            tree = parse_tree(sentence, parentidxs, labels)
            trees.append(tree)
    return trees


def parse_tree(sentence, parents, labels):
    nodes = {}
    parents = [p - 1 for p in parents]  # change to zero based
    sentence = [w for w in sentence.strip().split()]
    for i in range(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tNode(idx)
                if prev is not None:
                    assert prev.idx != node.idx
                    node.add_child(prev)

                node.label = labels[idx]
                nodes[idx] = node

                if idx < len(sentence):
                    node.word = sentence[idx]

                parent = parents[idx]
                if parent in nodes:
                    assert len(nodes[parent].children) < 2
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    return root


def BFStree(root):
    from collections import deque
    node = root
    leaves = []
    inodes = []
    queue = deque([node])
    func = lambda node: node.children == []

    while queue:
        node = queue.popleft()
        if func(node):
            leaves.append(node)
        else:
            inodes.append(node)
        if node.children:
            queue.extend(node.children)

    return leaves, inodes


def extract_tree_data(tree, max_degree=2, only_leaves_have_vals=True, with_labels=False):
    # processTree(tree)
    # fnlist=[tree.encodetokens,tree.relabel]
    # arglist=[voc.encode,fine_grained]
    # processTree(tree,fnlist,arglist)
    leaves, inodes = BFStree(tree)
    labels = []
    leaf_emb = []
    tree_str = []
    i = 0
    for leaf in reversed(leaves):
        leaf.idx = i
        i += 1
        labels.append(leaf.label)
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        node.idx = i
        c = [child.idx for child in node.children]
        tree_str.append(c)
        labels.append(node.label)
        if not only_leaves_have_vals:
            leaf_emb.append(-1)
        i += 1
    if with_labels:
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(leaf_emb, dtype='int32'),
                np.array(tree_str, dtype='int32'),
                np.array(labels, dtype=float),
                np.array(labels_exist, dtype=float))
    else:
        print(leaf_emb, 'asas')
        return (np.array(leaf_emb, dtype='int32'),
                np.array(tree_str, dtype='int32'))


def extract_batch_tree_data(batchdata, fillnum=120):
    dim1, dim2 = len(batchdata), fillnum
    # leaf_emb_arr,treestr_arr,labels_arr=[],[],[]
    leaf_emb_arr = np.empty([dim1, dim2], dtype='int32')
    leaf_emb_arr.fill(-1)
    treestr_arr = np.empty([dim1, dim2, 2], dtype='int32')
    treestr_arr.fill(-1)
    labels_arr = np.empty([dim1, dim2], dtype=float)
    labels_arr.fill(-1)
    for i, (tree, _) in enumerate(batchdata):
        input_, treestr, labels, _ = extract_tree_data(tree,
                                                       max_degree=2,
                                                       only_leaves_have_vals=False,
                                                       with_labels=True)
        leaf_emb_arr[i, 0:len(input_)] = input_
        treestr_arr[i, 0:len(treestr), 0:2] = treestr
        labels_arr[i, 0:len(labels)] = labels

    return leaf_emb_arr, treestr_arr, labels_arr


def extract_seq_data(data, numsamples=0, fillnum=100):
    seqdata = []
    seqlabels = []
    for tree, _ in data:
        seq, seqlbls = extract_seq_from_tree(tree, numsamples)
        seqdata.extend(seq)
        seqlabels.extend(seqlbls)

    seqlngths = [len(s) for s in seqdata]
    maxl = max(seqlngths)
    assert fillnum >= maxl
    if 1:
        seqarr = np.empty([len(seqdata), fillnum], dtype='int32')
        seqarr.fill(-1)
        for i, s in enumerate(seqdata):
            seqarr[i, 0:len(s)] = np.array(s, dtype='int32')
        seqdata = seqarr
    return seqdata, seqlabels, seqlngths, maxl


def extract_seq_from_tree(tree, numsamples=0):
    if tree.span is None:
        tree.postOrder(tree, tree.get_spans)

    seq, lbl = [], []
    s, l = tree.span, tree.label
    seq.append(s)
    lbl.append(l)

    if not numsamples:
        return seq, lbl

    num_nodes = tree.idx
    if numsamples == -1:
        numsamples = num_nodes
    # numsamples=min(numsamples,num_nodes)
    # sampled_idxs = random.sample(range(num_nodes),numsamples)
    # sampled_idxs=range(num_nodes)
    # print sampled_idxs,num_nodes

    subtrees = {}

    # subtrees[tree.idx]=
    # func=lambda tr,su:su.update([(tr.idx,tr)])
    def func_(self, su):
        su.update([(self.idx, self)])

    tree.postOrder(tree, func_, subtrees)

    for j in range(numsamples):  # sampled_idxs:
        i = random.randint(0, num_nodes)
        root = subtrees[i]
        s, l = root.span, root.label
        seq.append(s)
        lbl.append(l)

    return seq, lbl


def get_max_len_data(datadic):
    maxlen = 0
    for data in datadic.values():
        for tree, _ in data:
            tree.postOrder(tree, tree.get_numleaves)
            assert tree.num_leaves > 1
            if tree.num_leaves > maxlen:
                maxlen = tree.num_leaves

    return maxlen


def get_max_node_size(datadic):
    maxsize = 0
    for data in datadic.values():
        for tree, _ in data:
            tree.postOrder(tree, tree.get_size)
            assert tree.size > 1
            if tree.size > maxsize:
                maxsize = tree.size

    return maxsize


def test_fn():
    data_dir = './stanford_lstm/data/sst'
    fine_grained = 0
    data, _ = load_sentiment_treebank(data_dir, fine_grained)
    for d in data.itervalues():
        print(len(d))

    d = data['dev']
    a, b, c, _ = extract_seq_data(d[0:1], 5)
    print(a, b, c)

    print(get_max_len_data(data))

    return data


if __name__ == '__main__':
    test_fn()
