import networkx as nx
from image import Image
import matplotlib.pyplot as plt
import os
import scipy
import operator

class Graph:
	def __init__(self, training_set):
		self.G = nx.Graph()

		images = []
		for x in training_set:
			images.append(Image(x))

		for x in range(0, len(images)):
			for y in range(x + 1, len(images)):
				self.G.add_edge(images[x], images[y], weight=images[x].difference(images[y]))

	def save_images(self):
		count = 0
		if not os.path.exists('pics'):
			os.makedirs('pics')
		for node in self.G.nodes():
			count += 1
			scipy.misc.toimage(255 * node.image, cmin=0.0, cmax=255).save('pics/' + str(count) + '.png')



	def get_modified_input(self):
		MST = nx.minimum_spanning_tree(self.G)
		new_set = {}
		count = 0
		while(True):
			nodes_degree = MST.degree()
			leaves = [x for x in nodes_degree if nodes_degree[x]==1]
			for x in leaves:
				new_set[x] = 2**count
			count+=1
			MST.remove_nodes_from(leaves)
			if MST.number_of_nodes() == 1:
				new_set[MST.nodes()[0]] = 2**count
				break
			if MST.number_of_nodes() == 0:
				break
		examples = []
		for x in new_set:
			examples.extend([x.get_training_example()] * new_set[x])
			# examples.extend([x.get_training_example()] * 1)
		return examples

	def get_modified_input2(self):
		MST = nx.minimum_spanning_tree(self.G)
		new_set = {}
		count = 0
		while (True):
			nodes_degree = MST.degree()
			leaves = [x for x in nodes_degree if nodes_degree[x] == 1]
			for x in leaves:
				new_set[x] = 2 ** count
			count += 1
			MST.remove_nodes_from(leaves)
			if MST.number_of_nodes() == 1:
				new_set[MST.nodes()[0]] = 2 ** count
				break
			if MST.number_of_nodes() == 0:
				break
		list = sorted(new_set.items(), key=operator.itemgetter(1), reverse=True)
		count = 0
		for x in list:
			count += x[1]
		return count, list