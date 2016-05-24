import numpy as np
class Image:
	id = 0
	def __init__(self, image):
		self.image = np.reshape(image[0], (-1, 28))
		self.y = image[1]
		self.id = Image.id
		Image.id+=1
		pass

	def difference(self, other):
		if(self.image.shape != other.image.shape):
			raise ValueError("Dimensions not equal")
		count = 0
		for x in range(0, self.image.shape[0]):
			for y in range(0, self.image.shape[1]):
				count += abs(self.image[x][y] - other.image[x][y])
		return count

	def get_training_example(self):
		return (np.reshape(self.image, (-1, 1)), self.y)

	def __hash__(self):
		return hash(self.id)

	def __eq__(self, other):
		return self.id == other.id

	def __ne__(self, other):
		# Not strictly necessary, but to avoid having both x==y and x!=y
		# True at the same time
		return not (self == other)

def test():
	pass


if __name__ == "__main__":
	test()