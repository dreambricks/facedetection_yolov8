from scipy.spatial import distance

class TrackableObject:
	def __init__(self, objectID, centroid, rect): # constructor
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.rect = rect
		self.centroid = centroid
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		self.delta_dist = 0.0
		self.disappeared = 0

	def update(self, centroid, rect):
		self.delta_dist = self.calculate_delta_dist(centroid)
		self.centroid = centroid
		self.rect = rect
		self.disappeared = 0

	def calculate_delta_dist(self, centroid):
		return distance.euclidean(self.centroid, centroid)

	def get_disappeared(self):
		return self.disappeared

	def set_disappeared(self):
		self.disappeared += 1