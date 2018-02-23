#!/usr/bin/env python

class ViewSet:
	""" Stores views and connections between them. """
	def __init__(self):
		self.numViews = 0
		#  Views is a dictionary, that contains ViewId, Points, Orientation and location
		self.views = {}
		self.connections = {}
		self.projections = {}

	def add_connection(self):
		pass

	def add_view(self, im):
		self.views[self.numViews] = im
		self.numViews += 1
		#  Limit the number of stored views (temporary)
		if self.numViews > 5:
			del self.views[self.numViews - 5]