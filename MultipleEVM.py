from EVM import EVM
import numpy
import h5py
import time
import sys
import torch
import weibull
import logging
logger = logging.getLogger("MultiEVM")


def _train(args, data, label, distance_multiplier, distance_function, extra_negatives):
  """Internal function to separate data into positives and negatives and call the EVM class. Do not call directly."""
  self,i = args
  positives = data[i]
  negatives = []
  # collect negatives randomly from training data
  for j in range(len(data)):
    if i != j:
      negatives.extend(data[j])
      if extra_negatives is not None:
        negatives.extend(extra_negatives)
  # now, train the EVM
  evm = EVM(self.tailsize, self.cover_threshold, distance_multiplier, distance_function, log_level='debug', device=self.device)
  evm.train(positives, torch.stack(negatives), label)
  return evm


class MultipleEVM (object):
  """Computes a list of EVMs from a larger set of features

  For a given list of features from different classes, this class computes an EVM for each of the classes by taking all the other classes as negative training samples.

  Parameters
  ----------

  tailsize : int
    The number of sample distances to include in the Weibull fitting

  cover_threshold : float or ``None``
    If given, the EVMs compute a set cover with the given cover threshold

  distance_multiplier, distance_function, device: see :py:class:`EVM`
  """

  def __init__(self,
    tailsize,
    cover_threshold = None,
    distance_multiplier = 0.5,
    distance_function = 'cosine',
    device = "cuda"
  ):

    if isinstance(tailsize, (str, h5py.Group)):
      return self.load(tailsize)

    self.tailsize = tailsize
    self.cover_threshold = cover_threshold
    self.distance_function = distance_function.lower()
    self.distance_multiplier = distance_multiplier
    self.device = device

    self._evms = None


  def _train_evms(self, data, labels, extra_negatives):
    """Internal function to get enumerate through the number of classes to train, call the _train function, and return the models. Do not call directly. """
    models = []
    arguments = ((self, i) for i in range(len(data)))
    
    for i, arg in enumerate(arguments):
      if labels is not None:
        label = labels[i]
      else:
        label = None
      print("#################################################################")
      t1 = time.time()
      print("Now Training Class: %d / %d " % (i + 1, len(data)))
      models.append(_train(arg, data, label, self.distance_multiplier, self.distance_function, extra_negatives))
      t2 = time.time()
      print("Training time of class %d : %fs" % (i + 1, (t2 - t1)))
      print("#################################################################")
    return models

  def _train_evms_update(self, features, target_label, distance_multiplier, new, extra_negatives=None):
    """Internal function to call _train and return a model or update an exisiting model. Do not call directly. """
    
    arg = (self, 0)
    logger.info("#################################################################")
    t1 = time.time()
    logger.info("Training new points")
    new_model = _train(arg, features, target_label, distance_multiplier , self.distance_function, extra_negatives)
    t2 = time.time()
    logger.info("Training time of new points : %fs" % (t2 - t1))
    logger.info("#################################################################")
    if new:
      return [new_model]
    else:
      logger.info("Updating existing EVM model")
      class_len = len(self._evms[target_label]._positives)
      
      # combine EVM model's features with new features
      self._evms[target_label]._positives =  torch.cat((self._evms[target_label]._positives, new_model._positives),axis=0)
        
      # combine EVM model's weibulls with new weibulls
      params = self._evms[target_label]._extreme_vectors.return_all_parameters()    
      params2 = new_model._extreme_vectors.return_all_parameters()   
      params['Scale'] = torch.cat((params['Scale'].cpu(), params2['Scale'].cpu()),0)
      params['Shape'] = torch.cat((params['Shape'].cpu() ,params2['Shape'].cpu()),0)
      params['smallScoreTensor'] = torch.cat((params['smallScoreTensor'].cpu(), params2['smallScoreTensor'].cpu()),0)
        
      # combine EVM model's extreme vectors and covered vectors with new extreme vectors and covered vectors
      self._evms[target_label]._extreme_vectors_indexes = torch.cat((self._evms[target_label]._extreme_vectors_indexes, new_model._extreme_vectors_indexes+class_len),0)
      self._evms[target_label]._extreme_vectors = weibull.weibull(params)
      for i in range(len(new_model._extreme_vectors_indexes)):
        new_model._covered_vectors[i] + class_len
      self._evms[target_label]._covered_vectors.extend(new_model._covered_vectors)

  def train(self, class_features, labels=None, extra_negatives=None):
    """This function trains a separate EVM model for each class in class_features.

    Parameters
    ----------

    class_features : [torch.Tensor's]
      Each element (Tensor) in the list contains the training features of one class, in the order `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
      
    labels : [string or int] or ``None``
      Each element (string or int) in the list contains a label for the respective Tensor in class_features.
      The `len` of labels must be identical to class_features.
    
    extra_negatives : torch.Tensor or ``None``
      Extra negatives used for computation.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
    """
    self._evms = None
    
    # check if number of labels matches the number of classes
    if labels is not None and len(labels)==len(class_features):
      labels = labels
    else:
      if labels is not None:
        raise RuntimeError("The number of labels does not match the number of classes")
    
    if extra_negatives is not None and len([extra_negatives])==1:
      extra_negatives = extra_negatives
    else:
      if extra_negatives is not None:
        raise RuntimeError("Extra negatives must be a single tensor")
        
    # now, compute EVMs
    logger.info("Training %d EVM's", len(class_features))
    self._evms = self._train_evms(class_features,labels,extra_negatives)

  def train_update(self, new_points, label, distance_multiplier, extra_negatives=None):
    """This function updates an existing EVM model or trains a new one using new_points.

    Parameters
    ----------

    new_points: torch.Tensor
      New points to update an exisitng EVM class or train a new one.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
      
    label: string or int
      The label to be associated with new_points.
    
    distance_multiplier: int or float
      The distance multiplier to use when creating the MR object.
    
    extra_negatives: torch.Tensor or ``None``
      Extra negatives used for computation.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
    """
    if self._evms is None:
      raise RuntimeError("The model has not been trained yet")
    
    # get labels/features of exisiting classes
    labels = []
    train_features = []
    
    for i in range(self.size):
      train_features.append(self._evms[i]._positives)
      labels.append(self._evms[i]._label)

    # convert label of new points to a string    
    label = str(label)
    
    # combine data
    new_points = [new_points]
    new_points.extend(train_features)
    
    # process data accordingly (label exists or not)
    if label in labels:
      # retrieve index of the evm object of interest
      evms_index = labels.index(label)
      print('Label found, updating existing class and adding new points')
    
      # set boolean to false (not a new class)
      new = False

      # compute new EVM
      self._train_evms_update(new_points, evms_index, distance_multiplier, new, extra_negatives)
        
    else:
      print('Label not found, adding a new class')
      # set boolean to true (a new class)
      new = True
    
      # compute new EVM
      self._evms.extend(self._train_evms_update(new_points, label, distance_multiplier, new, extra_negatives))
    

  def probabilities(self, points):
    """Computes the probabilities for all EVMs for the given data points.

    Parameters
    ----------

    points: torch.Tensor
      The points, for which to compute the probability.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
    

    Returns
    -------

    [[[float]]] : a three-dimensional list of probabilities for each point, each evm and each extreme vector inside the evm.
      Indices for the probabilities are ``(point, evm, extreme_vector)``.
    """
    logger.info("Compute probabilities for all %d EVMS for the given %d data points\n" % (len(self._evms),len(points)))
    probs = []
    for i, evm in enumerate(self._evms):
      logger.info("#################################################################")
      t1 = time.time()
      logger.info("Now computing probabilities for EVM: %d / %d " % (i + 1, len(self._evms)))
      probs += [evm.probabilities(points)]
      logger.info("Size of propbs list: %d " % (sys.getsizeof(probs)))
      t2 = time.time()
      logger.info("Computing time of EVM %d : %fs" % (i + 1, (t2 - t1)))
      logger.info("#################################################################")
    # re-arange such that the first index is the point, the second the evm and the third the extreme vector
    return [[probs[e][p] for e in range(self.size)] for p in range(len(points))]


  def class_probabilities(self, points = None, probabilities = None):
    """Computes the maximum probabilities over all extreme vectors per class.

    Parameters
    ----------

    points: torch.Tensor
      The points, for which to compute the probability. Can be omitted when the ``probabilities`` parameter is given.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
    
    probabilities : [[[float]]] or ``None``
      The probabilities that were returned by the :py:meth:`probabilities` function.
      If not given, they are first computed from the given ``points``.


    Returns
    -------

    [[float]]
      The maximum probability of inclusion per class (i.e., per evm) for each of the points.
      The index order is ``(point, evm)``.
    """
    # compute probabilities
    if probabilities is None: probabilities = self.probabilities(points)
    # compute maximum per class
    return numpy.array([[numpy.max(feature_probs[i]) for i in range(len(feature_probs))] for feature_probs in probabilities])


  def max_probabilities(self, points = None, probabilities = None):
    """Computes the maximum probabilities and their accoring exteme vector for the given data points.

    Parameters
    ----------

    points: torch.Tensor
      The points, for which to compute the probability. Can be omitted when the ``probabilities`` parameter is given.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.
    
    probabilities : [[[float]]] or ``None``
      The probabilities that were returned by the :py:meth:`probabilities` function.
      If not given, they are first computed from the given ``points``.


    Returns
    -------

    [float]
      The maximum probability of inclusion for each of the points.

    [(int,int)]
      The list of tuples of indices into :py:attr:`evms` and their according `:py:attr:`EVM.extreme_vectors` for the given points.
    """

    # compute probabilities
    if probabilities is None: probabilities = self.probabilities(points)
    # get maximum probability per EVM
    indices = []
    for p in range(len(probabilities)):
      # compute maximum indices for all evs per evm
      max_per_ev = [numpy.argmax(probabilities[p][e]) for e in range(self.size)]
      max_per_evm = numpy.argmax([probabilities[p][e][max_per_ev[e]] for e in range(self.size)])
      indices.append((max_per_evm, max_per_ev[max_per_evm]))

    # return maximum probabilities and their according indexes
    return [probabilities[i][e][m] for i,(e,m) in enumerate(indices)], indices


  def save(self, h5):
    """Saves this object to HDF5

    Parameters
    ----------

    h5 : ``str`` or :py:class:`h5py.File` or :py:class:`h5py.Group`
      The name of the file to save, or the (subgroup of the) HDF5 file opened for writing.
    """
    if self._evms is None:
      raise RuntimeError("The model has not been trained yet")
    # open file for writing; create if not existent
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'w')

    # write EVMs
    for i, evm in enumerate(self._evms):
      evm.save(h5.create_group("EVM-%d" % (i+1)))

  def load(self, h5):
    """Loads this object from HDF5.

    Parameters
    ----------

    h5 : ``str``
      The name of the file to load
    """
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'r')

    # load evms
    self._evms = []
    i = 1
    while "EVM-%d" % i in h5:
      self._evms.append(EVM(h5["EVM-%d" % (i)], log_level='debug'))
      i += 1
        

  @property
  def size(self):
    """The number of EVM's."""
    if self._evms is not None:
      return len(self._evms)
    else:
      return 0

  @property
  def evms(self):
    """The EVM's, in the same order as the training classes."""
    return self._evms
