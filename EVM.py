import numpy
import weibull
import torch
import h5py
import pickle
import sys
from gpu_pairwise_distances_torch import gpu_pairwise_distances
import logging
import time
from tqdm import tqdm
logger = logging.getLogger("EVM")


class EVM (object):

  """This class represents the Extreme Value Machine.

  The constructor can be called in two different ways:

  1. Creating an empty (untrained) machine by setting the ``tailsize`` to any positive integer.
     All other parameters can be set as well.
     Please :py:meth:`train` the machine before using it.

     Example::

       EVM(100, 0.5, distance_function = 'cosine')

  2. Loading a pre-trained machine by providing the filename of -- or the :py:class:`h5py.Group` inside -- the HDF5 file.
     All other parameters are ignored, as they are read from the file.

     Example::

       h5 = h5py.File("EVM.hdf5", 'r')
       EVM(h5["/some/group"])
  """

  def __init__(self,
    tailsize,
    cover_threshold = None,
    distance_multiplier = 0.5,
    distance_function = 'cosine',
    log_level = 'info',
    device = 'cuda'
  ):
    self.log_level = log_level
    if isinstance(tailsize, (str, h5py.Group)):
      return self.load(tailsize)

    self.device = device

    self.tailsize = tailsize
    self.cover_threshold = cover_threshold
    self.distance_function = distance_function.lower()
    self.distance_multiplier = distance_multiplier

    self._positives = None
    self._margin_weibulls = None
    self._extreme_vectors = None
    self._extreme_vectors_indexes = None
    self._covered_vectors = None
    self._label = None

  def _fit_weibull(self, distances, distance_multiplier):
    """Internal function to do a weibull fitting on distances. Do not call directly."""
    mr = weibull.weibull()
    mr.FitLow(distances.double()*distance_multiplier, self.tailsize, 0)
    return mr

  def _distances(self, negatives):
    """Internal function to compute distances between positives and negatives. Do not call directly."""
    # compute distances between positives and negatives
    start_time = time.time()
    logger.info('Step 1: Computing %d distances between %d positive and %d negative points' % (len(self._positives) * len(negatives), len(self._positives), len(negatives)))
    distances = gpu_pairwise_distances(self._positives, negatives, self.distance_function, 0)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return distances


  def _positive_distances(self):
    """Computes the distances between positives. Do not call directly."""
    start_time = time.time()
    logger.info("Step 5: Computes the distances between positives")
    distances = gpu_pairwise_distances(self._positives, self._positives, self.distance_function, 0)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return distances


  def _set_cover(self):
    """Internal function to deduce the model to keep only the most informant features (the so-called Extreme Vectors). Do not call directly."""
    N = len(self._positives)
    
    logger.info("Step 4: Computing set-cover for %d points" % N)
    # compute distances between positives, if not given
    distances = self._positive_distances()
    
    start_time = time.time()

    # compute probabilities
    probabilities = self._margin_weibulls.wscore(distances)
    probabilities = probabilities.cuda()

    # threshold by cover threshold
    thresholded = probabilities >= self.cover_threshold
    thresholded[torch.eye(probabilities.shape[0]).type(torch.BoolTensor)] = True

    # greedily add points that cover most of the others
    covered = torch.zeros(N).type(torch.bool)
    _extreme_vectors = []
    self._covered_vectors = []

    pbar = tqdm(total=N)
    while not torch.all(covered).item():
      sorted_indices = torch.topk(torch.sum(thresholded[:,~covered],dim=1), len(_extreme_vectors)+1, sorted=False).indices
      for indx, sortedInd in enumerate(sorted_indices.tolist()):
        if sortedInd not in _extreme_vectors:
          break
      else:
        print("ENTERING INFINITE LOOP ... EXITING")
        break
      covered_by_current_ev = torch.nonzero(thresholded[sortedInd,:], as_tuple=False)
      pbar.update(torch.sum(thresholded[sortedInd,~covered]).item())
      covered[covered_by_current_ev]=True
      _extreme_vectors.append(sortedInd)
      self._covered_vectors.append(covered_by_current_ev)
    pbar.close()
    logger.debug(_extreme_vectors)
    self._extreme_vectors_indexes = torch.tensor(_extreme_vectors).to(self.device)
    params = self._margin_weibulls.return_all_parameters()
    scale = torch.gather(params['Scale'].to(self.device),0,self._extreme_vectors_indexes)
    shape = torch.gather(params['Shape'].to(self.device),0,self._extreme_vectors_indexes)
    smallScore = torch.gather(params['smallScoreTensor'][:,0].to(self.device),0,self._extreme_vectors_indexes)
    obj = dict(Scale = scale, Shape = shape, signTensor = params['signTensor'], translateAmountTensor = params['translateAmountTensor'], smallScoreTensor = smallScore)
    self._extreme_vectors = weibull.weibull(obj)
    self.log("Obtained %d extreme vectors", self._extreme_vectors_indexes.shape[0])
    logger.info("--- %s seconds ---" % (time.time() - start_time))


  def train(self, positives, negatives=None, label=None):
    """Trains the extreme value machine using the given positive samples of the class, and the negative samples that do not belong to the class.

    Parameters
    ----------
    
    positives : torch.Tensor
      The points of the class to model.

    negatives : torch.Tensor or ``None``
      Points of other classes, used to compute the distribution model.
      Ignored when ``distances`` are not ``None``.

    distances : torch.Tensor or ``None``
      Distances between positives and negatives, used to compute the distribution model.
      If no distances are given, they are computed from the ``negatives``.
      A different number of distances can be provided for each of the ``positives``.

    """

    assert negatives is not None, "Negatives must not be `None`"

    # store all positives and their according Weibull distributions as the model
    self._positives = positives

    # if given, store label to the model
    if label is not None:
      self._label = str(label)
    else:
      self._label = None

    # now, train the margin probability
    # first, train the weibull models for each positive point
    distances = self._distances(negatives)

    # compute weibulls from distances
    logger.info("Step 2: Compute weibulls from distances, Tailsize: %d " % (self.tailsize))
    self._margin_weibulls = self._fit_weibull(distances,self.distance_multiplier)

    # then, perform model reduction using set-cover
    logger.info("Step 3: Perform model reduction using set-cover")
    self._set_cover()


  def probabilities(self, points):
    """Computes the probabilities for all extreme vectors for the given points.

    Parameters
    ----------

    points: torch.Tensor
      The points, for which to compute the probability.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical

    Returns
    -------

    2D :py:class:`numpy.ndarray` of floats
      The probability of inclusion for each of the points to each of the extreme vectors.
      The array indices are in order `(point, extreme_vector)`.
    """
    assert points is not None, "points must not be `None`"
    new = [self._positives[e] for e in numpy.array(self._extreme_vectors_indexes.cpu())]
    logger.info("Step 1: Compute distances on GPU")
    distances = gpu_pairwise_distances(torch.stack(new), points, self.distance_function, 0)
    logger.info("Size of distances list: %d " % (sys.getsizeof(distances)))
    logger.info("Step 2: Compute weibull scores on GPU")
    w_scores = self._extreme_vectors.wscore(torch.transpose(distances,1,0))
    logger.info("Size of weibull scores: %d " % (sys.getsizeof(w_scores)))
    # finally, return the scores in corrected order
    return numpy.array(w_scores.cpu())
    

  def max_probabilities(self, points = None, distances = None, probabilities = None):
    """Computes the maximum probabilities and their accoring exteme vector for the points.

    Parameters
    ----------

    points: torch.Tensor
      The points, for which to compute the probability. Can be omitted when the ``probabilities`` parameter is given.
      Each element in the Tensor contains the feature of one point, and overall size `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.

    distances: torch.Tensor or ``None``
      The distances between points and the :py:attr:`extreme_vectors`, for which to compute the probabilities.
      Distances should be computed with the same distance function as used during training.
      Ignored when ``probabilities`` are given.

    probabilities : 2D :py:class:`numpy.ndarray` of floats or ``None``
      The probabilities that were returned by the :py:meth:`probabilities` function.
      If not given, they are first computed from the given ``points`` or ``distances``.


    Returns
    -------

    [float]
      The maximum probability of inclusion for each of the points.

    [int]
      The list of indices into `:py:attr:`extreme_vectors` for the given points.
    """

    if probabilities is None: probabilities = self.probabilities(points, distances)
    indices = numpy.argmax(probabilities, axis=1)
    return [probabilities[i, j] for i,j in enumerate(indices)], indices


  def save(self, h5):
    """Saves this object to HDF5

    Parameters
    ----------

    h5 : ``str`` or :py:class:`h5py.File` or :py:class:`h5py.Group`
      The name of the file to save, or the (subgroup of the) HDF5 file opened for writing.
    """
    if self._positives is None:
      raise RuntimeError("The model has not been trained yet")
    # open file for writing; create if not existent
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'w')
    # write features
    h5["Features"] = self._positives
    
    # write extreme vectors
    params = self._extreme_vectors.return_all_parameters()
    e = h5.create_group('ExtremeVectors')
    e.create_dataset('scale',data=params['Scale'].cpu())
    e.create_dataset('shape',data=params['Shape'].cpu())
    e.create_dataset('sign',data=params['signTensor'])
    e.create_dataset('translateAmount',data=params['translateAmountTensor'])
    e.create_dataset('smallScore',data=params['smallScoreTensor'].cpu())
    
    e.create_dataset('indexes',data = self._extreme_vectors_indexes.cpu())
    
    # write covered vectors
    for i in range(len(self._extreme_vectors_indexes)):
      e.create_dataset('CoveredVectors/'+str(i),data=self._covered_vectors[i].cpu().numpy())
    
    # write other parameteres (as attributes)
    e.attrs["Distance"] = numpy.string_(self.distance_function)
    e.attrs["Tailsize"] = self.tailsize
    e.attrs["Label"] = self._label if self._label is not None else -1
    h5["ExtremeVectors"].attrs["CoverThreshold"] = self.cover_threshold if self.cover_threshold is not None else -1.


  def load(self, h5):
    """Loads this object from HDF5.

    Parameters
    ----------

    h5 : ``str`` or :py:class:`h5py.File` or :py:class:`h5py.Group`
      The name of the file to load, or the (subgroup of the) HDF5 file opened for reading.
    """
    # open file for reading
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'r')
    
    # load features
    self._positives = torch.from_numpy(h5["Features"][:])
    
    # load extreme vectors
    e = h5['ExtremeVectors']
    obj = dict(Scale=torch.from_numpy(e['scale'][()]),Shape = torch.from_numpy(e['shape'][()]),signTensor = torch.tensor(e['sign'][()]),translateAmountTensor = torch.LongTensor(e['translateAmount'][()]),smallScoreTensor = torch.from_numpy(e['smallScore'][()]))
    self._extreme_vectors = weibull.weibull(obj)
    self._extreme_vectors_indexes = torch.tensor(e['indexes'][()])

    cv = []
    # load covered indices
    for i in range(len(self._extreme_vectors_indexes)):
      cv.append(torch.from_numpy(numpy.array(e['CoveredVectors/'+str(i)][()])))
    self._covered_vectors = cv
    
    # load other parameteres
    self.distance_function = e.attrs["Distance"]
    self.tailsize = e.attrs["Tailsize"]
    self._label = e.attrs["Label"]
    if self._label == -1: self._label = None
    self.cover_threshold = e.attrs["CoverThreshold"]
    if self.cover_threshold == -1.: self.cover_threshold = None
  
    
  def log(self, *args, **kwargs):
    """Logs the given message using debug or info logging, see :py:attr:`log_level`"""
    {'info' : logger.info, 'debug' : logger.debug}[self.log_level](*args, **kwargs)


  @property
  def size(self):
    """The number of extreme vectors."""
    if self._extreme_vectors_indexes is not None:
      return self._extreme_vectors_indexes.shape[0]
    else:
      return 0

  @property
  def shape(self):
    """The shape of the features that the :py:meth:`probability` function expects."""
    if self._positives is not None:
      return self._positives[0].shape

  @property
  def extreme_vectors(self):
    """The extreme vectors that this class stores."""
    if self._extreme_vectors is not None:
      return self._positives[self._extreme_vectors_indexes]

  @property
  def label(self):
    """The label that this class stores."""
    if self._label is not None:
      return self._label

  def covered(self, i):
    """Returns the vectors covered by the extreme vector with the given index"""
    if self._extreme_vectors is not None and i < self.size:
      return [self._positives[c] for c in self._covered_vectors[i]]

