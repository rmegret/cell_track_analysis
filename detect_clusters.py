import matplotlib.pyplot as plt

import skimage
from skimage.io import imread
from skimage.measure import regionprops
from skimage.color import hsv2rgb

import numpy as np
import pandas as pd
import math

from numpy.random import RandomState
from matplotlib import collections  as mc

from skimage import measure, segmentation, util, color

import networkx as nx

from scipy.optimize import linear_sum_assignment

import warnings



## PLOTTING

def xylim(r, ax=None):
  """xylim([xmin, max, ymin, ymax]): set x and y limits
  ax: axes to set to limits to (default current axes)
  """
  xmin,xmax,ymin,ymax=r
  if (ax is None):
     ax = plt.gca()
  ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)


def get_rgbmap(L, seed=0, hsv=True):
  rs = RandomState(seed)
  if (hsv):
    hsvmap = rs.rand( L.max()+2, 3 )
    hsvmap[:,1] = 1.0#hsvmap[:,1]*0.5+0.5
    hsvmap[:,2] = hsvmap[:,2]*0.5+0.5
    rgbmap = skimage.color.hsv2rgb(hsvmap)
  else:
    rgbmap = rs.rand( L.max()+2, 3 )
  return rgbmap


def labels2rgb(L, rgbmap=None, seed=0, hsv=False):
  if rgbmap is None:
    rs = RandomState(seed)
    if (hsv):
      hsvmap = rs.rand( L.max()+2, 3 )
      hsvmap[:,1] = 1.0#hsvmap[:,1]*0.5+0.5
      hsvmap[:,2] = hsvmap[:,2]*0.5+0.5
      rgbmap = skimage.color.hsv2rgb(hsvmap)
    else:
      rgbmap = rs.rand( L.max()+2, 3 )
    rgbmap[0,:] = 0
    rgbmap[-1,:] = 1
  return rgbmap[L,:]

def plot_cell_axis(L,rd=5):
  #rd=5
  P = regionprops(L)
  C = np.array([p.centroid for p in P])
  a = np.array([p.orientation for p in P])

  # C2: [ [y,x], ... ]
  C2 = np.array( [ [C[k,0] + np.cos(a[k]) * rd,  C[k,1] + np.sin(a[k]) * rd ] for k in range(C.shape[0]) ] )

  # lines: [ [(x1,y1),(x2,y2)], ... ]
  lines = [ [ (C[k,1] + np.sin(a[k]) * rd,  C[k,0] + np.cos(a[k]) * rd),  (C[k,1] - np.sin(a[k]) * rd,  C[k,0] - np.cos(a[k]) * rd) ] for k in range(C.shape[0]) ]

  lc = mc.LineCollection(lines, linewidths=1)
  plt.gca().add_collection(lc)
  plt.plot(C[:,1],C[:,0],'r.')
  plt.plot(C2[:,1],C2[:,0],'b.', markersize=2)


def vel2rgb(velocity, velocity_max=1.0, clip_magnitude=True):
  """velocity in y,x order"""
  V = velocity/velocity_max
  V[np.isnan(V[...,0]),:] = 0

  angle = np.arctan2(V[..., 1], -V[..., 0])  # Range [-pi, pi]
  hue = (angle / (2 * np.pi)) % 1  # Normalize to [0, 1]
  
  magnitude = np.linalg.norm(V, axis=-1)  # Compute magnitude
  if (clip_magnitude):
    saturation = np.clip(magnitude, 0, 1)  # Ensure within [0, 1]
  else:
    saturation = magnitude.copy()
    saturation[saturation>1] = 0
  
  h,w,_ = velocity.shape
  value = np.ones((h, w))  # Max brightness
  hsv_image = np.stack([hue, saturation, value], axis=-1)
  
  # Convert HSV to RGB
  rgb = hsv2rgb(hsv_image)
  return rgb



## PLOT RAG

def to_pandas_nodelist(rag):
  return pd.DataFrame([i[1] for i in rag.nodes(data=True)], index=[i[0] for i in rag.nodes(data=True)])


# Custom display RAG function

def show_rag2(
    labels,
    rag,
    image,
    border_color='black',
    edge_width=1.5,
    edge_cmap='magma',
    img_cmap='bone',
    node_feature=None,
    node_cmap='bwr',
    ax=None,
    dataname='weight',
    hide_zero=False,
    edge_filter=None,
    ignore_node=None,
    show_filtered=False,
    velocity_front=False,
    velocity_back=False,
    velocity_scale=1.0,
    velocity_min=1.0,
    arrow_config=None,
    split_merge_indicator=False,
):
    """Show a Region Adjacency Graph on an image.
    Adapted from https://github.com/scikit-image/scikit-image/blob/v0.25.0/skimage/graph/_rag.py:show_rag

    Given a labelled image and its corresponding RAG, show the nodes and edges
    of the RAG on the image with the specified colors. Edges are displayed between
    the centroid of the 2 adjacent regions in the image.

    Args:
      labels (ndarray, shape (M, N)):
          The labelled image.
      rag (RAG): The Region Adjacency Graph.
      image : ndarray, shape (M, N[, 3])
          Input image. If `colormap` is `None`, the image should be in RGB
          format.
      border_color : color spec, optional
          Color with which the borders between regions are drawn.
      edge_width : float, optional
          The thickness with which the RAG edges are drawn.
      edge_cmap : :py:class:`matplotlib.colors.Colormap`, optional
          Any matplotlib colormap with which the edges are drawn.
      node_cmap
      img_cmap : :py:class:`matplotlib.colors.Colormap`, optional
          Any matplotlib colormap with which the image is draw. If set to `None`
          the image is drawn as it is.
      in_place : bool, optional
          If set, the RAG is modified in place. For each node `n` the function
          will set a new attribute ``rag.nodes[n]['centroid']``.
      ax : :py:class:`matplotlib.axes.Axes`, optional
          The axes to draw on. If not specified, new axes are created and drawn
          on.
      dataname: which data name to show.
      hide_zero: hide if data value is zero
      ignore_node: node to remove before display

    Returns:
       lc : :py:class:`matplotlib.collections.LineCollection`
         A collection of lines that represent the edges of the graph. It can be
         passed to the :meth:`matplotlib.figure.Figure.colorbar` function.

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> from skimage import data, segmentation, graph
    >>> import matplotlib.pyplot as plt
    >>>
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> g =  graph.rag_mean_color(img, labels)
    >>> lc = graph.show_rag(labels, g, img)
    >>> cbar = plt.colorbar(lc)
    """
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    rag = rag.copy()
    if (ignore_node is not None):
        rag.remove_node(ignore_node)  # Remove background node

    if ax is None:
        fig, ax = plt.subplots()
    out = util.img_as_float(image, force_copy=True)

    if img_cmap is None:
        if image.ndim < 3 or image.shape[2] not in [3, 4]:
            msg = 'If colormap is `None`, an RGB or RGBA image should be given'
            raise ValueError(msg)
        # Ignore the alpha channel
        out = image[:, :, :3]
    else:
        img_cmap = plt.get_cmap(img_cmap)
        out = color.rgb2gray(image)
        # Ignore the alpha channel
        out = img_cmap(out)[:, :, :3]

    edge_cmap = plt.get_cmap(edge_cmap)

    # Handling the case where one node has multiple labels
    # offset is 1 so that regionprops does not ignore 0
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for n, d in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1

    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)

    for (n, data), region in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))

    cc = colors.ColorConverter()
    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)

    ax.imshow(out)

    # Defining the end points of the edges
    # The tuple[::-1] syntax reverses a tuple as matplotlib uses (x,y)
    # convention while skimage uses (row, column)
    if (edge_filter):
        lines = [
            [rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
            for (n1, n2, d) in rag.edges(data=True) if edge_filter(rag,n1,n2)
        ]
        edge_weights = [d[dataname] for x, y, d in rag.edges(data=True) if edge_filter(rag,x,y)]
    elif (hide_zero):
        lines = [
            [rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
            for (n1, n2, d) in rag.edges(data=True) if d[dataname]!=0.0
        ]
        edge_weights = [d[dataname] for x, y, d in rag.edges(data=True) if d[dataname]!=0.0]
    else:
        lines = [
            [rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
            for (n1, n2) in rag.edges()
        ]
        edge_weights = [d[dataname] for x, y, d in rag.edges(data=True)]

    lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    lc.set_array(np.array(edge_weights))
    ax.add_collection(lc)

    if (edge_filter and show_filtered):
        lines = [
            [rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
            for (n1, n2, d) in rag.edges(data=True) if not edge_filter(rag,n1,n2)
        ]
        edge_weights = [d[dataname] for x, y, d in rag.edges(data=True) if not edge_filter(rag,x,y)]
        edge_width_filtered = 1.0
        lc = LineCollection(lines, linewidths=edge_width_filtered, cmap=edge_cmap, linestyle='dotted')
        lc.set_array(np.array(edge_weights))
        ax.add_collection(lc)

    _arrow_config = dict(length_includes_head=True,
          width=1, head_width=4, head_length=4, color=None)
    if (arrow_config is not None):
      _arrow_config.update(arrow_config)
    if (velocity_front): # Show velocity from current frame to next
      arrows = []
      if (_arrow_config['color'] is None): _arrow_config['color']='red'
      for u in rag.nodes:
        node = rag.nodes[u]
        c = node['centroid']
        v = node.get('velocity_front')
        if (v is None) or (np.isnan(v[0])): # Ending track
          #ax.text(c[1],c[0],'E', color='red', transform=ax.transData)
          ax.plot(c[1],c[0],'x', color=_arrow_config['color'], markersize=5)
          continue
        if (np.linalg.norm(v) <= velocity_min):
          ax.plot(c[1],c[0],'o', color=_arrow_config['color'], markersize=5, markerfacecolor='none')
          continue
        v = v*velocity_scale
        #ac = LineCollection(lines, linewidths=edge_width_filtered, cmap=edge_cmap, linestyle='dotted')
        arrows.append(ax.arrow(c[1],c[0], v[1], v[0], **_arrow_config))
    if (velocity_back): # Show velocity from previous frame to current
      arrows = []
      if (_arrow_config['color'] is None): _arrow_config['color']='blue'
      for u in rag.nodes:
        node = rag.nodes[u]
        c = node['centroid']
        v = node.get('velocity_back')
        if (v is None) or (np.isnan(v[0])): # Starting track
          #ax.text(c[1],c[0],'S', color='red', transform=ax.transData)
          ax.plot(c[1],c[0],'+', color=_arrow_config['color'], markersize=5)
          continue
        if (np.linalg.norm(v) <= velocity_min):
          ax.plot(c[1],c[0],'o', color=_arrow_config['color'], markersize=5, markerfacecolor='none')
          continue
        v = v*velocity_scale
        c = c-v
        #ac = LineCollection(lines, linewidths=edge_width_filtered, cmap=edge_cmap, linestyle='dotted')
        arrows.append(ax.arrow(c[1],c[0], v[1], v[0], **_arrow_config))

    if (node_feature is not None):
      for u in rag.nodes:
        node = rag.nodes[u]
        c = node['centroid']
        if (node_feature=='split_merge_indicator'):
          # if (node.get('split_children')):
          #   ax.plot(c[1],c[0],'v', color='r', markersize=5)
          # if (node.get('merge_parents')):
          #   ax.plot(c[1],c[0],'^', color='b', markersize=5)
          if (node.get('split_parent')):
            ax.plot(c[1],c[0],'v', color='r', markersize=5)
          if (node.get('merge_child')):
            ax.plot(c[1],c[0],'^', color='b', markersize=5)
          continue
        att = node[node_feature]
        if (att is None) or (np.isnan(att)):
          ax.plot(c[1],c[0],'+', color='k', markersize=5)
          continue
        if (node_feature=='area_change_back'):
          ax.plot(c[1],c[0],'o', color=plt.cm.bwr(np.clip(att/2+0.5,0,1)), markersize=5)
        if (node_feature=='trackid'):
          ax.plot(c[1],c[0],'o', color=labels2rgb(att), markersize=5)

    return lc


def _mapping_attribute(L, rag, featurename, default=0):
  # Map graph attribute dataname to pixels
  # Assume node 0 is background
  mapping = np.zeros( (np.max(L, None)+1,) )
  #print(mapping.shape)
  for u in rag.nodes:
    #print(u)
    mapping[u] = rag.nodes[u].get(featurename,default)
  return mapping

def rag_attribute_image(L, rag, featurename, default=0, mode=None, velocity_max = 1.0):
  """Map RAG node features `featurename` back to an image"""
  mapping = _mapping_attribute(L, rag, featurename, default)
  return mapping[L]

def rag_velocity_image(L, rag, featurename, default=None):
  """Map RAG node 2D velocity features `featurename` back to an image"""
  if (default is None): 
    default = np.zeros( (2,) )
  else:
    default = np.array(default)
  mapping = np.zeros( (np.max(L, None)+1,2,) )
  for u in rag.nodes:
    mapping[u,:] = rag.nodes[u].get(featurename,default)
  return mapping[L]




## COMPUTE RAG

def generate_graph(L):
  """Generate a Region Adjencency Graph (RAG) from labeled image L, and attach features to nodes and edges
  Parameters
  ----------
  L: ndarray, shape (M, N)
     The labelled image.

  Returns
  -------
  rag: networkx graph extracted from the label image

  Note: node 0, which is usually the background, is kept. Can be removed afterwards with `rag.remove_node(0)`
  """

  rag = skimage.graph.rag_boundary(L, np.ones_like(L,dtype=np.float32))
  #rag.remove_node(0)  # Remove background node # Keep the background node in the graph for countour ratio computation purpose

  for u in rag.nodes:
      node = rag.nodes[u]
      node['id'] = u

  P = regionprops(L)

  labels = np.array([p.label for p in P],dtype=np.uint16)
  label2k = { labels[k]:k for k in range(labels.size) }

  # ADD ANGLE AND RELATIVE ANGLE FEATURES TO RAG
  ar = np.array([p.area for p in P])
  for (u) in rag.nodes:
    if (u==0): 
      rag.nodes[u]['area'] = 0.0
      continue
    ku = label2k[u]
    rag.nodes[u]['area'] = ar[ku]

  # ADD ANGLE AND RELATIVE ANGLE FEATURES TO RAG
  a = np.array([p.orientation for p in P])

  for (u) in rag.nodes:
    if (u==0): 
      rag.nodes[u]['angle'] = 0.0
      continue
    ku = label2k[u]
    rag.nodes[u]['angle'] = a[ku]

  for (u,v) in rag.edges:
    if (u==0) or (v==0): 
      rag.get_edge_data(u,v)['relative_angle'] = 90.0
      continue

    au = rag.nodes[u]['angle']
    av = rag.nodes[v]['angle']

    relative_angle =  90.0 - abs( 90.0 - np.mod((au-av)/math.pi*180.0,180.0) ) 
    rag.get_edge_data(u,v)['relative_angle'] = relative_angle
    #print(u,v,relative_angle)
    
  # ADD CENTROID AND CENTROID DISTANCE FEATURES TO RAG
  #DD = D_centroid.toarray()
  C = np.array([p.centroid for p in P])

  for (u) in rag.nodes:
    if (u==0): 
      rag.nodes[u]['centroid'] = np.array([500.0,500.0])
      continue
    ku = label2k[u]
    rag.nodes[u]['centroid'] = C[ku,:]

  for (u,v) in rag.edges:
    if (u==0) or (v==0): 
      rag.get_edge_data(u,v)['centroid_distance'] = 1000.0
      continue
    #ku = label2k[u]
    #kv = label2k[v]

    cu = rag.nodes[u]['centroid']
    cv = rag.nodes[v]['centroid']

    centroid_distance = np.sum( (cu-cv)**2 )**0.5
    rag.get_edge_data(u,v)['centroid_distance'] = centroid_distance
    #print(u,v,centroid_distance)

  # ADD ANGLE AND RELATIVE ANGLE FEATURES TO RAG
  for (u) in rag.nodes:
    edges = rag.edges(u)
    boundary_count = np.sum([rag.get_edge_data(v,w)['count'] for v,w in edges])
    rag.nodes[u]['boundary_count'] = boundary_count

  for (u,v) in rag.edges:
    bcu = rag.nodes[u]['boundary_count']
    bcv = rag.nodes[v]['boundary_count']
    ce = rag.get_edge_data(u,v)['count']

    max_count_ratio = ce / np.min([bcu,bcv])
    rag.get_edge_data(u,v)['max_count_ratio'] = max_count_ratio

  return rag




## CLUSTERING

### CLUSTERING RULES

from dataclasses import dataclass

@dataclass
class EdgeFilterAll:
  """filter=EdgeFilterAll()"""
  def __call__(self, rag,u,v):
    return True

@dataclass
class EdgeFilterNone:
  """filter=EdgeFilterNone()"""
  def __call__(self, rag,u,v):
    return False
  
@dataclass
class EdgeFilterAngle:
  """filter=EdgeFilterAngle(amax)"""
  amax: float = 15
  def __call__(self, rag,u,v):
    dat = rag.get_edge_data(u,v)
    return (dat['relative_angle']<=self.amax)
  
@dataclass
class EdgeFilterAngleCentroid:
  """filter=EdgeFilterAngleCentroid(amax,dmax)"""
  amax: float = 15
  dmax: float = 20
  def __call__(self, rag,u,v):
    dat = rag.get_edge_data(u,v)
    return (dat['relative_angle']<=self.amax) and (dat['centroid_distance']<=self.dmax)

def EdgeFilterAngleCount(amax=15, count_min=20):
  return lambda rag,u,v: (rag.get_edge_data(u,v)['relative_angle']<=amax) and (rag.get_edge_data(u,v)['count']>=count_min)

@dataclass
class EdgeFilterAngleCountVelocity:
  """filter=EdgeFilterAngleCentroid(amax,dmax)"""
  amax: float = 15.0
  count_min: float = 20
  vdiffmax: float = 40.0
  def __call__(self, rag,u,v):
    dat = rag.get_edge_data(u,v)
    rv = dat.get('relative_velocity')
    if (rv is not None):
      velOk = (rv**2).sum()<self.vdiffmax**2
    else:
      velOk = True
    return (dat['relative_angle']<=self.amax) and (dat['count']>=self.count_min) and velOk
  
def EdgeFilterAngleRatio(amax=15, count_ratio_min=0.2):
  return lambda rag,u,v: (rag.get_edge_data(u,v)['relative_angle']<=amax) and (rag.get_edge_data(u,v)['max_count_ratio']>=count_ratio_min)


### ACTUAL CLUSTERING

def compute_clusters(rag, min_cluster_size=2, edge_filter=EdgeFilterAll()):
  """Compute connected components of RAG graph
  Default cluster is -1, background is 0
  """

  rag0 = rag.copy(); rag0.remove_node(0)
  H = nx.Graph(((u, v, e) for u,v,e in rag0.edges(data=True) if edge_filter(rag0,u,v)))

  CC = nx.connected_components(H)
  CC = list(CC)

  for u in rag.nodes:
    rag.nodes[u]['cluster']=-1  # Default cluster has id -1
  rag.nodes[0]['cluster']=0  # background has id 0
  for k,g in enumerate(CC):
    cluster_id = k+1
    if (len(g)<min_cluster_size): # Skip if cluster has only one region
      continue
    for u in g:
      rag.nodes[u]['cluster'] = cluster_id

  return H


## TRACKING

from sklearn.metrics.pairwise import pairwise_distances

def trackidFromDPTrack(rag2, df1, thresh = 3.0, next_id=None):
  # 1. Compute centroid distances
  df1 = df1.copy().rename(columns={'X (px)':'cx','Y (px)':'cy'})
  df1.loc[-1] = dict(trackid=0, frame=1, cx=0, cy=0)

  df2 = to_pandas_nodelist(rag2)
  df2['cx'] = df2['centroid'].apply(lambda v: v[1])
  df2['cy'] = df2['centroid'].apply(lambda v: v[0])
  
  df1 = df1.sort_values('trackid')
  df2 = df2.sort_values('id')

  C1 = df1[['cx','cy']].to_numpy()
  C2 = df2[['cx','cy']].to_numpy()
  n1 = C1.shape[0] #int(c1.max())+1
  n2 = C2.shape[0] #int(c2.max())+1
  
  c1 = df1['trackid'].to_numpy()  # C1 index k1 => trackid
  c2 = df2['id'].to_numpy()       # C2 index k2 => id

  #print(c1)
  #print(c2)

  #print(f'n1={n1}, n2={n2}')
  
  D = pairwise_distances(C1, C2, 'euclidean')

  #fig = plt.figure(figsize=(10,10))
  #plt.imshow(D<3)
  #plt.plot((D<3).sum(axis=0),'.')
  
  D[0,:] = 1e6
  D[:,0] = 1e6
  D[0,0] = 0 # Background matcheswith itself

  # 2. Find matches
  row_ind, col_ind = linear_sum_assignment(D)

  nm = row_ind.size
  if (thresh is not None):
    valid = np.zeros( (nm,), dtype=bool )
    for i in range(row_ind.size):
      valid[i] = D[row_ind[i],col_ind[i]] <= thresh

    valid_rows = row_ind[valid]
    valid_cols = col_ind[valid]
  else:
    valid_rows = row_ind
    valid_cols = col_ind

  nv = valid.sum()
  matched1 = np.sort(np.unique(valid_rows))
  unmatched1 = np.setdiff1d(range(n1), matched1)
  matched2 = np.sort(np.unique(valid_cols))
  unmatched2 = np.setdiff1d(range(n2), matched2)
  #nv, n1, n2, matched2.size, unmatched2.size

  N1 = int(c1.max())+1
  if (next_id is None): next_id = N1
  if (next_id < N1):
    warnings.warn('next_id < N1')
    next_id = N1

  N2 = int(c2.max())+1
  map2to1 = np.zeros( (N2,), dtype=int )
  map2to1[c2[valid_cols]] = c1[valid_rows]   # matched labels, reuse old id
  new_labels = range(next_id, next_id+unmatched2.size) # Create new ids for unmatched
  map2to1[c2[unmatched2]] = new_labels
  #map2to1[0] == 0  # Supposed to match itself already

  # 3. Update RAG
  for u in rag2.nodes:
    node = rag2.nodes[u]
    node['trackid'] = map2to1[node['id']]

  next_id2 = next_id+unmatched2.size

  print(f'Cell DPTrack match: matched = {matched2.size}, unmatched={unmatched1.size}, new = {unmatched2.size}, map2to1[0] = {map2to1[0]}')

  return next_id2

def match_cells(L1, L2, thresh = 0.3, next_id=None):
  # 1. Compute pixelwise overlap metrics between cells
  n1 = int(L1.max())+1
  n2 = int(L2.max())+1
  MI = np.zeros( (n1,n2) )
  l1 = L1.ravel()
  l2 = L2.ravel()
  for i in range(l1.size):
    MI[l1[i],l2[i]] += 1
  C1 = MI.sum(axis=1)
  C2 = MI.sum(axis=0)

  MU = C1.reshape( n1,1 ) + C2.reshape( 1,n2 ) - MI

  IOU = MI / MU
  IOU[np.isnan(IOU)] = 0

  IOU[0,:] = 0 # Remove background link to other regions
  IOU[:,0] = 0
  IOU[0,0] = n1*n2 # Background matcheswith itself

  # 2. Find matches
  row_ind, col_ind = linear_sum_assignment(-IOU)

  nm = row_ind.size
  valid = np.zeros( (nm,), dtype=bool )
  for i in range(row_ind.size):
    valid[i] = IOU[row_ind[i],col_ind[i]] >= thresh

  valid_rows = row_ind[valid]
  valid_cols = col_ind[valid]

  nv = valid.sum()
  matched1 = np.sort(np.unique(valid_rows))
  unmatched1 = np.setdiff1d(range(n1), matched1)
  matched2 = np.sort(np.unique(valid_cols))
  unmatched2 = np.setdiff1d(range(n2), matched2)
  #nv, n1, n2, matched2.size, unmatched2.size

  if (next_id is None):
    next_id = n1
  if (next_id < n1): 
    next_id = n1
  map2to1 = np.zeros( (n2,), dtype=int )
  map2to1[valid_cols] = valid_rows   # matched labels, reuse old id
  new_labels = range(next_id, next_id+unmatched2.size) # Create new ids for unmatched
  map2to1[unmatched2] = new_labels
  #map2to1[0] == 0  # Supposed to match itself already

  next_id = next_id+unmatched2.size

  print(f'Cell match: matched = {matched2.size}, new = {unmatched2.size}, map2to1[0] = {map2to1[0]}')

  return map2to1, next_id

def get_map_id_to_trackid(rag1, L1):
  n1 = int(L1.max())+1
  map1toT = np.zeros( (n1,), dtype=int )
  for u in rag1.nodes:
    node = rag1.nodes[u]
    map1toT[node['id']] = node['trackid']
  return map1toT

def track_cells(rag2, L2, rag1=None, L1=None, next_id=None):
  # If first frame
  if (rag1 is None):
    next_id = 1
    for u in rag2.nodes:
      node = rag2.nodes[u]
      node['trackid'] = node['id']
      if (node['trackid']>next_id):
        next_id = node['trackid']
    next_id = next_id+1
    return next_id

  # If tracking from existing frame
  map1toT = get_map_id_to_trackid(rag1, L1)
  LT = map1toT[L1]
  map2toT, next_id = match_cells(LT, L2, next_id=next_id) # Convert L2 id to trackid

  for u in rag2.nodes:
    node = rag2.nodes[u]
    node['trackid'] = map2toT[node['id']]

  return next_id


def matchClusters(rag1, rag2, thresh = 0.2, metric='iou', next_cid=None):
  # Match clusters, assume cells are already tracked

  if (rag1 is None):
    if (next_cid is None): next_cid = 1
    df2 = to_pandas_nodelist(rag2)
    n2 = int(df2.cluster.max())+1
    map2to1 = np.zeros( (n2,), dtype=int )
    unmatched2 = np.setdiff1d(df2.cluster.unique(), [0,-1])
    new_labels = range(next_cid, next_cid+unmatched2.size) # Create new ids for unmatched
    map2to1[unmatched2] = new_labels
    return map2to1, next_cid

  df1 = to_pandas_nodelist(rag1)
  df2 = to_pandas_nodelist(rag2)

  df1 = df1[df1.clustertrack>0]
  df1.set_index('trackid',inplace=True)
  df2 = df2[df2.cluster>0]
  df2.set_index('trackid',inplace=True)

  c1 = df1.clustertrack.rename('cluster1')
  c2 = df2.cluster.rename('cluster2')
  dfm = pd.concat([c1,c2],axis=1) # join based on index which is trackid
  dfm = dfm.dropna(axis=0) # drop cluster pairs with no matches
  
  #print(dfm)

  vc = pd.DataFrame(dfm.value_counts()) # Count common cells
  vc1 = c1.value_counts()
  vc2 = c2.value_counts()
  vc['count1'] = vc.index.map( lambda id: vc1[id[0]] )
  vc['count2'] = vc.index.map( lambda id: vc2[id[1]] )
  vc['union'] = vc['count1'] + vc['count2'] - vc['count']
  vc['iou'] = vc['count'] / vc['union']
  #print(vc)

  # to 2d matrix
  n1 = int(c1.max())+1
  n2 = int(c2.max())+1
    
  if (metric=='iou'):
    RCV = vc['iou'].reset_index().to_numpy()
  elif (metric=='count'):
    RCV = vc['count'].reset_index().to_numpy()
  else:
    raise ValueError('metric should be iou or count')

  metric_matrix = np.zeros( (n1,n2) ) 
  for row in RCV:
    metric_matrix[int(row[0]),int(row[1])] = row[2]

  row_ind, col_ind = linear_sum_assignment(-metric_matrix)

  nm = row_ind.size
  valid = np.zeros( (nm,), dtype=bool )
  for i in range(row_ind.size):
    valid[i] = metric_matrix[row_ind[i],col_ind[i]] >= thresh

  valid_rows = row_ind[valid]
  valid_cols = col_ind[valid]

  nv = valid.sum()
  matched1 = np.sort(np.unique(valid_rows))
  unmatched1 = np.setdiff1d(range(n1), matched1)
  matched2 = np.sort(np.unique(valid_cols))
  unmatched2 = np.setdiff1d(range(n2), matched2)
  nv, n1, n2, matched2.size, unmatched2.size

  if (next_cid is None): next_cid = n1
  if (next_cid < n1):
    warnings.warn('next_cid < n1')
    next_cid = n1

  map2to1 = np.zeros( (n2,), dtype=int )
  map2to1[valid_cols] = valid_rows   # matched labels, reuse old id
  new_labels = range(next_cid, next_cid+unmatched2.size) # Create new ids for unmatched
  map2to1[unmatched2] = new_labels
  next_cid = next_cid+unmatched2.size

  print(f'Cluster match: matched = {matched2.size}, new = {unmatched2.size}')

  return map2to1, next_cid

def track_clusters(rag1=None, rag2=None,
                   edge_filter=None,
                   metric='iou',
                   thresh=0.2, 
                   next_cid=None):
  
  map2to1, next_cid = matchClusters(rag1, rag2, thresh = thresh, metric=metric, next_cid=next_cid)

  for u in rag2.nodes:
    node = rag2.nodes[u]
    c = node['cluster']
    if (c <= 0):
      newc = c  # Background 0 and no cluster -1 treated separately
    else:
      newc = map2to1[c]  
    node['clustertrack'] = newc

  return next_cid

def detectAndTrackClusters(rag2, L2, rag1=None, L1=None, 
                           edge_filter=None,
                           cluster_metric='iou',
                           cluster_thresh=0.2,
                           next_cid=None):
  """Process one frame with: clustering, cluster tracking
  Input is RAG object with: 'trackid', 'velocity_back'
  Output is a RAG object with attributes in: `cluster`, `clustertrack`.
  Previous frame is provided as a RAG object with these properties already defined
  if track_cells is True, perform cell tracking, else assume label images contain `trackid`
  if track_dataframe is not none, use dataframe to assign track_id."""
  
  if (edge_filter is None):
    edge_filter = EdgeFilterAngleRatio(amax=15, count_ratio_min=0.15)
    
  # Assume rag2 has trackid and velocity_back defined

  # 3. Compute velocities (need trackid)
  computeRelativeVelocities(rag2, source='velocity_back')

  # 4. Compute framewise clusters (may need velocities)
  compute_clusters(rag2, min_cluster_size=2, edge_filter=edge_filter);

  # 5. Track clusters
  next_cid = track_clusters(rag1, rag2, thresh=cluster_thresh, metric=cluster_metric, next_cid=next_cid)
  
  return next_cid


def computeVelocities(rag1, rag2, default=np.nan):
  if (rag1 is None):
    for u in rag2.nodes:
      rag2.nodes[u]['velocity_back'] = np.array([default, default])
    return

  df1 = to_pandas_nodelist(rag1)[['trackid','id','centroid']]
  df2 = to_pandas_nodelist(rag2)[['trackid','id','centroid']]
  df = pd.merge(df1, df2, how='outer', on='trackid', indicator=True)
  id1_end = df.loc[df['_merge']=='left_only','id_x']
  id2_start = df.loc[df['_merge']=='right_only','id_y']
  id_both = df.loc[df['_merge']=='both',['id_x','id_y']]

  for _,item in df[df._merge=='both'].iterrows():
    #print(item.id_x, item.id_y)
    c1 = np.array(item.centroid_x)
    c2 = np.array(item.centroid_y)
    #print(c2-c1)
    rag1.nodes[item.id_x]['velocity_front'] = c2-c1
    rag2.nodes[item.id_y]['velocity_back'] = c2-c1
  for _,item in df[df._merge=='left_only'].iterrows():
    rag1.nodes[item.id_x]['velocity_front'] = np.array([default, default])
  for _,item in df[df._merge=='right_only'].iterrows():
    rag2.nodes[item.id_y]['velocity_back'] = np.array([default, default])

def computeAreaChange(rag1, rag2, default=np.nan):
  if (rag1 is None):
    for u in rag2.nodes:
      rag2.nodes[u]['area_change_back'] = default
    return

  df1 = to_pandas_nodelist(rag1)[['trackid','id','area']]
  df2 = to_pandas_nodelist(rag2)[['trackid','id','area']]
  df = pd.merge(df1, df2, how='outer', on='trackid', indicator=True)
  id1_end = df.loc[df['_merge']=='left_only','id_x']
  id2_start = df.loc[df['_merge']=='right_only','id_y']
  id_both = df.loc[df['_merge']=='both',['id_x','id_y']]

  for _,item in df[df._merge=='both'].iterrows():
    #print(item)
    a1 = np.array(item.area_x)
    a2 = np.array(item.area_y)
    rag1.nodes[item.id_x]['area_change_front'] = np.log2(a2/a1)
    rag2.nodes[item.id_y]['area_change_back'] = np.log2(a2/a1)
  for _,item in df[df._merge=='left_only'].iterrows():
    rag1.nodes[item.id_x]['area_change_front'] = default
  for _,item in df[df._merge=='right_only'].iterrows():
    rag2.nodes[item.id_y]['area_change_back'] = default


def computeRelativeVelocities(rag, source='velocity_back'):
  for (u,v) in rag.edges:
    vu = rag.nodes[u][source]
    vv = rag.nodes[v][source]

    if (vu is not None) and (vv is not None) and (not np.isnan(vu).any()) and (not np.isnan(vv).any()):
      rag.get_edge_data(u,v)['relative_velocity'] = vv-vu
    else:
      rag.get_edge_data(u,v)['relative_velocity'] = np.array([0,0])

def loadDPTracks(filename):
  df = pd.read_csv(filename)
  df.sort_values('Frame')
  df = df.rename(columns={'Particle ID': 'trackid','Frame':'frame','X (px)':'cx','Y (px)':'cy'})
  return df