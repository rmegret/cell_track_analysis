
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Output, Button, IntSlider
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import matplotlib

import matplotlib.pyplot as plt

import skimage
from skimage.io import imread
from skimage.measure import regionprops
import numpy as np
import math
from os.path import join
import pandas as pd
import os
from os.path import join
from pathlib import Path
import traceback

### GUI VIEW


from detect_clusters import show_label_df


class FrameViewer:
    def __init__(self, df1, LL, rawimage=None):
        # State
        self.df1 = df1.copy() # Dataframe to use and edit
        self.LL = LL # Original label image stack
        self.LL2 = LL.copy() # Modified label image stack
        self.rawimage = rawimage
        self.current_frame = 0
        self.selected = [] # Cells selected, as list of (label,frame) tuples
        self.id_show = True
        self.id_attr = 'label'
        self.id_fontsize = 7
        self.raw_show = False

        nbframes = self.df1.frame.unique().max()+1 # Assume min frame is 0
        
        # Widgets
        self.frame_slider = IntSlider(
            value=0, min=0, max=nbframes-3, step=1,
            description='Frame ID:', continuous_update=False
        )
        self.id_chb = widgets.Checkbox(
            value=False, description='Show ID', indent=False
        )
        self.id_dropdown = widgets.Dropdown(
            options=['label','trackid','spotid','frame_diff'],
            value='label',
            description='ID attribute:',
        )
        self.raw_chb = widgets.Checkbox(
            value=False, description='Show Raw Img', indent=False
        )
        
        self.btn1 = Button(description="Merge")
        self.btn2 = Button(description="Split")
        self.btn_swap = Button(description="Swap")
        self.btn_refresh = Button(description="Refresh")
        self.btn_clear = Button(description="Clear Selection")

        self.frame_slider.observe(self.on_frame_change, names='value')
        self.btn1.on_click(self.on_merge_click)
        self.btn2.on_click(self.on_split_click)
        self.btn_swap.on_click(self.on_swap_click)
        self.id_chb.observe(self.on_id_show_change, names='value')
        self.raw_chb.observe(self.on_raw_show_change, names='value')
        self.id_dropdown.observe(self.on_id_attr_change, names='value')
        self.btn_refresh.on_click(self.refresh)
        self.btn_clear.on_click(self.clear_selection)

        self.output = Output()
        
        # Figure
        self.ax_size = 6
        nx = 3
        with plt.ioff():
          self.fig, self.axes = plt.subplots(1,nx,figsize=(nx*self.ax_size,self.ax_size), sharex='all', sharey='all')
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('button_press_event', self.on_axis_click)
        
        # Widget container
        self.ui = VBox([
            HBox([self.frame_slider, self.id_chb, self.id_dropdown, self.raw_chb]),
            widgets.Box([self.fig.canvas]),
            HBox([self.btn1, self.btn2, self.btn_swap, self.btn_refresh, self.btn_clear]),
            self.output
        ])
        
        # Connect callbacks


        # Decorate each method to send print to self.output
        self._wrap_methods_with_error_handler()

        self.update_plot()

    def refresh(self, *argv):
       self.output.clear_output()
       self.update_plot()

    def clear_selection(self, *argv):
       self.selected = []
       self.update_plot()

    def on_frame_change(self, change):
        #self.output.clear_output()
        if change['name'] == 'value':
            self.current_frame = change['new']
            with self.output:
              print(f'Current frame = {self.current_frame }')
            self.update_plot()

    def on_id_show_change(self, change):
        #self.output.clear_output()
        if change['name'] == 'value':
            self.id_show = change['new']
            with self.output:
              print(f'Toggled id_show = {self.id_show }')
            self.update_plot()
    def on_id_attr_change(self, change):
        #self.output.clear_output()
        if change['name'] == 'value':
            self.id_attr = change['new']
            with self.output:
              print(f'Changed id_attr = {self.id_attr }')
            self.update_plot()
    def on_raw_show_change(self, change):
        #self.output.clear_output()
        if change['name'] == 'value':
            self.raw_show = change['new']
            with self.output:
              print(f'Changed raw_show = {self.raw_show }')
            self.update_plot()


    def center_plot(self, cx,cy, w=150):
      w = 150
      r = [cx-w/2,cx+w/2,cy+w/2,cy-w/2,]  # Caution: ylim requires decreasing ranges to keep top-left origin
      plt.sca(self.axes[0])
      xylim(r)

    def update_plot(self):
      #with self.output:
      #  print(f'update_plot')

      # Basic approach: just erase and redraw everything
      for ax in self.axes:
        ax.cla()

      df1 = self.df1
      LL2 = self.LL2
      frame1,frame2,frame3 = self.current_frame,self.current_frame+1,self.current_frame+2

      selected_df = df1[df1.apply(self.labelframetuple, axis=1).isin(self.selected)]
      if (len(self.selected)>=1):
        selected_df_ref = df1[df1.apply(self.labelframetuple, axis=1) == (self.selected[0])]  # Ref selection is the first one
      else:
         selected_df_ref = None
      #with self.output:
      #  display(selected_df)

      def update_axis(ax, frame):
        plt.sca(ax)

        if (self.raw_show):
           rawimage = self.rawimage/self.rawimage.max()
        else:
           rawimage = None

        if (self.id_show):
           ax = show_label_df(self.LL2, df1, rawimage, frame=frame, label_map='trackid', node_id=self.id_attr, node_id_fontsize=self.id_fontsize, ax=ax,
                              border_color='blue')  # Label image spots
        else:
          ax = show_label_df(self.LL2, df1, rawimage, frame=frame, label_map='trackid', marker='+', markercolor='red', markersize=9, ax=ax,
                             border_color='blue')  # Label image spots
        #show_label_df(None, df2, frame=frame1, marker='x', markercolor='blue', ax=ax)      # Trackmate spots
        #xylim(r)
        plt.title(f'Frame {frame}')

        sf = selected_df[selected_df.frame==frame] # Selected for frame
        ax.plot( sf.cx, sf.cy, 'or', markersize=20, markerfacecolor='None' )
        if (selected_df_ref is not None):
          sfr = selected_df_ref[selected_df_ref.frame==frame] # Selected for frame
          ax.plot( sfr.cx, sfr.cy, 'or', markersize=20, markerfacecolor='None', markeredgewidth=3 )

      update_axis(self.axes[0], frame1)
      update_axis(self.axes[1], frame2)
      update_axis(self.axes[2], frame3)

      self.fig.tight_layout()

    def on_axis_click(self, event):
        #self.output.clear_output()
        with self.output: # Send print to output, else it does not show
          #print(f'Clicked at {event.xdata},{event.ydata}') 
          #print(event)
          #axes[0].plot(240,200,'rx') # For testing than click is registered
          if ((not event.inaxes) or (event.inaxes not in self.axes)):
            print(f'Clicked, not in axes. IGNORED')
            return
        
          #self.last_event = event  # For debugging the event
          #print(f'Clicked at {event.xdata},{event.ydata}') # Does work in Jupyter by default, would need an Ouput widget
          #axes[0].plot(event.xdata,event.ydata,'rx')
          ax_idx = list(self.axes).index(event.inaxes)
          frame1,frame2,frame3 = self.current_frame,self.current_frame+1,self.current_frame+2
          selected_frame = [frame1, frame2, frame3][ax_idx]
          print(f'In Frame {selected_frame}') # Does work in Jupyter by default, would need an Ouput widget
          
          x,y = round(event.xdata),round(event.ydata) # Get integer position to query the label image
          label = self.LL2[selected_frame][y,x]
          trackid = self.df1[(self.df1.frame==selected_frame)&(self.df1.label==label)] # Could also get the trackid

          sel = (label,selected_frame)  # Use tuple to keep both id and frame and be hashable

          if ( sel not in self.selected):
            self.selected.append( sel )
            print(f'Selected {sel}')
          else:
            self.selected.remove( sel )
            print(f'Unselected {sel}')

          print(f'New selection: {self.selected}')

          self.update_plot()
    
    def on_button_click(self, event):
       with self.output:
          print(f'Button clicked {event}')

    def merge_same_frame( self, label1, frame, label2 ):
      '''
      Delete spot (label2,frame) to merge into (label1, frame).
      For consistency:
      - update the dataframe for cx, cy
      - change all pixels with label2 in frame to label1 in LL2
      '''
      df = self.df1
      LL2 = self.LL2
      frame1 = frame
      frame2 = frame

      ## UPDATE DATAFRAME
      # Merge inside df1
      idx1 = df[ (df.label==label1) & (df.frame==frame1) ].index[0]
      idx2 = df[ (df.label==label2) & (df.frame==frame2) ].index[0]

      # Update idx1 to receive the merged cell
      # Basic average, TODO: average weighted by ratio of area
      df.loc[idx1, ['cx','cy']] = (df.loc[idx1, ['cx','cy']] + df.loc[idx2, ['cx','cy']])/2
      # TODO: update other columns to remain consistent

      # Delete row idx2
      df.drop(idx2, inplace=True)

      ## UPDATE LABELIMAGE
      # Relabel all pixels with value label2 with new value label1
      mask = (LL2[frame2]==label2).copy()
      LL2[ frame2, mask ] = label1

    def merge_different_frame( self, label1, frame1, label2, frame2 ):
      '''
      Change trackid of (label2,frame2) to match (label1, frame1)
      '''
      df = self.df1
      LL2 = self.LL2

      idx1 = df[ (df.label==label1) & (df.frame==frame1) ].index[0]
      idx2 = df[ (df.label==label2) & (df.frame==frame2) ].index[0]

      # Check if trackid already predsent in frame2
      trackid = df.loc[idx1,'trackid']

      conflict_df = df.loc[ (df.frame==frame2) & (df.trackid == trackid) ]
      if ( conflict_df.shape[0]>0 ):
        print(f'ERROR: trackid {trackid} already exist in frame {frame2}. please merge inside the frame')
        display( conflict_df )
        return

      ## UPDATE DATAFRAME

      # idx1 remains the same 

      # Modify trackid for row idx2
      df.loc[idx2, 'trackid'] = trackid

      ## NO NEED TO UPDATE LABELIMAGE
    
    def split_similar_shape( self, label1, frame1, labels2, frame2 ):
      '''
      Split cell (label1, frame1) to match cells (labels2[*], frame2)
      '''
      df = self.df1
      LL2 = self.LL2

      idx1 = df[ (df.label==label1) & (df.frame==frame1) ].index[0]
      idx2 = df[ (df.label==label2) & (df.frame==frame2) ].index[0]

      # Check if trackid already predsent in frame2
      trackid = df.loc[idx,'trackid']
      if (df.loc[ df[df.frame==frame2] ,'trackid'].shape[0]>0):
        print(f'ERROR: trackid {trackid} already exist in frame {frame2}. please merge inside the frame')
        return

      ## UPDATE DATAFRAME

      # idx1 remains the same 

      # Modify trackid for row idx2
      df.loc[idx2, 'trackid'] = trackid

      ## NO NEED TO UPDATE LABELIMAGE

    def swap_trackid( self, label1, frame, label2 ):
      '''
      Swap trackid of (label2,frame2) to match (label1, frame1) and all frames after
      '''
      df = self.df1
      frame1 = frame
      frame2 = frame

      idx1 = df[ (df.label==label1) & (df.frame==frame1) ].index[0]
      idx2 = df[ (df.label==label2) & (df.frame==frame2) ].index[0]

      # Check if trackid already predsent in frame2
      trackid1 = df.loc[idx1,'trackid']
      trackid2 = df.loc[idx2,'trackid']

      with self.output:
         print(trackid1, trackid2)

      if ( np.isnan(trackid1) or np.isnan(trackid2) ):
        print(f'ERROR: trackid cannot be nan, {(trackid1,trackid2)}')
        return

      ## UPDATE DATAFRAME

      idx1_to_swap = df[ (df.trackid==trackid1) & (df.frame>=frame1) ].index
      idx2_to_swap = df[ (df.trackid==trackid2) & (df.frame>=frame1) ].index

      df.loc[idx1_to_swap,'trackid'] = trackid2
      df.loc[idx2_to_swap,'trackid'] = trackid1

      ## NO NEED TO UPDATE LABELIMAGE

    def on_merge_click(self, event):
       with self.output:
          print(f'Button clicked MERGE')

          if (len(self.selected)!=2):
             print(f'ERROR: not implemented, MERGE only for 2 cells')
             return
          
          df = self.df1
          label1,frame1 = self.selected[0]
          label2,frame2 = self.selected[1]

          if (frame1 == frame2):
            # Same frame, need to relabel the image

            self.merge_same_frame( label1, frame1, label2 )

          if (frame1 != frame2):
            # Different frame, we may get away with just changing the trackid

            self.merge_different_frame( label1, frame1, label2, frame2 )

          self.update_plot()
       
    def on_split_click(self, event):
       with self.output:
          print(f'Button clicked SPLIT')

          frames2 = set([frame for frame in self.selected[1:]])
          if (len(self.selected)<3) or (self.selected[0][1] in frames2) or ( len( frames2 )>1 ):
             print(f'ERROR: not implemented, SPLIT need 1 cell to split and 2 or more split cells in another frame')
             return
          
          df = self.df1
          label1,frame1 = self.selected[0]
          _,frame2 = self.selected[1]
          labels2 = [ sel[0] for sel in self.selected[1:] ]

          self.split_similar_shape( label1, frame1, labels2, frame2 )

          self.update_plot()
    
    def on_swap_click(self, event):
       with self.output:
          print(f'Button clicked SWAP (trackid current selection and after)')

          if (len(self.selected)!=2):
             print(f'ERROR: not implemented, SWAP only for 2 cells in same frame')
             return
          
          label1,frame1 = self.selected[0]
          label2,frame2 = self.selected[1]

          if (frame1 != frame2):
            print(f'ERROR: not implemented, SWAP only for 2 cells in same frame')
            return

          self.swap_trackid( label1, frame1, label2 )

          self.update_plot()

    def display(self):
        display(self.ui)

    @staticmethod   # Utility function, does not need self
    def labelframetuple(row):
      frame = row['frame']
      label = row['label']
      if (not np.isnan(frame)): 
        frame=int(frame) 
      else: 
        frame=-1
      if (not np.isnan(label)): 
        label=int(label) 
      else: 
        label=-1
      return (label, frame)

    def catch_errors(self):
        output_widget = self.output
        def decorator(fn):
            def wrapped(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    with output_widget:
                        print(f"[ERROR in {fn.__name__}]: {e}")
                        full_trace = traceback.format_exc()
                        print(full_trace)  # or log it
            return wrapped
        return decorator
    
    def _wrap_methods_with_error_handler(self):
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(self, name)
            if callable(attr):
                setattr(self, name, self.catch_errors()(attr))
