# %%
import numpy as np
from bsplyne import new_cylinder, new_disk
from bsplyne.save_YETI import Domain, Interface, write

master   , ctrl_pts_master    = new_cylinder(center_front=[0, 0, 0], orientation=[0, 0, 1], radius=1, length=2)
slave    , ctrl_pts_slave     = new_cylinder(center_front=[0, 0, 2], orientation=[0, 0, 1], radius=1, length=2)
interface, ctrl_pts_interface =     new_disk(      center=[0, 0, 2],      normal=[0, 0, 1], radius=1)

geomdl_master    =    master.getGeomdl(ctrl_pts_master   )
geomdl_slave     =     slave.getGeomdl(ctrl_pts_slave    )
geomdl_interface = interface.getGeomdl(ctrl_pts_interface)

dom_master    =   Domain.DefaultDomain(geometry=geomdl_master   ,    id_dom=1, elem_type="U1")
dom_slave     =   Domain.DefaultDomain(geometry=geomdl_slave    ,    id_dom=2, elem_type="U1")
dom_interface = Interface.InterfDomain(geometry=geomdl_interface, id_interf=1)
# %%
