import numpy as np
import gala.dynamics as gd
from gala.units import UnitSystem



class CartesianOrbit:
    """Interpolated orbit containing phase space position, time and absolute magnitudes of xyz and vxyz in cartesian coordinates.
    """
    def __init__(
        self,
        xyz,
        vxyz,
        t,
        units=None
    ):
        self.units = UnitSystem(units) if type(units) == list else units

        self.t =  self.units.decompose(t)
        self.xyz =  self.units.decompose(xyz)
        self.vxyz =  self.units.decompose(vxyz)
        self.r = np.linalg.norm(self.xyz, axis=1)
        self.v = np.linalg.norm(self.vxyz, axis=1)

    def __add__(self, other):
        if not isinstance(other, CartesianOrbit):
            return NotImplemented

        t_all = np.concatenate((self.t, other.t))
        xyz_all = np.concatenate((self.xyz.T, other.xyz.T)).T
        vxyz_all = np.concatenate((self.vxyz.T, other.vxyz.T)).T

        sort_idx = np.argsort(t_all)
        t_sorted = t_all[sort_idx]
        xyz_sorted = xyz_all[:, sort_idx]
        vxyz_sorted = vxyz_all[:, sort_idx]

        unique_t, unique_indices = np.unique(t_sorted, return_index=True)
        unique_xyz = xyz_sorted[:, unique_indices]
        unique_vxyz = vxyz_sorted[:,unique_indices]

        return CartesianOrbit(
            unique_xyz, 
            unique_vxyz, 
            unique_t, 
            units=self.units
        )
    
    def to_gala_orbit(self):
        """Translates to gala.dynamics.orbit instance so that the different methods available can be used.
        """
        return gd.Orbit(
            self.xyz,
            self.vxyz,
            self.t
        )