import gala.dynamics as gd
from gala.units import UnitSystem



class PhaseSpaceInstant:
    """Class to store Phase Space position and Time Stamp. It uses gala.dynamics.PhaseSpacePosition.
    """
    def __init__(
        self, 
        pos, 
        vel, 
        time, 
        redshift=None, 
        snapshot=None,
        units=None
    ):
        self.units = UnitSystem(units) if type(units) == list else units
        self.pos = self.units.decompose(pos)
        self.vel = self.units.decompose(vel)
        self.time = self.units.decompose(time)
        self.redshift = redshift
        self.snapshot = snapshot
        assert (time is not None) or (redshift is not None), "You must provide either time or redshift!"
        
        self.PhaseSpacePos =  gd.PhaseSpacePosition(
            pos=self.pos,
            vel=self.vel
        )

    def change_units(self, new_system):
        """Changes the units
        """
        new_units = UnitSystem(new_system) if type(new_system) == list else new_system
        return PhaseSpaceInstant(
            self.pos, 
            self.vel, 
            self.time, 
            redshift=self.redshift, 
            snapshot=self.snapshot, 
            units=new_units
        )

    def to_gala(self):
        """Changes to gala PhaseSpacePosition class
        """
        return gd.PhaseSpacePosition(
            pos=self.pos,
            vel=self.vel
        )
