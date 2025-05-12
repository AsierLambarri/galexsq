import gala.potential as gp
from pytreegrav import PotentialTarget, AccelTarget, ConstructTree



class ParticlePotential(gp.PotentialBase):
    m = gp.PotentialParameter("mass", physical_type="mass")
    pos = gp.PotentialParameter("pos", physical_type="length")
    soft = gp.PotentialParameter("soft", physical_type="length")
    ndim = 3
        
    def _compute_tree_ifnone(self):
        if not hasattr(self, "tree"):
            self.tree = ConstructTree(
                pos=self.parameters["pos"].value,
                m=self.parameters["m"].value,
                softening=self.parameters["soft"].value,
                quadrupole=True
            )
        else:
            if self.tree is None:
                self.tree = ConstructTree(
                    pos=self.parameters["pos"].value,
                    m=self.parameters["m"].value,
                    softening=self.parameters["soft"].value,
                    quadrupole=True
                )

    def _set_tree(self, tree):
        self.tree = tree
                
    def _energy(self, q, t=0):
        self._compute_tree_ifnone()
        pot = PotentialTarget(
            pos_target=q, 
            pos_source=self.parameters["pos"].value, 
            m_source=self.parameters["m"].value,
            softening_source=self.parameters["soft"].value,
            G=self.units.get_constant("G"),
            parallel=False,
            quadrupole=True,
            tree=self.tree
        ) 
        return pot

    def _acceleration(self, q, t=0):
        self._compute_tree_ifnone()
        accel = AccelTarget(
            pos_target=q, 
            pos_source=self.parameters["pos"].value, 
            m_source=self.parameters["m"].value,
            softening_source=self.parameters["soft"].value,
            G=self.units.get_constant("G"),
            parallel=False,
            quadrupole=True,
            tree=self.tree
        ) 
        return accel

    def _gradient(self, q, t=0):
        self._compute_tree_ifnone()
        accel = AccelTarget(
            pos_target=q, 
            pos_source=self.parameters["pos"].value, 
            m_source=self.parameters["m"].value,
            softening_source=self.parameters["soft"].value,
            G=self.units.get_constant("G"),
            parallel=False,
            quadrupole=True,
            tree=self.tree
        ) 
        return -1 * accel



