import numpy as np

from ._helpers import best_dtype

import gc

class ParticleDataOld:
    """Simple class to handle particle data, with an option to provide a mask
    """
    def __init__(self,
                 sp,
                 particle_names
                ):
        """Init function.
        """
        self.datasrc = sp 
        self.ds = sp.ds
        self.nbody = particle_names["nbody"]
        self.dm = particle_names["darkmatter"]
        self.stars = particle_names["stars"]

        self._data = {
            (self.nbody, "particle_position") : sp[self.nbody, "particle_position"].to("kpc"),
            (self.nbody, "particle_mass") :  sp[self.nbody, "particle_mass"].to("Msun"),
            (self.nbody, "particle_velocity") :  sp[self.nbody, "particle_velocity"].to("km/s"),
            (self.nbody, "particle_index") :  sp[self.nbody, "particle_index"],

            (self.stars, "particle_position") : sp[self.stars, "particle_position"].to("kpc"),
            (self.stars, "particle_mass") :  sp[self.stars, "particle_mass"].to("Msun"),
            (self.stars, "particle_velocity") :  sp[self.stars, "particle_velocity"].to("km/s"),
            (self.stars, "particle_index") :  sp[self.stars, "particle_index"],

            (self.dm, "particle_position") : sp[self.dm, "particle_position"].to("kpc"),
            (self.dm, "particle_mass") :  sp[self.dm, "particle_mass"].to("Msun"),
            (self.dm, "particle_velocity") :  sp[self.dm, "particle_velocity"].to("km/s"),
            (self.dm, "particle_index") :  sp[self.dm, "particle_index"],
        }

        self._set_minimum_bytes()
        
        self._masks = {}
        sp.clear_data()

        
    def __getitem__(self, key):
        return self._data[key]

    def _get_keys(self, selected_ptype):
        """Get keys for a given particle
        """
        return  [field for (ptype, field) in self._data.keys() if ptype == selected_ptype]

    def add_field(self, field_name, field_value, field_indices):
        """ Adds fields to data.
        """
        indexes = self._data[field_name[0], "particle_index"]

        sorter = np.argsort(field_indices)
        field_value_sorted = field_value[sorter]
        field_indices_sorted = field_indices[sorter]
        
        positions = np.searchsorted(field_indices_sorted, indexes)
        ordered_field_value = field_value_sorted[positions]

        self._data[field_name] = ordered_field_value

        
        
        
    def add_bound_mask(self, bound_indices, suffix="_bound"):
        """Adds fields for boundness
        """
        particles = [self.nbody, self.dm, self.stars]
        for ptype in particles:
            all_fields = self._get_keys(ptype)
            mask = np.isin(self._data[ptype, "particle_index"], bound_indices)
            self._masks[ptype] = mask
            for field in all_fields:
                self._data[ptype + suffix, field] = self._data[ptype, field][mask]

    
    def rename_particle(self, ptype, new_name):
        """Renames particle
        """
        all_fields = self._get_keys(ptype)
        for field in all_fields:
            self._data[new_name, field] = self._data.pop(ptype, field)

        gc.collect()


    def remove_particle(self, ptype):
        """Removes particle from data
        """
        all_fields = self._get_keys(ptype)
        for field in all_fields:
            self._data.pop(ptype, field)

        gc.collect()
        
        
        
        
        


class ParticleData:
    """Simple class to handle particle data with optional boolean masking for suffix fields."""
    def __init__(self, sp, particle_names):
        """Init function."""
        self.datasrc = sp
        self.ds = sp.ds
        self.nbody = particle_names["nbody"]
        self.dm = particle_names["darkmatter"]
        self.stars = particle_names["stars"]

        # Store raw data arrays keyed by (ptype, field)
        self._data = {
            (self.nbody, "particle_position"): sp[self.nbody, "particle_position"].to("kpc"),
            (self.nbody, "particle_mass"):     sp[self.nbody, "particle_mass"].to("Msun"),
            (self.nbody, "particle_velocity"): sp[self.nbody, "particle_velocity"].to("km/s"),
            (self.nbody, "particle_index"):    sp[self.nbody, "particle_index"],

            (self.stars, "particle_position"): sp[self.stars, "particle_position"].to("kpc"),
            (self.stars, "particle_mass"):     sp[self.stars, "particle_mass"].to("Msun"),
            (self.stars, "particle_velocity"): sp[self.stars, "particle_velocity"].to("km/s"),
            (self.stars, "particle_index"):    sp[self.stars, "particle_index"],

            (self.dm, "particle_position"):    sp[self.dm, "particle_position"].to("kpc"),
            (self.dm, "particle_mass"):        sp[self.dm, "particle_mass"].to("Msun"),
            (self.dm, "particle_velocity"):    sp[self.dm, "particle_velocity"].to("km/s"),
            (self.dm, "particle_index"):       sp[self.dm, "particle_index"],
        }
        
        self._set_minimum_bytes()
        # Masks keyed by (ptype, suffix)
        self._masks = {}
        # Clear original data source to free memory
        sp.clear_data()

    def _set_minimum_bytes(self):
        """Sets appropriate array types for each entry. Helps saving memory.
        """
        # for field, field_value in self._data.items():
        #     self._data[field] = field_value.astype(best_dtype(field_value.value, eps=1E-4))

        # gc.collect()
        return None

    
    def __getitem__(self, key):
        """
        Retrieve data arrays, applying mask if the particle type includes a suffix.

        key: tuple (ptype, field)
        """
        ptype, field = key
        # Direct access for raw data
        if (ptype, field) in self._data:
            return self._data[(ptype, field)]
        # Handle masked (suffix) access
        for (base_ptype, suffix), mask in self._masks.items():
            if ptype == base_ptype + suffix:
                return self._data[(base_ptype, field)][mask]
        # Not found
        raise KeyError(f"No such data field: {ptype}, {field}")

    def _get_keys(self, selected_ptype):
        """Get fields registered for a given particle type."""
        return [field for (ptype, field) in self._data.keys() if ptype == selected_ptype]

    def add_field(self, field_name, field_value, field_indices):
        """Adds a custom field to the stored data arrays."""
        indexes = self._data[(field_name[0], "particle_index")]
        sorter = np.argsort(field_indices)
        field_value_sorted = field_value[sorter]
        field_indices_sorted = field_indices[sorter]
        positions = np.searchsorted(field_indices_sorted, indexes)
        ordered_field_value = field_value_sorted[positions]
        self._data[field_name] = ordered_field_value

    def add_bound_mask(self, bound_indices, suffix="_bound"):
        """Registers a boolean mask for particles whose indices are in bound_indices."""
        particles = [self.nbody, self.dm, self.stars]
        for ptype in particles:
            # Create mask for this particle type
            mask = np.isin(self._data[(ptype, "particle_index")], bound_indices)
            # Store under (ptype, suffix)
            self._masks[(ptype, suffix)] = mask

    def rename_particle(self, ptype, new_name):
        """Renames a particle type, updating data and any masks."""
        # Update data keys
        for field in self._get_keys(ptype):
            self._data[(new_name, field)] = self._data.pop((ptype, field))
        # Update masks keys
        for (base, suffix), mask in list(self._masks.items()):
            if base == ptype:
                self._masks[(new_name, suffix)] = self._masks.pop((base, suffix))
        gc.collect()

    def remove_particle(self, ptype):
        """Removes all data and masks for a particle type."""
        # Remove data fields
        for field in self._get_keys(ptype):
            self._data.pop((ptype, field))
        # Remove masks
        for key in list(self._masks.keys()):
            if key[0] == ptype:
                self._masks.pop(key)
        gc.collect()
        
        
        
        
        
        