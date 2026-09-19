"""Class-specific placement on measured destination pads, not physical bins."""
from dataclasses import dataclass
import math
from core.config import BIN_SCENE_KEYS, CLASS_DESTINATIONS


class DestinationError(ValueError):
    pass


@dataclass(frozen=True)
class DestinationSurface:
    bin_name: str
    body_id: int
    lower: tuple[float,float,float]
    upper: tuple[float,float,float]


@dataclass(frozen=True)
class Placement:
    bin_name: str
    body_id: int
    object_position: tuple[float,float,float]
    object_orientation: tuple[float,float,float,float] = (0.,0.,0.,1.)


class DestinationPlanner:
    """One upright, axis-aligned object per pad; reject occupied/full pads.

    The initial release gap and edge margin are configurable simulation values.
    Stacking and arbitrary bin packing are deliberately not assumed.
    """
    def __init__(self,surfaces,*,release_gap=0.003,edge_margin=0.003):
        self.surfaces={surface.bin_name:surface for surface in surfaces}
        if set(self.surfaces)!=set(BIN_SCENE_KEYS):
            raise DestinationError("All three bin surfaces are required.")
        if any(not math.isfinite(v) or v<0 for v in (release_gap,edge_margin)):
            raise DestinationError("Placement margins must be finite and nonnegative.")
        for surface in self.surfaces.values():
            if not all(map(math.isfinite,surface.lower+surface.upper)) or any(lo>=hi for lo,hi in zip(surface.lower,surface.upper)):
                raise DestinationError("Invalid destination AABB.")
        self.release_gap,self.edge_margin=release_gap,edge_margin
        self.occupied=set()

    @classmethod
    def from_scene(cls,destinations,client_id=0,**kwargs):
        import pybullet as p
        surfaces=[]
        for name,key in BIN_SCENE_KEYS.items():
            if key not in destinations:
                raise DestinationError(f"Missing destination body: {key}")
            body=destinations[key]
            lower,upper=p.getAABB(body,physicsClientId=client_id)
            surfaces.append(DestinationSurface(name,body,tuple(lower),tuple(upper)))
        return cls(surfaces,**kwargs)

    def placement_for(self,class_name,dimensions):
        if class_name not in CLASS_DESTINATIONS:
            raise DestinationError("unsupported_class")
        if len(dimensions)!=3 or any(not math.isfinite(v) or v<=0 for v in dimensions):
            raise DestinationError("invalid_object_dimensions")
        name=CLASS_DESTINATIONS[class_name]
        if name in self.occupied:
            raise DestinationError("destination_occupied")
        surface=self.surfaces[name]
        if any(size+2*self.edge_margin>surface.upper[d]-surface.lower[d] for d,size in enumerate(dimensions[:2])):
            raise DestinationError("object_does_not_fit_destination")
        centre=((surface.lower[0]+surface.upper[0])/2,(surface.lower[1]+surface.upper[1])/2,
                surface.upper[2]+dimensions[2]/2+self.release_gap)
        return Placement(name,surface.body_id,centre)

    def mark_occupied(self,bin_name):
        if bin_name not in self.surfaces:
            raise DestinationError("unknown_bin")
        self.occupied.add(bin_name)
