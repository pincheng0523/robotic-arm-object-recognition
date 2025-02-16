import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion

import random

def generate_random_filename():
    # 定義包含所有文件名的列表
    filenames = [
        "objects/bottle.xml",
        "objects/bottle1.xml",
        # 您可以繼續添加更多文件名
    ]
    
    # 隨機從列表中選擇一個文件名
    random_filename = random.choice(filenames)
    
    return random_filename

# 生成隨機文件名
random_filename = generate_random_filename()

        

class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Bottle1Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle1.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Bottle2Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle2.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Bottle3Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle3.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
class Bottle4Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle4.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
        
class Bottle5Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle5.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
class Bottle6Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle6.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
class Bottle7Object(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle7.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )

class Lemon1Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon1.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )

class Lemon2Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon2.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )

class Lemon3Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon3.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )

class Lemon4Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon4.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )
class Lemon5Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon5.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )
class Lemon6Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon6.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )
class Lemon7Object(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon7.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )
        
class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class Cereal1Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal1.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Cereal2Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal2.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Cereal3Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal3.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Cereal4Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal4.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
class Cereal5Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal5.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
class Cereal6Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal6.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
class Cereal7Object(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal7.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
        
class SquareNut1Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut1.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
class SquareNut2Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut2.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
class SquareNut3Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut3.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
class SquareNut4Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut4.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
class SquareNut5Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut5.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
        
class SquareNut6Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut6.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
        
class SquareNut7Object(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut7.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
class RoundNut1Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut1.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
        
class RoundNut2Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut2.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
        
class RoundNut3Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut3.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
class RoundNut4Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut4.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic
        
class RoundNut5Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut5.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic        
class RoundNut6Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut6.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic                
        
class RoundNut7Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut7.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic     
        
                               

class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/plate-with-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(
            xml_path_completion(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic
