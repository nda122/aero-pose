import numpy as np

from aero_pose.lifting.spatial_vectors import (
    H36M_JOINT_NAMES, H36M_PARENTS, bone_vector,
)


class Skeleton3D:
    def __init__(self, joints: np.ndarray | None = None) -> None:
        self.joints: np.ndarray = joints.copy() if joints is not None else np.zeros((17, 3), dtype=np.float32)

    @property
    def joint_names(self) -> list[str]:
        return H36M_JOINT_NAMES

    @property
    def parents(self) -> list[int]:
        return H36M_PARENTS

    def get_bone_vector(self, child: int, parent: int) -> np.ndarray:
        return bone_vector(self.joints, parent, child)

    def get_bone_vectors(self) -> dict[str, np.ndarray]:
        vectors = {}
        for i, parent in enumerate(H36M_PARENTS):
            if parent >= 0:
                name = f"{H36M_JOINT_NAMES[parent]}_to_{H36M_JOINT_NAMES[i]}"
                vectors[name] = self.get_bone_vector(i, parent)
        return vectors

    def get_joint(self, name: str) -> np.ndarray:
        return self.joints[H36M_JOINT_NAMES.index(name)]

    def center_at_hip(self) -> None:
        hip = self.joints[0].copy()
        self.joints -= hip

    def copy(self) -> "Skeleton3D":
        return Skeleton3D(self.joints)
