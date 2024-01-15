from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pointrcnn_score_head import PointRCNNScoreHead
from .pvrcnn_head import PVRCNNHead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'PointRCNNScoreHead': PointRCNNScoreHead,
    'VoxelRCNNHead': VoxelRCNNHead
}
