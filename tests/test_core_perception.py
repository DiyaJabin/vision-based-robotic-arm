"""Deterministic core contract/perception tests using only the standard library."""
import math
import unittest
from core.config import CLASS_IDS, CLASS_NAMES, ConfidenceThresholds
from core.contracts import BoundingBox, DetectionResult, ImagePoint, ObjectPose2D, ValidatedDetection
from perception.calibration import compute_homography, pixel_to_world
from perception.hybrid_validator import confidence_band, ContourMetrics, validate_metrics, ValidationConfig
from perception.pose_estimator import normalise_orientation
from perception.yolo_detector import GenericDetection, ModelUnavailableError, YoloDetector


class ContractTests(unittest.TestCase):
    def test_project_classes(self):
        self.assertEqual(dict(CLASS_NAMES),{0:"cube",1:"cylinder",2:"box"})
        self.assertEqual(dict(CLASS_IDS),{"cube":0,"cylinder":1,"box":2})
        self.assertNotIn("bottle",CLASS_IDS)
        self.assertNotIn("cup",CLASS_IDS)

    def test_contracts_keep_unobservable_values(self):
        point=ImagePoint(4.5,6.)
        detection=DetectionResult(0,"cube",.8,BoundingBox(0,0,9,12),point)
        self.assertIsNone(ValidatedDetection(detection,True,None,"high_confidence").validation_score)
        self.assertIsNone(ObjectPose2D(point,10,10,None).orientation_deg)

    def test_invalid_thresholds(self):
        for high,medium in ((.5,.5),(.4,.5),(1.1,.5),(.8,-.1),(math.nan,.5),(.8,math.inf)):
            with self.subTest(high=high,medium=medium), self.assertRaises(ValueError):
                ConfidenceThresholds(high,medium)


class GatingTests(unittest.TestCase):
    def test_boundary_values(self):
        for score,expected in ((0,"low"),(.499999,"low"),(.5,"medium"),(.799999,"medium"),(.8,"high"),(1,"high")):
            self.assertEqual(confidence_band(score),expected)
        self.assertEqual(confidence_band(.85,ConfidenceThresholds(.9,.6)),"medium")

    def test_invalid_confidence(self):
        for value in (-.1,1.1,math.nan,math.inf):
            with self.assertRaises(ValueError): confidence_band(value)

    def test_medium_validation_pass_and_reason(self):
        detection=DetectionResult(0,"cube",.6,BoundingBox(0,0,20,20),ImagePoint(10,10))
        good=ContourMetrics(.8,300,.95,.9,.8,1.)
        result=validate_metrics(detection,good)
        self.assertTrue(result.accepted)
        self.assertIs(result.detection,detection)
        self.assertNotEqual(result.validation_score,detection.confidence)
        bad=ContourMetrics(.1,300,.95,.9,.8,1.)
        result=validate_metrics(detection,bad)
        self.assertFalse(result.accepted)
        self.assertIn("colour_fraction",result.reason)

    def test_shape_and_area_failures(self):
        for name,class_id,metrics,reason in (
            ("cylinder",1,ContourMetrics(.8,100,.9,.8,.2,1),"circularity"),
            ("cube",0,ContourMetrics(.8,100,.9,.9,.9,2),"aspect_ratio"),
            ("box",2,ContourMetrics(.8,10,.9,.9,.9,2),"contour_area"),
            ("box",2,ContourMetrics(.8,100,.4,.9,.9,2),"solidity")):
            result=validate_metrics(DetectionResult(class_id,name,.6,BoundingBox(0,0,20,20),ImagePoint(10,10)),metrics)
            self.assertFalse(result.accepted)
            self.assertIn(reason,result.reason)

    def test_invalid_validation_config(self):
        with self.assertRaises(ValueError): ValidationConfig(min_solidity=2)


class DetectorMappingTests(unittest.TestCase):
    def generic(self,model_id,name):
        return GenericDetection(model_id,name,.9,BoundingBox(2,3,10,20),ImagePoint(7,13))

    def test_native_ids_cannot_be_project_ids(self):
        detector=YoloDetector(model=object())
        self.assertEqual(detector.map_project_detections([self.generic(0,"person"),self.generic(1,"bicycle"),self.generic(2,"car")]),[])

    def test_project_names_map_even_when_model_ids_differ(self):
        detector=YoloDetector(model=object())
        result=detector.map_project_detections([self.generic(9,"box"),self.generic(7,"cube"),self.generic(8,"cylinder")])
        self.assertEqual([d.class_id for d in result],[2,0,1])

    def test_explicit_mapping_and_unknown_names(self):
        detector=YoloDetector(model=object(),class_mapping={"red_block":"cube"})
        result=detector.map_project_detections([self.generic(99,"red_block"),self.generic(2,"cup")])
        self.assertEqual([d.class_name for d in result],["cube"])
        with self.assertRaises(ValueError): YoloDetector(model=object(),class_mapping={"cup":"cup"})

    def test_missing_custom_weights(self):
        with self.assertRaises(ModelUnavailableError):
            YoloDetector(custom_weights="missing-core-test-weights.pt",model=object())


class HomographyTests(unittest.TestCase):
    def test_affine_calibration_and_held_out_points(self):
        pixels=((0,0),(640,0),(640,480),(0,480))
        world=((.2,.3),(.8,.3),(.8,-.3),(.2,-.3))
        matrix=compute_homography(pixels,world)
        for uv,xy in zip(pixels,world):
            self.assertLess(math.dist(pixel_to_world(matrix,*uv),xy),1e-9)
        self.assertLess(math.dist(pixel_to_world(matrix,320,240),(.5,0)),1e-9)
        self.assertLess(math.dist(pixel_to_world(matrix,160,120),(.35,.15)),1e-9)

    def test_projective_calibration(self):
        expected=((.002,.0001,.1),(.0002,-.002,.3),(.0003,.0001,1.))
        pixels=((10,20),(600,20),(600,450),(10,450))
        world=tuple(pixel_to_world(expected,*uv) for uv in pixels)
        matrix=compute_homography(pixels,world)
        self.assertLess(math.dist(pixel_to_world(matrix,215,177),pixel_to_world(expected,215,177)),1e-8)

    def test_invalid_correspondences(self):
        good=((0,0),(1,0),(1,1),(0,1))
        for bad in (((0,0),)*4,((0,0),(1,1),(2,2),(3,3)),((0,0),(1,0),(1,1)),((0,0),(1,0),(1,1),(0,math.nan))):
            with self.assertRaises(ValueError): compute_homography(bad,good)
        with self.assertRaises(ValueError): pixel_to_world(((0,0,0),)*3,0,0)
        with self.assertRaises(ValueError): pixel_to_world(((1,0,0),(0,1,0),(1,0,1)),-1,0)


class SymmetryTests(unittest.TestCase):
    def test_cylinder_yaw_unobservable(self):
        self.assertEqual(normalise_orientation(10,10,123,"cylinder"),(10,10,None))

    def test_square_equivalent_axes(self):
        self.assertEqual(normalise_orientation(20,20,15,"cube"),normalise_orientation(20,20,105,"cube"))

    def test_box_long_axis_and_normalisation(self):
        self.assertEqual(normalise_orientation(10,20,10,"box"),(20,10,100))
        self.assertEqual(normalise_orientation(20,10,190,"box"),(20,10,10))
        with self.assertRaises(ValueError): normalise_orientation(0,10,0,"cube")


if __name__=="__main__": unittest.main()
