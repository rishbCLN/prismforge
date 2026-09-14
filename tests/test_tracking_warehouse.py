"""Unit tests for VigiAI Warehouse Tracking and Track History Layer."""
import unittest
from src.perception.yolo import Detection
from src.tracking.history import TrackHistory, TrackObservation
from src.tracking.warehouse_tracker import WarehouseTracker, WarehouseTrack, compute_iou


class TestWarehouseTracking(unittest.TestCase):
    """Validates persistent ID tracking, sliding-window track history, and spatial associations."""

    def test_compute_iou(self):
        """Tests bounding box IoU calculation."""
        # Identical boxes
        self.assertAlmostEqual(compute_iou((10, 10, 50, 50), (10, 10, 50, 50)), 1.0)
        # Disjoint boxes
        self.assertEqual(compute_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        # Half overlap: Box A [0, 0, 20, 20] (area 400), Box B [10, 0, 30, 20] (area 400), intersection [10, 0, 20, 20] (area 200)
        # Union = 400 + 400 - 200 = 600, IoU = 200 / 600 = 0.3333
        self.assertAlmostEqual(compute_iou((0, 0, 20, 20), (10, 0, 30, 20)), 1.0 / 3.0, places=3)

    def test_track_history_sliding_window_and_kinematics(self):
        """Tests that TrackHistory maintains sliding window cap and computes derivatives."""
        hist = TrackHistory(track_id=1, max_window_size=4)
        self.assertEqual(len(hist), 0)

        # Add observations moving downward in y
        for i in range(7):
            obs = hist.add_observation(
                frame_id=i,
                timestamp=i * 0.04,
                center_x=100.0,
                center_y=100.0 + i * 10.0,
                width=30.0,
                height=30.0,
                confidence=0.95,
                bbox=(85.0, 85.0 + i * 10.0, 115.0, 115.0 + i * 10.0),
                center_x_norm=0.5,
                center_y_norm=(100.0 + i * 10.0) / 360.0,
                width_norm=30.0 / 640.0,
                height_norm=30.0 / 360.0
            )

        # Ensure sliding window caps at max_window_size (4)
        self.assertEqual(len(hist), 4)
        latest = hist.get_latest_observation()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame_id, 6)

        # Kinematic velocity check: downward motion vy > 0
        self.assertGreater(latest.velocity_y, 0.0)
        self.assertAlmostEqual(latest.velocity_x, 0.0)
        self.assertGreater(latest.speed, 0.0)

        # Check trajectory retrieval
        traj = hist.get_trajectory(max_points=3)
        self.assertEqual(len(traj), 3)

    def test_id_persistence_across_consecutive_frames(self):
        """Tests that moving carton maintains the exact same persistent track_id."""
        tracker = WarehouseTracker(iou_threshold=0.20, max_lost_frames=10)

        tracked_ids = []
        for i in range(5):
            # Box moving slightly: x shifts by 4px per frame (high IoU overlap)
            x1 = 100.0 + i * 4.0
            y1 = 200.0
            x2 = 140.0 + i * 4.0
            y2 = 240.0
            det = Detection(
                frame_id=i,
                timestamp=i * 0.04,
                track_id=None,
                class_id=1,
                class_name="carton",
                confidence=0.92,
                x1=x1, y1=y1, x2=x2, y2=y2,
                center_x=(x1 + x2) / 2.0,
                center_y=(y1 + y2) / 2.0,
                width=40.0, height=40.0,
                center_x_norm=((x1 + x2) / 2.0) / 640.0,
                center_y_norm=((y1 + y2) / 2.0) / 360.0,
                width_norm=40.0 / 640.0,
                height_norm=40.0 / 360.0
            )

            active_tracks = tracker.update(i, i * 0.04, [det])
            self.assertEqual(len(active_tracks), 1)
            tracked_ids.append(active_tracks[0].track_id)

        # All 5 frames must have the exact same track ID
        self.assertEqual(len(set(tracked_ids)), 1)
        self.assertEqual(tracked_ids[0], 1)
        # Frames seen must be 5
        self.assertEqual(active_tracks[0].frames_seen, 5)

    def test_track_recovery_across_brief_occlusion(self):
        """Tests that an object retains its ID when occluded for 2 frames then reappearing."""
        tracker = WarehouseTracker(iou_threshold=0.20, max_lost_frames=5)

        # Frame 0: carton detected
        det0 = Detection(
            frame_id=0, timestamp=0.0, track_id=None, class_id=1, class_name="carton",
            confidence=0.90, x1=100.0, y1=200.0, x2=140.0, y2=240.0,
            center_x=120.0, center_y=220.0, width=40.0, height=40.0
        )
        tracks0 = tracker.update(0, 0.0, [det0])
        initial_id = tracks0[0].track_id

        # Frames 1 and 2: empty (occluded)
        tracker.update(1, 0.04, [])
        tracker.update(2, 0.08, [])

        # Frame 3: carton reappears in almost the same spot
        det3 = Detection(
            frame_id=3, timestamp=0.12, track_id=None, class_id=1, class_name="carton",
            confidence=0.88, x1=104.0, y1=202.0, x2=144.0, y2=242.0,
            center_x=124.0, center_y=222.0, width=40.0, height=40.0
        )
        tracks3 = tracker.update(3, 0.12, [det3])
        self.assertEqual(len(tracks3), 1)
        # Must retain initial_id, not spawn a new ID
        self.assertEqual(tracks3[0].track_id, initial_id)

    def test_spatial_worker_and_equipment_association(self):
        """Tests nearest worker and overlapping equipment associations on carton."""
        tracker = WarehouseTracker(iou_threshold=0.20)

        person_det = Detection(
            frame_id=0, timestamp=0.0, track_id=None, class_id=0, class_name="person",
            confidence=0.95, x1=300.0, y1=150.0, x2=350.0, y2=270.0,
            center_x=325.0, center_y=210.0, width=50.0, height=120.0,
            center_x_norm=325.0 / 640.0, center_y_norm=210.0 / 360.0
        )

        carton_det = Detection(
            frame_id=0, timestamp=0.0, track_id=None, class_id=1, class_name="carton",
            confidence=0.92, x1=310.0, y1=200.0, x2=350.0, y2=240.0,
            center_x=330.0, center_y=220.0, width=40.0, height=40.0,
            center_x_norm=330.0 / 640.0, center_y_norm=220.0 / 360.0
        )

        active = tracker.update(0, 0.0, [person_det, carton_det])
        self.assertEqual(len(active), 2)

        carton_track = [t for t in active if t.class_name == "carton"][0]
        person_track = [t for t in active if t.class_name == "person"][0]

        # Carton must be associated with the nearby worker
        self.assertEqual(carton_track.associated_worker_id, person_track.track_id)
        self.assertIsNotNone(carton_track.worker_distance_norm)
        self.assertLess(carton_track.worker_distance_norm, 0.10)


if __name__ == "__main__":
    unittest.main()
