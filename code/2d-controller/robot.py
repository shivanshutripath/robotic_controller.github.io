# robot.py
"""
Minimal robotics utilities for the map navigation project.

Provides:
- Map2D: occupancy grid builder (from a pygame.Surface)
- Robot: differential-drive kinematics
- Ultrasonic: ray-cast sensing against an occupancy map (needs is_occupied(x,y)->bool)
- Graphics: pygame rendering helpers (map, robot sprite, sensor rays, path, text)

Notes:
- Controller code may choose NOT to use Map2D and instead provide its own OccMap2D wrapper.
- Ultrasonic only requires that the provided map object implements is_occupied(x,y)->bool.
"""

import pygame
import math
import numpy as np


# -------------------------
# Map abstraction (robust)
# -------------------------
class Map2D:
    """
    Builds an occupancy grid from a map surface using:
      - grayscale conversion
      - Otsu thresholding
      - morphology cleanup (optional)
      - obstacle inflation (optional)

    occ[y, x] = True means obstacle.
    """
    def __init__(
        self,
        map_surface: pygame.Surface,
        obstacle_is_dark: bool = True,
        morph_passes: int = 2,
        inflate_px: int = 10,
    ):
        self.surface = map_surface
        self.width = map_surface.get_width()
        self.height = map_surface.get_height()

        gray = self._surface_to_gray(map_surface)         # (H,W) float
        t = self._otsu_threshold(gray)                    # scalar threshold

        # threshold to obstacle mask
        occ = (gray < t) if obstacle_is_dark else (gray > t)

        # cleanup (closing-ish)
        if morph_passes > 0:
            occ = self._cleanup(occ, passes=morph_passes)

        # inflate obstacles for robot footprint
        if inflate_px > 0:
            occ = self._inflate_fast(occ, radius=inflate_px)

        self.occ = occ.astype(bool)

    def is_occupied(self, x: int, y: int) -> bool:
        """Out-of-bounds treated as occupied."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return bool(self.occ[int(y), int(x)])

    @staticmethod
    def _surface_to_gray(surface: pygame.Surface) -> np.ndarray:
        # array3d -> (W,H,3), transpose -> (H,W,3)
        arr = pygame.surfarray.array3d(surface).astype(np.float32)
        arr = np.transpose(arr, (1, 0, 2))
        # luminance
        gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        return gray

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> float:
        g = np.clip(gray, 0, 255).astype(np.uint8)
        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        total = g.size
        if total == 0:
            return 128.0

        p = hist / total
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]

        denom = omega * (1.0 - omega)
        denom[denom == 0] = 1e-12
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom

        return float(np.argmax(sigma_b2))

    @staticmethod
    def _dilate8(mask: np.ndarray) -> np.ndarray:
        m = mask
        d = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                d |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        return d

    @staticmethod
    def _erode8(mask: np.ndarray) -> np.ndarray:
        m = mask
        e = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                e &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        return e

    def _cleanup(self, mask: np.ndarray, passes: int = 1) -> np.ndarray:
        """
        Closing-ish: dilate then erode to fill small gaps / remove speckles.
        """
        m = mask.copy()
        for _ in range(passes):
            m = self._dilate8(m)
        for _ in range(passes):
            m = self._erode8(m)
        return m

    def _inflate_fast(self, occ: np.ndarray, radius: int) -> np.ndarray:
        """
        Inflation using repeated 8-neighborhood dilation.
        Fast and good enough for navigation.
        """
        m = occ.copy()
        for _ in range(int(radius)):
            m = self._dilate8(m)
        return m


# -------------------------
# Differential drive robot
# -------------------------
class Robot:
    """
    Differential-drive robot model in pixel coordinates.

    State:
      x, y, heading (radians)

    Inputs:
      wheel linear speeds vl, vr (px/s)

    Kinematics:
      v = (vl+vr)/2
      omega = (vr-vl)/w
    """
    def __init__(self, startpos, axle_length_px):
        self.m2p = 3779.52  # kept for reference from your original project
        self.w = float(axle_length_px)  # wheelbase in pixels

        self.x = float(startpos[0])
        self.y = float(startpos[1])
        self.heading = 0.0

        # wheel velocities (px/s)
        self.maxspeed = 0.02 * self.m2p
        self.vl = 0.01 * self.m2p
        self.vr = 0.01 * self.m2p

    @property
    def pose(self):
        return (self.x, self.y, self.heading)

    def set_wheels(self, vl, vr):
        self.vl = float(max(-self.maxspeed, min(self.maxspeed, vl)))
        self.vr = float(max(-self.maxspeed, min(self.maxspeed, vr)))

    def stop(self):
        self.vl = 0.0
        self.vr = 0.0

    def kinematics(self, dt):
        v = (self.vl + self.vr) * 0.5
        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        self.heading = (self.heading + ((self.vr - self.vl) / self.w) * dt) % (2 * math.pi)


# -------------------------
# Graphics
# -------------------------
class Graphics:
    """
    Handles drawing:
      - background map image
      - robot sprite (rotated)
      - sensor hit points
      - paths / points
      - text overlays
    """
    def __init__(self, dimensions, robot_img_path, map_img_path):
        pygame.init()

        self.height, self.width = dimensions
        pygame.display.set_caption("Maze Navigation (A* + Ultrasonic Override)")
        self.map = pygame.display.set_mode((self.width, self.height))

        # convert() requires display already created, so this order is correct
        self.robot = pygame.image.load(robot_img_path).convert_alpha()
        self.map_img = pygame.image.load(map_img_path).convert()

        self.red = (255, 0, 0)

        # draw the initial map
        self.map.blit(self.map_img, (0, 0))

    def clear(self):
        self.map.blit(self.map_img, (0, 0))

    def draw_robot(self, x, y, heading):
        scale = 0.5
        rotated = pygame.transform.rotozoom(self.robot, -math.degrees(heading), scale)
        rect = rotated.get_rect(center=(int(x), int(y)))
        self.map.blit(rotated, rect)

    def draw_sensor_data(self, point_cloud):
        for px, py in point_cloud:
            pygame.draw.circle(self.map, self.red, (int(px), int(py)), 2)

    def draw_points(self, pts, color=(0, 120, 255), r=2, step=1):
        for (x, y) in pts[::step]:
            pygame.draw.circle(self.map, color, (int(x), int(y)), r)

    def draw_text(self, text, x=8, y=8, color=(30, 30, 30)):
        font = pygame.font.SysFont(None, 22)
        surf = font.render(text, True, color)
        self.map.blit(surf, (x, y))


# -------------------------
# Ultrasonic sensing (raycast vs occupancy)
# -------------------------
class Ultrasonic:
    """
    Ray-cast sensor against an occupancy map.

    The provided map2d object must implement:
      is_occupied(x:int, y:int) -> bool
    """
    def __init__(self, sensor_range, map2d, surface: pygame.Surface, n_rays=16):
        self.range_px, self.half_fov = sensor_range
        self.map2d = map2d
        self.surface = surface
        self.n_rays = int(n_rays)

    def sense(self, x, y, heading):
        cloud = []
        start_angle = heading - self.half_fov
        finish_angle = heading + self.half_fov

        # cast rays
        for ang in np.linspace(start_angle, finish_angle, self.n_rays, endpoint=False):
            x2 = x + self.range_px * math.cos(ang)
            y2 = y + self.range_px * math.sin(ang)

            hit = None
            # sample along the segment
            for i in range(1, 140):
                u = i / 140.0
                sx = int(x2 * u + x * (1 - u))
                sy = int(y2 * u + y * (1 - u))

                # treat outside as obstacle via is_occupied behavior
                if self.map2d.is_occupied(sx, sy):
                    hit = (sx, sy)
                    break

            if hit is not None:
                cloud.append([hit[0], hit[1]])

        ranges = self._sector_ranges(cloud, x, y, heading)
        return {"cloud": cloud, "ranges": ranges}

    @staticmethod
    def _sector_ranges(cloud, rx, ry, heading):
        dL = float("inf")
        dF = float("inf")
        dR = float("inf")

        for px, py in cloud:
            dx = px - rx
            dy = py - ry
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx) - heading
            ang = (ang + math.pi) % (2 * math.pi) - math.pi

            # front: +/- 30 degrees
            if -math.pi / 6 <= ang <= math.pi / 6:
                dF = min(dF, dist)
            # left: > +30 degrees
            elif ang > math.pi / 6:
                dL = min(dL, dist)
            # right: < -30 degrees
            else:
                dR = min(dR, dist)

        return {"left": dL, "front": dF, "right": dR}
