"""Fake ROS 2 environment so the unmodified real GUI's `hud_node.py` can run on
a developer laptop with no ROS install / no Docker / no robot.

Approach: register a handful of synthetic modules into `sys.modules` BEFORE
`hud_node` is imported, so `import rclpy`, `from sensor_msgs.msg import Image`,
etc. all succeed. The real GUI's `_HAS_ROS` flag then evaluates True and the
`HudNode` class gets defined — but every `create_subscription` / `create_publisher`
call is a no-op that just records the callback so the runner can feed
synthetic messages directly into the node's `_cb_*` callbacks later.

Public API:
    install()  — registers all fake modules. Call BEFORE `import hud_node`.

Once installed, the runner builds messages as `types.SimpleNamespace` trees and
invokes the recorded callbacks. Anything the GUI does internally (state
updates, worker submissions, panel draws) flows through real GUI code paths.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace


_INSTALLED = False


# ── Fake rclpy.node.Node ──────────────────────────────────────────────────

class _FakeLogger:
    """Drop-in for rclpy's get_logger() result."""
    def __init__(self, name):
        self._name = name

    def _emit(self, level, msg):
        # Keep stdout quiet by default — the GUI is busy. Flip _VERBOSE to
        # see logger output for debugging.
        if _VERBOSE:
            print(f'[{level}] {self._name}: {msg}')

    def info(self, msg):  self._emit('INFO',  msg)
    def warn(self, msg):  self._emit('WARN',  msg)
    def warning(self, msg): self._emit('WARN', msg)
    def error(self, msg): self._emit('ERROR', msg)
    def debug(self, msg): pass  # silent

_VERBOSE = False


class _FakeSubscription:
    """Returned by create_subscription. Holds the callback + topic so the
    runner can look it up and deliver fake messages."""
    def __init__(self, topic, callback):
        self.topic = topic
        self.callback = callback


class _FakePublisher:
    """Returned by create_publisher. publish() loops back to any local
    subscription on the same topic so request/observe patterns (e.g.
    /data/toggle_collect driven by the GUI itself) work without an
    outside publisher."""
    def __init__(self, topic, node):
        self.topic = topic
        self._node = node
    def publish(self, msg):
        sub = self._node._fake_subs.get(self.topic)
        if sub is not None:
            sub.callback(msg)


class _FakeNode:
    """Stand-in for rclpy.node.Node. Real HudNode inherits from this and
    calls create_subscription / create_publisher freely.

    Subscriptions / publishers are exposed at:
        self._fake_subs[topic] -> _FakeSubscription
        self._fake_pubs[topic] -> _FakePublisher
    so the runner can `node._fake_subs['/odom'].callback(msg)` to deliver
    a fake odometry sample.
    """
    def __init__(self, node_name='fake', *args, **kwargs):
        self._fake_node_name = node_name
        self._fake_logger = _FakeLogger(node_name)
        self._fake_subs = {}
        self._fake_pubs = {}

    def get_logger(self):
        return self._fake_logger

    def get_clock(self):
        return _FakeClock()

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = _FakeSubscription(topic, callback)
        # Allow multiple subscriptions per topic — keep the latest only.
        self._fake_subs[topic] = sub
        return sub

    def create_publisher(self, msg_type, topic, qos):
        pub = _FakePublisher(topic, self)
        self._fake_pubs[topic] = pub
        return pub

    def create_timer(self, period_sec, callback):
        # No-op; the runner drives data on its own schedule.
        return SimpleNamespace(cancel=lambda: None)

    def destroy_node(self):
        pass

    def declare_parameter(self, name, default=None, *args, **kwargs):
        return SimpleNamespace(value=default,
                               get_parameter_value=lambda: SimpleNamespace(
                                   string_value=default if isinstance(default, str) else '',
                                   integer_value=default if isinstance(default, int) else 0,
                                   double_value=default if isinstance(default, float) else 0.0,
                                   bool_value=default if isinstance(default, bool) else False,
                               ))

    def get_parameter(self, name):
        return SimpleNamespace(value=None)


class _FakeClock:
    def now(self):
        ns = int(time.time() * 1e9)
        return SimpleNamespace(
            nanoseconds=ns,
            to_msg=lambda: SimpleNamespace(sec=ns // 10**9,
                                            nanosec=ns % 10**9),
        )


# ── Fake QoS ──────────────────────────────────────────────────────────────

class _Enum:
    """Minimal enum-like surface: ANY_ATTR is just an int placeholder."""
    def __getattr__(self, name):
        return name


ReliabilityPolicy = _Enum()
HistoryPolicy = _Enum()
DurabilityPolicy = _Enum()


class QoSProfile:
    def __init__(self, depth=10, reliability=None, durability=None,
                 history=None, **kwargs):
        self.depth = depth
        self.reliability = reliability
        self.durability = durability
        self.history = history
        for k, v in kwargs.items():
            setattr(self, k, v)


qos_profile_sensor_data = QoSProfile(depth=5)
qos_profile_system_default = QoSProfile(depth=10)


# ── Fake executors ────────────────────────────────────────────────────────

class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        self._nodes = []
    def add_node(self, node):
        self._nodes.append(node)
    def spin(self):
        # Block forever — but the runner runs the data driver on a Qt
        # timer instead, so this is never actually called.
        while True:
            time.sleep(60)
    def shutdown(self):
        pass


# ── Fake top-level rclpy functions ────────────────────────────────────────

def _rclpy_init(args=None):
    pass

def _rclpy_shutdown():
    pass

def _rclpy_spin_once(node, timeout_sec=0.0):
    pass


# ── Module registration ───────────────────────────────────────────────────

def _build_module(name, **attrs):
    """Create a synthetic module, populate attrs, register in sys.modules."""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _msg_class(name):
    """Return a permissive message class that stores whatever is set on it."""
    cls = type(name, (SimpleNamespace,), {})
    cls.__module__ = 'sim_msgs'
    return cls


def install(verbose=False):
    """Register every fake module needed by the real hud_node.py.

    Safe to call once; subsequent calls are no-ops. After install(),
    `import rclpy`, `from rclpy.node import Node`, the message imports,
    `from cv_bridge import CvBridge`, and `from sensor_msgs_py import
    point_cloud2` all succeed.
    """
    global _INSTALLED, _VERBOSE
    if _INSTALLED:
        return
    _VERBOSE = verbose

    # rclpy core
    rclpy_mod = _build_module(
        'rclpy',
        init=_rclpy_init,
        shutdown=_rclpy_shutdown,
        spin_once=_rclpy_spin_once,
    )

    rclpy_node_mod = _build_module('rclpy.node', Node=_FakeNode)
    rclpy_mod.node = rclpy_node_mod

    rclpy_qos_mod = _build_module(
        'rclpy.qos',
        QoSProfile=QoSProfile,
        ReliabilityPolicy=ReliabilityPolicy,
        HistoryPolicy=HistoryPolicy,
        DurabilityPolicy=DurabilityPolicy,
        qos_profile_sensor_data=qos_profile_sensor_data,
        qos_profile_system_default=qos_profile_system_default,
    )
    rclpy_mod.qos = rclpy_qos_mod

    rclpy_executors_mod = _build_module(
        'rclpy.executors', SingleThreadedExecutor=_FakeExecutor,
    )
    rclpy_mod.executors = rclpy_executors_mod

    # sensor_msgs.msg
    sensor_msgs_mod = _build_module('sensor_msgs')
    sensor_msgs_msg_mod = _build_module(
        'sensor_msgs.msg',
        Image=_msg_class('Image'),
        LaserScan=_msg_class('LaserScan'),
        NavSatFix=_msg_class('NavSatFix'),
        Imu=_msg_class('Imu'),
        PointCloud2=_msg_class('PointCloud2'),
        Joy=_msg_class('Joy'),
    )
    sensor_msgs_mod.msg = sensor_msgs_msg_mod

    # nav_msgs.msg — note: hud_node imports `Path` from nav_msgs but also
    # has `from pathlib import Path` at the top, so the nav_msgs Path
    # shadows pathlib.Path further down — same as the real GUI.
    nav_msgs_mod = _build_module('nav_msgs')
    nav_msgs_msg_mod = _build_module(
        'nav_msgs.msg',
        Odometry=_msg_class('Odometry'),
        OccupancyGrid=_msg_class('OccupancyGrid'),
        Path=_msg_class('Path'),
    )
    nav_msgs_mod.msg = nav_msgs_msg_mod

    # geometry_msgs.msg
    geom_mod = _build_module('geometry_msgs')
    geom_msg_mod = _build_module(
        'geometry_msgs.msg',
        PoseWithCovarianceStamped=_msg_class('PoseWithCovarianceStamped'),
    )
    geom_mod.msg = geom_msg_mod

    # std_msgs.msg
    std_msgs_mod = _build_module('std_msgs')
    std_msgs_msg_mod = _build_module(
        'std_msgs.msg',
        Float32=_msg_class('Float32'),
        Int32=_msg_class('Int32'),
        Int32MultiArray=_msg_class('Int32MultiArray'),
        Bool=_msg_class('Bool'),
    )
    std_msgs_mod.msg = std_msgs_msg_mod

    # sensor_msgs_py.point_cloud2 — used optionally for PCL2 decoding.
    # Provide a no-op read_points so the GUI can call it without
    # crashing (returns an empty iterator).
    spy_mod = _build_module('sensor_msgs_py')
    spy_pc2_mod = _build_module(
        'sensor_msgs_py.point_cloud2',
        read_points=lambda *args, **kwargs: iter(()),
    )
    spy_mod.point_cloud2 = spy_pc2_mod

    # cv_bridge — optional; provide a minimal CvBridge stub.
    cv_bridge_mod = _build_module(
        'cv_bridge',
        CvBridge=type('CvBridge', (), {
            '__init__': lambda self: None,
            'imgmsg_to_cv2':
                lambda self, msg, desired_encoding='passthrough': None,
        }),
    )

    _INSTALLED = True
