import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from custom_linalg import rotation_matrix_by_angle


class BaseField:
    max_shift = .01

    def direction(self, p):
        ''' returns unit vector '''
        v = self._vector(p)
        return v / (la.norm(v) + .000001)

    def vector(self, p):
        return self._vector(p)

    def _vector(self, p):
        return 1, 0
    
    def point_close_to_source(self, p):
        return False
        
    def compute_traces(self, starting_points, trace_segments=100, dv=.02):
        ''' main logic here '''
        num_traces = len(starting_points)
        traces = np.zeros((num_traces, trace_segments, 2))
        traces[:,0,:] = starting_points
        for t in range(trace_segments - 1):
            for i in range(num_traces):
                if self.point_close_to_source(traces[i, t]):
                    traces[i, t + 1] = traces[i, t]
                    continue                        
                delta = dv * self.vector(traces[i, t])
                if la.norm(delta) > self.max_shift:
                    delta = self.direction(traces[i, t]) * self.max_shift
                traces[i, t + 1] = traces[i, t] + delta
        return traces


class ExpandingField(BaseField):
    def _vector(self, p):
        return p


class CosineField(BaseField):
    def _vector(self, p):
        x, y = p
        v = np.array((np.cos(3 * (x + 2 * y)), np.sin(3 * (x - 2 * y))))
        return 100 * v


class TwoCuspsField(BaseField):
    def _vector(self, p):
        x, y = p
        v = np.array((x**2 + 2 * x * y, y**2 + 2 * x * y))
        return 100 * v


class DipoleField(BaseField):
    ''' someone on Internet said this is expression for dipole field '''
    def _vector(self, p):
        x, y = 10 * p
        v = np.array(((x + 1) / ((y + 1)**2 + y**2) - (x - 1) / ((x - 1)**2 + y**2),
                      y / ((y + 1)**2 + x**2) - y / ((x - 1)**2 + y**2)
                      ))
        return 100 * v


class CurlField(BaseField):
    ''' CurlField is a compostion of spiral fields produced by sources.
        `sources` is a tuple of tuples with coordinates of some 'particles'
        and direction of spiral (in radians) relative to source position '''
    sources = ((np.array((.1, .2)), np.pi/2),
               (np.array((.1, .9)), np.pi/6),
               (np.array((.7, .9)), 0))

    def forces(self, p):
        return [( (p - src) @ rotation_matrix_by_angle(angle) )
                    / (la.norm(p - src) ** 2 + 0.0001)
                for src, angle in self.sources]

    def __init__(self, sources=None):
        if sources:
            self.sources = sources

    def _vector(self, p):
        return sum(self.forces(p), np.array([0, 0]))
    
    def point_close_to_source(self, p):
        for src, _ in self.sources:
            if la.norm(src - p) < .005:
                return True
        return False


class DivCurlField(BaseField):
    ''' this was initial version of CurlField '''
    
    sources = ((np.array((.6, .2)), 'curl', 1),
               (np.array((.2, .9)), 'div', .5))

    def forces(self, p):
        rotation = {
            'curl': np.array([[0, -1], [1, 0]]),
            'div': np.identity(2),
        }
        return [( (p - src) @ rotation[_type] * mass )
                    / ( la.norm(p - src)**2 + .001 )
                for src, _type, mass in self.sources]

    def __init__(self, sources=None):
        if sources:
            self.sources = sources

    def _vector(self, p):
        return sum(self.forces(p), np.array([0, 0]))


fields = (ExpandingField,
          CosineField,
          TwoCuspsField,
          DipoleField,
          DivCurlField,
          CurlField,
          )


def preview_flow(field, n_traces=100, trace_segments=15,
                 dv=.01, dots=False, starting_points=None, subplot=None):
    if not subplot:
        _, subplot = plt.subplots()
    setup_empty_subplot(subplot, field.__class__.__name__)
    if not starting_points:
        starting_points = np.random.rand(n_traces, 2) - np.array((.5, .5))
    traces = field.compute_traces(starting_points, trace_segments, dv=dv)
    for trace in traces:
        subplot.plot(*trace.T, color='grey')
    if dots:
        subplot.scatter(*traces[:,0,:].T, color='black', s=3)


def setup_empty_subplot(subplot, title=None):
    subplot.axis('equal')
    subplot.set_title(title)
