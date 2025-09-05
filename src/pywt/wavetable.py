from collections.abc import Collection, Iterator, Sequence
from typing import Protocol, NamedTuple, SupportsIndex, overload

import numpy as np


class HasSampleRate(Protocol):
    @property
    def samplerate(self) -> int: ...


class Sized(Protocol):
    @property
    def size(self) -> int: ...


class Shaped[*Ts](Protocol):
    @property
    def shape(self) -> tuple[*Ts]: ...


class PlotTimeDomain(Protocol):
    def plot_time_domain(self) -> tuple[np.ndarray, np.ndarray]: ...


class PlotFreqDomain(Protocol):
    def plot_freq_domain(self) -> tuple[np.ndarray, np.ndarray]: ...


class WavetableFrame(PlotTimeDomain, PlotFreqDomain, HasSampleRate, Sized, Shaped[int], Protocol):
    @property
    def n_partials(self) -> int: ...


class WavetableFrames(HasSampleRate, Shaped[int, int], Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[WavetableFrame]: ...

    def __contains__(self, item: WavetableFrame) -> bool: ...

    @overload
    def __getitem__(self, index: SupportsIndex) -> WavetableFrame: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[WavetableFrame]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> WavetableFrame | Sequence[WavetableFrame]: ...


class SupportsFrames(Protocol):
    def frames(self) -> WavetableFrames: ...

    @property
    def number_of_frames(self) -> int:
        return len(self.frames())

    @property
    def frame_size(self) -> int: ...


class Wavetable(PlotTimeDomain, SupportsFrames, HasSampleRate, Sized, Protocol):
    @property
    def n_partials(self) -> int: ...


class NDArrayWavetableFrame(NamedTuple):
    frame_data: np.ndarray
    fs: int

    def plot_time_domain(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(self.frame_data.size, dtype=np.float64)
        t /= self.fs
        return t, self.frame_data

    def plot_freq_domain(self) -> tuple[np.ndarray, np.ndarray]:
        fft = np.fft.rfft(self.frame_data)
        return np.abs(fft), np.angle(fft)

    @property
    def n_partials(self):
        return self.size // 2 + 1

    @property
    def samplerate(self) -> int:
        return self.fs

    @property
    def size(self):
        return self.frame_data.size

    @property
    def shape(self) -> tuple[int]:
        return self.frame_data.shape


class NDArrayWavetableFrames:
    def __init__(self, _wavetable: "NDArrayWavetable", /):
        self._wavetable = _wavetable

    def __len__(self) -> int:
        return self.number_of_frames

    def __iter__(self) -> Iterator[NDArrayWavetableFrame]:
        return (self._get_frame(idx) for idx in range(len(self)))

    def _get_frame(self, idx: int, /) -> NDArrayWavetableFrame:
        initial_idx = self.frame_size * idx
        return NDArrayWavetableFrame(
            frame_data=self._wavetable.wt_data[initial_idx:initial_idx + self.frame_size],
            fs=self._wavetable.fs)

    @overload
    def __getitem__(self, index: SupportsIndex) -> NDArrayWavetableFrame:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[NDArrayWavetableFrame]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self._get_frame(i) for i in range(index.start, index.stop, index.step))
        elif hasattr(index, '__index__'):
            return self._get_frame(index.__index__())
        raise ValueError(f'Invalid index: {index}')

    def __contains__(self, item: WavetableFrame) -> bool:
        # TODO
        return False

    @property
    def samplerate(self) -> int:
        return self._wavetable.fs

    @property
    def shape(self) -> tuple[int, int]:
        return self.number_of_frames, self.frame_size

    @property
    def frame_size(self) -> int:
        return self._wavetable.frame_size

    @property
    def number_of_frames(self) -> int:
        return self._wavetable.number_of_frames


class NDArrayWavetable(NamedTuple):
    wt_data: np.ndarray
    n_frames: int
    fs: int

    def __len__(self) -> int:
        return self.size

    @property
    def samplerate(self) -> int:
        return self.fs

    @property
    def size(self):
        return self.wt_data.size

    @property
    def number_of_frames(self) -> int:
        return self.n_frames

    @property
    def frame_size(self) -> int:
        return self.wt_data.size // self.number_of_frames

    @property
    def n_partials(self) -> int:
        return self.frame_size // 2 + 1

    def frames(self) -> NDArrayWavetableFrames:
        return NDArrayWavetableFrames(self)

    def plot_time_domain(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(self.wt_data.size, dtype=np.float64)
        t /= self.fs
        return t, self.wt_data


def wavetable[T](_wt_raw: Collection[T], /, n_frames: int = 256, fs: int = 96000) -> Wavetable:
    return NDArrayWavetable(wt_data=np.array(_wt_raw, dtype=np.float64), n_frames=n_frames, fs=fs)
