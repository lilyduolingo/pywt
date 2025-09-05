from abc import ABC
from pathlib import Path
from typing import NamedTuple, TypedDict, Self

import soundfile as sf  # type: ignore


class ParseArgs(NamedTuple):
    args: tuple[str, ...]
    kwargs: dict[str, str | tuple[str, ...]]


def _parse_args(*args: str) -> ParseArgs:
    arguments = []
    options: dict[str, str | tuple[str, ...]] = {}

    separator_encountered = False
    current_key: str | None = None
    for arg in args:
        match arg, separator_encountered, current_key:
            case '--', False, _:
                separator_encountered = True
            case arg, False, _ if arg.startswith('--'):
                current_key = arg[2:]
            case arg, _, _ if (separator_encountered or not arg.startswith('--')):
                arguments.append(arg)
            case arg, False, curr_key if curr_key is not None:
                match options.get(curr_key):
                    case None:
                        options[curr_key] = arg
                    case (val, *vals):
                        options[curr_key] = (arg, val, *vals)
                    case val:
                        options[curr_key] = (arg, val)
            case _, _, _:
                raise ValueError(f"Parse error: {arg}")

    return ParseArgs(arguments, options)


class BaseOptions(TypedDict, total=False):
    ...


class Command[*Ts](ABC):
    @classmethod
    @abstractmethod
    def new(cls, *args: *Ts) -> Self: ...

    @abstractmethod
    def __call__(self, **kwargs: BaseOptions) -> None: ...


class ViewOptions(BaseOptions, TypedDict):
    ...


class View(NamedTuple):
    file_path: Path
    n_frames: int

    @classmethod
    def new(cls, _file_path: str | Path, _n_frames: int, /) -> Self:
        return cls(file_path=Path(_file_path), n_frames=_n_frames)

    def __call__(self, **kwargs: ViewOptions) -> None:
        wt_raw, fs = sf.read(self.file_path)
        wt = wavetable(wt_raw, n_frames=int(self.n_frames), fs=fs)
        view_wavetable(wt)
